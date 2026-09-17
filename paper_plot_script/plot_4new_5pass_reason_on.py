#!/usr/bin/env python3
"""Single call vs manager for the four pinned-backend models, LCB-100 x 5 passes at 128k, ON."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_16k_reason_off_5_pass import (
    CI_LW, EDGE_LW, FIGSIZE, FS_BODY, FS_SUB, FS_TITLE,
    MARGINS, PLOTS, THEMES,
    apply_theme, fmt_p_num, holm, pass_ci, perm_sign_p, ring, slug, wrap_title, write_figure,
)
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import binomtest, t as t_dist

import palette

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PASSES = [1, 2, 3, 4, 5]
R4 = f"{ROOT}/runs/4models-1pass-reason-on/results"
RF = f"{ROOT}/runs/fable5-5pass-single/results"
FN = f"{ROOT}/runs/q38-fn-5pass/results"
DS = f"{ROOT}/runs/ds-v41-f-5pass/results"

MODELS = [
    ("fable", "Claude Fable 5", f"{RF}/fable5_single_p%d.patched.json", None),
    ("terra", "GPT-5.6-Terra", f"{R4}/terra_single_p%d.patched.json",
                               f"{R4}/terra_multiagent_p%d.patched.json"),
    ("luna",  "GPT-5.6-Luna",  f"{R4}/luna_single_p%d.patched.json",
                               f"{R4}/luna_multiagent_p%d.patched.json"),
    ("q38",   "Qwen3.8-27B",   f"{R4}/q38_single_p%d.cap128k.patched.json",
                               f"{R4}/q38_multiagent_p%d.patched.json"),
    ("q38fn", "Qwen3.8-Flash-Next", f"{FN}/q38_fn_single_p%d.patched.json",
                                    f"{FN}/q38_fn_multiagent_p%d.patched.json"),
    ("dsv41", "DeepSeek-V4.1-Flash", f"{DS}/ds_v41_f_single_p%d.patched.json",
                                     f"{DS}/ds_v41_f_multiagent_p%d.patched.json"),
]

Q38_ASGEN = f"{R4}/q38_single_p%d.patched.json"

FILLS = {k: palette.FILLS[k] for k in ("q38", "q38fn", "luna", "terra", "dsv41", "fable")}
FABLE_DARK = FILLS["fable"][1]

TITLE = ("Manager vs single call, six models "
         "— LCB-100, 5 passes, 128k max tokens, reasoning ON")

CAPTION = ("\\textbf{Manager vs.\\ single call.} The scores from Table~\\ref{tab:main} in a graph.")

# Six full model names will not fit on one row of ticks -- at four they did, at six
# "Qwen3.8-27B", "Qwen3.8-Flash-Next" and "DeepSeek-V4.1-Flash" run into each other. These
# are the same names broken over two lines, as in plot_v2_20260915.py.
XLABELS = {
    "fable": "Claude\nFable 5",
    "terra": "GPT-5.6\nTerra",
    "luna": "GPT-5.6\nLuna",
    "q38": "Qwen3.8\n27B",
    "q38fn": "Qwen3.8\nFlash-Next",
    "dsv41": "DeepSeek\nV4.1-Flash",
}

BAR_W = 0.38
PITCH = 1.0
# Just clears the tallest value label (Qwen3.8-27B's manager CI tops out at 94.7); the axis
# still ticks to 100. Anything higher is empty band between the title and the bars.
YLIM = 103

FIGSIZE_V = (FIGSIZE[0], FIGSIZE[0] * 0.52)
M4 = dict(MARGINS, left=0.062, right=0.99, top=0.865, bottom=0.20)
LEG_Y = 0.015


def notes(stats):
    # The manager-only/single-only problem-pass counts are Table~\ref{tab:agreement} in the
    # appendix now; make_arm_tables.py builds it from the stats dicts compute() returns.
    return [
        "Bars are pass@1 on the same 100 problems, the line through each the 95% CI across "
        "the 5 passes (t, df = 4). The left bar of each pair is the single call, light; the "
        "manager is beside it, dark; hatch is the model. Fable 5 ran single-only, so it has one bar, which the "
        "dashed rule carries across the chart."
    ]


# --------------------------------------------------------------------------- data

def load_arm(pattern):
    qids, rows = None, []
    for p in PASSES:
        recs = json.load(open(pattern % p))["lcb"]["records"]
        ids = [r["question_id"] for r in recs]
        if qids is None:
            qids = ids
        assert ids == qids, f"{pattern % p}: id drift"
        rows.append([bool(r["passed"]) for r in recs])
    return qids, np.array(rows, float)


def compute():
    stats, qids0 = [], None
    for key, label, s_pat, m_pat in MODELS:
        qids, s = load_arm(s_pat)
        if qids0 is None:
            qids0 = qids
        assert qids == qids0, f"{key}: different problem set"
        st = dict(key=key, label=label, has_mgr=m_pat is not None,
                  single=100 * s.mean(), single_scores=100 * s.mean(axis=1),
                  single_ci=pass_ci(100 * s.mean(axis=1)),
                  single_prob=s.mean(axis=0))
        if key == "q38":
            _, g = load_arm(Q38_ASGEN)
            st["single_asgen"] = 100 * g.mean()
        if m_pat is not None:
            _, m = load_arm(m_pat)
            st.update(multi=100 * m.mean(), multi_scores=100 * m.mean(axis=1),
                      multi_ci=pass_ci(100 * m.mean(axis=1)),
                      multi_prob=m.mean(axis=0),
                      delta=100 * (m.mean() - s.mean()))
            d = m.mean(axis=0) - s.mean(axis=0)
            _, p, floored = perm_sign_p(d)
            st.update(p_raw=p, floored=floored)
            mgr_only = int(((m == 1) & (s == 0)).sum())
            sgl_only = int(((m == 0) & (s == 1)).sum())
            st.update(mgr_only=mgr_only, sgl_only=sgl_only, n_pp=int(m.size),
                      mcnemar_p=binomtest(mgr_only, mgr_only + sgl_only, 0.5).pvalue)
        stats.append(st)
    tested = [s for s in stats if s["has_mgr"]]
    for s, p in zip(tested, holm([s["p_raw"] for s in tested])):
        s["p_holm"] = p

    ref = next(s for s in stats if s["key"] == "fable")["single_prob"]
    for s in tested:
        d = s["multi_prob"] - ref
        obs, p, floored = perm_sign_p(d)
        s.update(vs_fable=100 * obs, vs_p_raw=p, vs_floored=floored)
    for s, p in zip(tested, holm([s["vs_p_raw"] for s in tested])):
        s["vs_p_holm"] = p

    # One-sided 95% lower bound on manager - Fable 5: how large a deficit the data leave open,
    # where the permutation test above only answers whether there is a gap at all. Built on the
    # same per-problem differences that test uses, so the two cannot disagree -- a bound that
    # clears 0 and a p above .05 would be a contradiction. Pairing over the five passes instead
    # would shrink the SE from 1.67 to 0.45 by holding the problem set fixed, which drops the
    # dominant variance component, and its pass-to-pass pairing is arbitrary besides: the two
    # arms are independent runs, and reordering one arm's passes moves the bound across 0.
    for s in tested:
        d = (s["multi_prob"] - ref) * 100
        s["vs_fable_lo"] = (d.mean()
                            - t_dist.ppf(0.95, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size))
    return stats


# --------------------------------------------------------------------------- table

def score_tex(mean, ci):
    return "$%.1f \\pm %.1f$" % (mean, (ci[1] - ci[0]) / 2)


def p_tex(p, floored=False):
    """fmt_p_num renders its exponent in unicode superscripts; tabular() does not run
    the caption escaper over table cells, so the same number is written as math here."""
    lead = "<" if floored else ""
    if p >= 1e-3:
        return "$%s%.2g$" % (lead, p)
    mantissa, exponent = ("%.1e" % p).split("e")
    return "$%s%s\\times10^{%d}$" % (lead, mantissa, int(exponent))


def stack(top, bottom):
    return "\\shortstack{%s\\\\%s}" % (top, bottom)


# Eight columns of one-line headers run 12pt past \linewidth, and \footnotesize cannot buy
# that back -- iclr2027_conference.sty defines it as \small. Stacking the two-word headers
# over two lines does, and keeps the body at a readable size.
MAIN_HEADER = ("Model", stack("Single", "call"), stack("With", "manager"),
               stack("$\\Delta$ vs", "single"), "$p$",
               stack("$\\Delta$ vs", "Fable 5"), stack("95\\%", "bound"), "$p$")
MAIN_SPEC = ("@{}l" + "@{\\hspace{0.55em}}c" * 2 + "@{\\hspace{0.55em}}r@{\\hspace{0.55em}}c"
             + "@{\\hspace{0.55em}}r@{\\hspace{0.55em}}r@{\\hspace{0.55em}}c@{}")


def main_rows(stats):
    rows = []
    for s in stats:
        if not s["has_mgr"]:
            rows.append([s["label"], score_tex(s["single"], s["single_ci"]),
                         "---", "---", "---", "reference", "---", "---"])
            continue
        rows.append([s["label"],
                     score_tex(s["single"], s["single_ci"]),
                     score_tex(s["multi"], s["multi_ci"]),
                     "$%+.1f$" % s["delta"], p_tex(s["p_holm"], s["floored"]),
                     "$%+.1f$" % s["vs_fable"], "$%+.1f$" % s["vs_fable_lo"],
                     p_tex(s["vs_p_holm"], s["vs_floored"])])
    return rows


TABLES = [(MAIN_HEADER, MAIN_SPEC, main_rows,
           "\\textbf{Manager vs.\\ single call, five passes.} pass@1 on LCB-100 at a 128k "
           "cap, reasoning on; $\\pm$ is a 95\\% $t$ interval across the five passes "
           "(df = 4). $\\Delta$ is in percentage points, against the model's own single "
           "call and against Fable 5's single call. Each $p$ is a paired sign-flip "
           "permutation test, unit = problem ($n = 100$), Holm-corrected within its family "
           "of three; $<$ marks the permutation floor. \\textbf{95\\% bound} is the "
           "one-sided lower bound on $\\Delta$ vs Fable 5 over the same per-problem "
           "differences (t, df = 99): the largest deficit the data leave open.")]


# --------------------------------------------------------------------------- plot

def draw(stats, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    stats = sorted(stats, key=lambda s: (s["key"] == "fable", s["single"]))
    fig, ax = plt.subplots(figsize=FIGSIZE_V)
    fig.subplots_adjust(**M4)
    n = len(stats)
    xs = [i * PITCH for i in range(n)]
    ax.set(xlim=(-0.62, xs[-1] + 0.62), ylim=(0, YLIM))
    ax.set_ylabel("Accuracy (pass@1, %)", fontsize=FS_BODY, color=t["ink2"],
                  loc="top")
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xticks(xs)
    ax.set_xticklabels([XLABELS.get(s["key"], s["label"]) for s in stats],
                       fontsize=FS_BODY, fontweight="bold", color=t["ink"],
                       linespacing=1.15)
    ax.tick_params(length=0, labelsize=FS_BODY)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    fable = next(s for s in stats if s["key"] == "fable")["single"]
    ax.axhline(fable, ls=(0, (6, 4)), lw=1.6, color=FABLE_DARK, alpha=0.85, zorder=2)

    for x, s in zip(xs, stats):
        light, dark = FILLS[s["key"]]
        arms = [("single", light, -BAR_W / 2 if s["has_mgr"] else 0.0)]
        if s["has_mgr"]:
            arms.append(("multi", dark, BAR_W / 2))
        for arm, fill, dx in arms:
            v, lo, hi = s[arm], *s[f"{arm}_ci"]
            xb = x + dx
            bar = palette.bar_kw(s["key"], "manager" if arm == "multi" else "single",
                                 surface=t["surface"])
            bar.setdefault("edgecolor", ring(fill, theme))
            ax.bar(xb, v, BAR_W, linewidth=EDGE_LW, zorder=3, **bar)
            ax.plot([xb, xb], [lo, hi], color=ring(fill, theme), lw=CI_LW, zorder=5,
                    solid_capstyle="round")
            for cap in (lo, hi):
                ax.plot([xb - 0.055, xb + 0.055], [cap, cap], color=ring(fill, theme),
                        lw=CI_LW, zorder=5)
            ax.annotate(f"{v:.1f}", xy=(xb, hi), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=FS_BODY, color=t["ink"], zorder=6,
                        bbox=dict(facecolor=t["surface"], edgecolor="none",
                                  boxstyle="square,pad=0.12"))

    def swatch(colour):
        return Line2D([], [], marker="o", ls="", ms=12, color=colour,
                      markeredgecolor=ring(colour, theme), markeredgewidth=EDGE_LW)

    pale, deep = ("#d5d4cd", "#6f6d67") if theme == "light" else ("#b8b6ae", "#5e5c57")
    key = [(swatch(pale), "single call"),
           (swatch(deep), "with manager"),
           (Line2D([], [], ls=(0, (6, 4)), lw=1.6, color=FABLE_DARK),
            f"Fable 5 single call, {fable:.1f}%")]

    fig.suptitle(wrap_title(TITLE), x=M4["left"], ha="left", y=0.99, va="top",
                 fontsize=FS_TITLE, fontweight="bold", color=t["ink"], linespacing=1.25)
    leg = fig.legend([h for h, _ in key], [n for _, n in key], loc="lower center",
                     bbox_to_anchor=(0.5, LEG_Y), ncol=len(key), frameon=False,
                     fontsize=FS_SUB, labelcolor=t["ink2"], handletextpad=0.7,
                     handlelength=2.2, columnspacing=2.8)
    fig.add_artist(leg)
    if save:
        write_figure(fig, save)
    return fig


def main():
    stats = compute()
    hdr = (f"{'model':16} {'single':>16} {'manager':>16} {'delta':>7} {'Holm p':>10}"
           f" {'vs Fable':>9} {'Holm p':>10}")
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        sci = f"{s['single']:5.1f} [{s['single_ci'][0]:4.1f},{s['single_ci'][1]:5.1f}]"
        if s["has_mgr"]:
            mci = f"{s['multi']:5.1f} [{s['multi_ci'][0]:4.1f},{s['multi_ci'][1]:5.1f}]"
            print(f"{s['label']:16} {sci:>16} {mci:>16} {s['delta']:+7.1f} "
                  f"{fmt_p_num(s['p_holm'], 2, s['floored']):>10} {s['vs_fable']:+9.1f} "
                  f"{fmt_p_num(s['vs_p_holm'], 2, s['vs_floored']):>10}")
        else:
            print(f"{s['label']:16} {sci:>16} {'- (single only)':>16} {'-':>7} {'-':>10} "
                  f"{'reference':>9} {'-':>10}")
    q38 = next(s for s in stats if s["key"] == "q38")
    print(f"\nQwen3.8-27B single as generated at 250k: {q38['single_asgen']:.1f} "
          f"({q38['single_asgen'] - q38['single']:+.1f} vs the 128k replay the chart uses)")

    os.makedirs(PLOTS, exist_ok=True)
    for theme in ("light",):
        draw(stats, theme,
             save=os.path.join(PLOTS, f"{slug(TITLE)}_bars_{theme}.pdf"))


if __name__ == "__main__":
    main()
