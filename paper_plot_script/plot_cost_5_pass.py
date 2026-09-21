#!/usr/bin/env python3
"""What one pass costs: the three manager arms against Fable 5, LCB-100 x 5 passes."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import palette

from plot_16k_reason_off_5_pass import (
    ALPHA, CI_LW, EDGE_LW, FIGSIZE, FS_BODY, FS_HEAD, FS_NOTE, FS_TITLE,
    MARGINS, PLOTS, THEMES,
    apply_theme, below_panel, boot_ci, fmt_p_num, holm, model_block, pass_ci,
    perm_sign_p, ring, slug, wrap_title, write_figure,
)
FILLS = {f"{k}_{arm}": palette.FILLS[k][i]
         for k in ("q38", "q38fn", "luna", "terra", "dsv41", "fable")
         for i, arm in ((0, "single"), (1, "multi"))
         if not (k == "fable" and arm == "multi")}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R4 = f"{ROOT}/runs/4models-1pass-reason-on/results"
FN = f"{ROOT}/runs/q38-fn-5pass/results"
DS = f"{ROOT}/runs/ds-v41-f-5pass/results"
TOKENS = os.path.join(ROOT, "runs/per_problem_tokens.json")
PASSES = [1, 2, 3, 4, 5]

ARMS = {
    "q38_single": ("Qwen3.8-27B, single call",
                   f"{R4}/q38_single_p%d.cap128k.patched.json", (0.214, 2.550)),
    "q38_multi": ("Qwen3.8-27B, with manager",
                  f"{R4}/q38_multiagent_p%d.patched.json", (0.214, 2.550)),
    "q38fn_single": ("Qwen3.8-Flash-Next, single call",
                     f"{FN}/q38_fn_single_p%d.patched.json", (0.150, 0.470)),
    "q38fn_multi": ("Qwen3.8-Flash-Next, with manager",
                    f"{FN}/q38_fn_multiagent_p%d.patched.json", (0.150, 0.470)),
    "dsv41_single": ("DeepSeek-V4.1-Flash, single call",
                     f"{DS}/ds_v41_f_single_p%d.patched.json", (0.30, 1.20)),
    "dsv41_multi": ("DeepSeek-V4.1-Flash, with manager",
                    f"{DS}/ds_v41_f_multiagent_p%d.patched.json", (0.30, 1.20)),
    "luna_single": ("GPT-5.6-Luna, single call",
                    f"{R4}/luna_single_p%d.patched.json", (0.20, 1.20)),
    "luna_multi": ("GPT-5.6-Luna, with manager",
                   f"{R4}/luna_multiagent_p%d.patched.json", (0.20, 1.20)),
    "terra_single": ("GPT-5.6-Terra, single call",
                     f"{R4}/terra_single_p%d.patched.json", (2.0, 12.0)),
    "terra_multi": ("GPT-5.6-Terra, with manager",
                    f"{R4}/terra_multiagent_p%d.patched.json", (2.0, 12.0)),
    "fable_single": ("Fable 5, single call",
                     f"{ROOT}/runs/fable5-5pass-single/results/fable5_single_p%d.patched.json",
                     (10.0, 50.0)),
}
CAP = 128_000
# The Qwen3.8-27B rate, quoted in the rate-sensitivity note below as well as in ARMS.
QWEN_RATE = (0.214, 2.550)

SOURCES = ("List rates: Qwen3.8-27B \\citep{openrouter2026}, Qwen3.8-Flash-Next "
           "\\citep{openrouter2026flash}, "
           "GPT-5.6-Luna and GPT-5.6-Terra \\citep{openai2026price}, "
           "DeepSeek-V4.1-Flash \\citep{deepseek2026price}, "
           "Fable~5 \\citep{anthropic2026price}. OpenAI uses short-context rates "
           "and Anthropic and DeepSeek use the base rates; no long-context tier, batch/off-peak "
           "discount, or prompt-caching multiplier is applied. Qwen and DeepSeek rates were read on 2026-09-15.")

TOK_KEY = {"q38_single": "q38_single"}

TESTS = [
    ("q38_single", "q38_multi", "Qwen3.8-27B: manager $-$ single"),
    ("q38fn_single", "q38fn_multi", "Qwen3.8-Flash-Next: manager $-$ single"),
    ("dsv41_single", "dsv41_multi", "DeepSeek-V4.1-Flash: manager $-$ single"),
    ("luna_single", "luna_multi", "GPT-5.6-Luna: manager $-$ single"),
    ("terra_single", "terra_multi", "GPT-5.6-Terra: manager $-$ single"),
    ("q38_multi", "fable_single", "Fable 5 single $-$ Qwen3.8-27B manager"),
    ("q38fn_multi", "fable_single", "Fable 5 single $-$ Qwen3.8-Flash-Next manager"),
    ("dsv41_multi", "fable_single", "Fable 5 single $-$ DeepSeek-V4.1-Flash manager"),
    ("luna_multi", "fable_single", "Fable 5 single $-$ GPT-5.6-Luna manager"),
    ("terra_multi", "fable_single", "Fable 5 single $-$ GPT-5.6-Terra manager"),
    ("luna_multi", "terra_single", "GPT-5.6-Terra single $-$ GPT-5.6-Luna manager"),
]

RATE_RANGE = ((0.33, 2.40), (0.45, 3.20))

BAR_W = 0.38
PITCH = 1.5
# The old right-hand column of significance marks sat at x = 82; with it gone the axis only
# has to clear Fable 5's bar at $61.11 plus its price and pass@1 labels.
XLIMC = (0, 80)
XTICKS = [0, 20, 40, 60]

TITLE = ("What one pass costs — LCB-100, 5 passes, "
         "single call vs manager, against Fable 5")

COST_MARGINS = dict(MARGINS, left=0.025, right=0.98, top=0.842, bottom=0.206)
LEGEND_Y = 0.015
FIGSIZE_H = (FIGSIZE[0], FIGSIZE[0] * 0.86)

CAPTION = ("\\textbf{The scaffold's bill.} Cost of one pass over the same 100 problems; "
           "Table~\\ref{tab:cost} is the arithmetic behind every bar. No cached-input "
           "discount is "
           "taken, and Qwen3.8-27B is priced at OpenRouter market rates. The top bar of "
           "each pair is the single call, light; the manager is under it, dark; all are at "
           "a 128k output cap, and hatch is the model. Models run top to bottom from the "
           "cheapest single call to the dearest, with Fable 5 last. "
           "Table~\\ref{tab:cost} carries the tests.")


def money(v):
    return f"\\${v:.2f}" if v >= 0.1 else f"\\${v:.3f}"


def notes(stats):
    # The per-comparison p-values live in Table~\ref{tab:cost} now, not here.
    a, t = stats["arms"], stats["tests"]
    shown = lambda k: round(a[k]["mean"], 2)
    cross = next(x for x in t if x["a"] == "luna_multi" and x["b"] == "terra_single")
    return [
        f"Unbracketed comparisons: every manager $-$ single increase is significant, and "
        f"GPT-5.6-Luna's manager undercuts GPT-5.6-Terra's single call by "
        f"\\${shown(cross['b']) - shown(cross['a']):.2f}.",
        f"At {stats['crossover']:.2f}x the assumed Qwen rate "
        f"(\\${QWEN_RATE[0] * stats['crossover']:.3f}/"
        f"\\${QWEN_RATE[1] * stats['crossover']:.2f} per MTok, inside the spread across hosted "
        f"providers) the manager's cost advantage over Fable 5 disappears entirely — a "
        f"larger uncertainty than any p-value here.",
    ]

# --------------------------------------------------------------------------- data

def compute():
    tok = json.load(open(TOKENS))
    arms, qids = {}, None
    for key, (label, pattern, (ri, ro)) in ARMS.items():
        a = np.array(tok[key]["tokens"], float)
        assert qids is None or tok[key]["qids"] == qids, f"{key}: different problems"
        qids = tok[key]["qids"]
        if key == "q38_single":
            a = a.copy()
            a[:, :, 1] = np.minimum(a[:, :, 1], CAP)
            a[:, :, 3] = np.minimum(a[:, :, 3], CAP)
        tin, tout = a[:, :, 0] + a[:, :, 2], a[:, :, 1] + a[:, :, 3]
        cost = (tin * ri + tout * ro) / 1e6
        passed = np.array([[bool(r["passed"]) for r in
                            json.load(open(pattern % p))["lcb"]["records"]] for p in PASSES])
        per_pass = cost.sum(axis=1)
        arms[key] = dict(
            key=key, label=label, rate=(ri, ro), cost=cost, per_pass=per_pass,
            mean=per_pass.mean(), ci=pass_ci(per_pass),
            mtok_in=tin.sum(axis=1).mean() / 1e6, mtok_out=tout.sum(axis=1).mean() / 1e6,
            per_problem=cost.mean(axis=0),
            acc=100 * passed.mean(), per_solve=cost.sum() / passed.sum(),
        )
        if key == "q38_single":
            # What the same generations would have billed at the 250k cap they were made
            # under, before the 128k cap-match this table prices them at.
            raw = np.array(tok[key]["tokens"], float)
            arms[key]["cost_asgen"] = (
                ((raw[:, :, 0] + raw[:, :, 2]) * ri
                 + (raw[:, :, 1] + raw[:, :, 3]) * ro) / 1e6).sum(axis=1).mean()

    from scipy import stats as sps
    tests = []
    for ka, kb, name in TESTS:
        x, y = arms[ka]["per_pass"], arms[kb]["per_pass"]
        welch = sps.ttest_ind(y, x, equal_var=False)
        d = arms[kb]["per_problem"] - arms[ka]["per_problem"]
        delta, p_prob, floored = perm_sign_p(d)
        tests.append(dict(a=ka, b=kb, name=name, delta_pass=y.mean() - x.mean(),
                          delta_prob=delta, prob_ci=boot_ci(d),
                          p_pass=welch.pvalue, p_prob=p_prob, floored=floored))
    for tst, ph in zip(tests, holm([t["p_pass"] for t in tests])):
        tst["p_pass_holm"] = ph
    for tst, ph in zip(tests, holm([t["p_prob"] for t in tests])):
        tst["p_prob_holm"] = ph
        tst["sig"] = ph < ALPHA

    crossover = arms["fable_single"]["cost"].sum() / arms["q38_multi"]["cost"].sum()
    return dict(arms=arms, tests=tests, crossover=crossover)


# The rate header is the widest cell in its column; stacking it recovers the ~26pt the two
# new p columns cost.
RATE_HEADER = ("Arm", "\\shortstack{Rate \\$/MTok\\\\in / out}", "In (MTok)", "Out (MTok)",
               "\\$/pass", "\\$/solved",
               "\\shortstack{$p$ vs\\\\single}", "\\shortstack{$p$ vs\\\\Fable 5}")
RATE_SPEC = ("@{}l@{\\hspace{0.5em}}l@{\\hspace{0.5em}}r@{\\hspace{0.5em}}r"
             "@{\\hspace{0.5em}}r@{\\hspace{0.5em}}r@{\\hspace{0.5em}}c"
             "@{\\hspace{0.5em}}c@{}")

TABLE_HEADER = ("Comparison", "$\\Delta$ \\$/pass",
                "$p$ (per pass, $n=5$)", "$p$ (per problem, $n=100$)")
TABLE_SPEC = "@{}l@{\\hspace{1em}}c@{\\hspace{1em}}c@{\\hspace{1em}}c@{}"


def fmt_p_tex(p, floored=False):
    lt = "<" if floored else ""
    if p >= 1e-3:
        return f"${lt}{p:#.2g}$" if lt else f"{p:#.2g}"
    mantissa, exponent = ("%.1e" % p).split("e")
    return f"${lt}{mantissa}\\times10^{{{int(exponent)}}}$"


def rate_tex(v):
    return f"\\${v:g}" if v == int(v) else f"\\${v:.2f}"


def arm_short(key, arm):
    # Model over arm rather than side by side: on one line the longest names
    # ("DeepSeek-V4.1-Flash manager") set this column wide enough to push the table past
    # \linewidth. Stacked, the column is only as wide as the longest model name.
    return ("\\shortstack[l]{%s\\\\%s}"
            % (arm["label"].split(",")[0],
               "single" if key.endswith("single") else "manager"))


def rate_rows(stats):
    def p_of(a, b):
        tst = next((t for t in stats["tests"] if (t["a"], t["b"]) == (a, b)), None)
        return "---" if tst is None else fmt_p_tex(tst["p_pass_holm"])

    rows = []
    for key, arm in stats["arms"].items():
        ri, ro = arm["rate"]
        mk, armname = key.rsplit("_", 1)
        if key == "fable_single":
            p_single, p_fable = "---", "reference"
        elif armname == "multi":
            p_single, p_fable = p_of(f"{mk}_single", key), p_of(key, "fable_single")
        else:
            p_single, p_fable = "---", "---"
        rows.append((arm_short(key, arm), f"{rate_tex(ri)} / {rate_tex(ro)}",
                     f"{arm['mtok_in']:.4f}", f"{arm['mtok_out']:.4f}",
                     money(arm["mean"]), money(arm["per_solve"]), p_single, p_fable))
    return rows


def table_rows(stats):
    def shown(key):
        return round(stats["arms"][key]["mean"], 2)
    return [(t["name"], f"{shown(t['b']) - shown(t['a']):+.2f}",
             fmt_p_tex(t["p_pass_holm"]), fmt_p_tex(t["p_prob_holm"], t["floored"]))
            for t in stats["tests"]]


def rate_caption(stats):
    arms = stats["arms"]
    q = arms["q38_single"]
    cheap_k, cheap = min(arms.items(), key=lambda kv: kv[1]["mean"])
    best_k, best = max(arms.items(), key=lambda kv: kv[1]["acc"])
    return (
        "\\textbf{What one pass cost each arm.} List rate $\\times$ the tokens it "
        "consumed. The two $p$ columns test each manager arm against its own single call "
        "and against Fable 5's single call: Welch over the five passes, Holm-corrected "
        f"across {len(TESTS)} comparisons. Qwen3.8-27B's single arm is the 128k cap-matched "
        f"one; at 250k it costs {money(q['cost_asgen'])}/pass. "
        f"Retried and discarded attempts are counted: they "
        "were generated and would be billed. "
        f"The cheapest arm is {arms[cheap_k]['label'].split(',')[0]} "
        f"{'manager' if cheap_k.endswith('multi') else 'single'} at "
        f"{money(cheap['mean'])} a pass and the most accurate is "
        f"{arms[best_k]['label'].split(',')[0]} "
        f"{'manager' if best_k.endswith('multi') else 'single'} at "
        f"{money(best['mean'])} --- a {best['mean'] / cheap['mean']:.0f}$\\times$ spread in "
        f"price for {best['acc'] - cheap['acc']:+.1f} points."
    )


TABLES = [(RATE_HEADER, RATE_SPEC, rate_rows, rate_caption)]


# --------------------------------------------------------------------------- plot

def draw(stats, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    fig, ax = plt.subplots(figsize=FIGSIZE_H)
    fig.subplots_adjust(**COST_MARGINS)
    arms = stats["arms"]

    models = sorted((k.rsplit("_", 1)[0] for k in arms if k.endswith("_single")),
                    key=lambda mk: (mk == "fable", arms[f"{mk}_single"]["mean"]))
    X = {}
    for i, mk in enumerate(models):
        y = i * PITCH
        paired = f"{mk}_multi" in arms
        X[f"{mk}_single"] = y - BAR_W / 2 if paired else y
        if paired:
            X[f"{mk}_multi"] = y + BAR_W / 2

    ax.set(xlim=XLIMC, ylim=(max(X.values()) + 0.62 + BAR_W / 2, -1.55))
    ax.set_yticks([])
    ax.set_xlabel("Cost of one pass (USD)", fontsize=FS_BODY, color=t["ink2"],
                  loc="left")
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    ax.set_xticks(XTICKS, [f"\\${v:g}" for v in XTICKS])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    for mk in models:
        ax.annotate(arms[f"{mk}_single"]["label"].split(",")[0],
                    xy=(0, X[f"{mk}_single"] - BAR_W / 2), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=FS_BODY, fontweight="bold", color=t["ink"])

    for key, arm in arms.items():
        y, v, (lo, hi) = X[key], arm["mean"], arm["ci"]
        mk, armname = key.rsplit("_", 1)
        bar = palette.bar_kw(mk, "manager" if armname == "multi" else "single",
                             surface=t["surface"])
        bar.setdefault("edgecolor", ring(FILLS[key], theme))
        ax.barh(y, v, BAR_W, linewidth=EDGE_LW, zorder=3, **bar)
        ax.plot([lo, hi], [y, y], lw=CI_LW, color=t["ink"], solid_capstyle="butt", zorder=5)
        for end in (lo, hi):
            ax.plot([end], [y], marker="|", ms=10, mew=CI_LW, color=t["ink"], zorder=5)
        price = money(v)
        ax.annotate(price, xy=(hi, y), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS_BODY, color=t["ink"])
        ax.annotate(f"· {arm['acc']:.1f}%", xy=(hi, y),
                    xytext=(5 + 0.62 * FS_BODY * (len(price) - 1), 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS_NOTE, color=t["muted_text"])

    ax.annotate("Same 100 problems, 5 passes per condition; bars are the mean pass,\n"
                "CI across the 5; the figure after each bar is its pass@1.",
                xy=(1, 0), xycoords="axes fraction", xytext=(0, -30),
                textcoords="offset points", ha="right", va="top", linespacing=1.35,
                fontsize=FS_NOTE, color=t["muted_text"])

    pale, deep = ("#d5d4cd", "#6f6d67") if theme == "light" else ("#b8b6ae", "#5e5c57")

    def patch(colour):
        return Patch(facecolor=colour, edgecolor=ring(colour, theme), linewidth=EDGE_LW)

    fig.suptitle(wrap_title(TITLE), x=MARGINS["left"], ha="left", y=0.99,
                 va="top", fontsize=FS_TITLE, fontweight="bold", color=t["ink"],
                 linespacing=1.25)
    leg = fig.legend([patch(pale), patch(deep)], ["single call", "with manager"],
                     loc="lower center", bbox_to_anchor=(0.5, LEGEND_Y), ncol=2,
                     frameon=False, fontsize=FS_BODY, labelcolor=t["ink2"],
                     handletextpad=0.7, columnspacing=2.8)
    fig.add_artist(leg)
    if save:
        write_figure(fig, save)
    plt.close(fig)


def main():
    stats = compute()

    hdr = (f"{'arm':26s} {'$/pass':>8s} {'95% CI':>16s} {'5 passes':>9s} "
           f"{'pass@1':>7s} {'$/solve':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for arm in stats["arms"].values():
        print(f"{arm['label']:26s} {arm['mean']:8.2f}   [{arm['ci'][0]:5.2f},{arm['ci'][1]:6.2f}]"
              f" {arm['cost'].sum():9.2f} {arm['acc']:7.1f} {arm['per_solve']:8.3f}")

    print(f"\nsame difference, two units of analysis "
          f"(both Holm-corrected across the {len(TESTS)}):")
    for t in stats["tests"]:
        pre = "<" if t["floored"] else " "
        print(f"  {t['name'].replace('$-$', '-'):32s} {t['delta_pass']:+7.2f} $/pass"
              f"  ({t['delta_prob']:+.3f} $/problem, CI [{t['prob_ci'][0]:+.3f},"
              f"{t['prob_ci'][1]:+.3f}])")
        print(f"  {'':32s} per-pass Welch p {t['p_pass']:.2e} -> Holm {t['p_pass_holm']:.2e}"
              f"   per-problem paired p {pre}{t['p_prob']:.2e} -> Holm {t['p_prob_holm']:.2e}")
    print(f"\n  manager cost = Fable 5 cost at {stats['crossover']:.3f}x the assumed Qwen rate"
          f"  (${0.35 * stats['crossover']:.3f}/${2.75 * stats['crossover']:.2f} per MTok)")

    os.makedirs(PLOTS, exist_ok=True)
    for theme in ("light",):
        draw(stats, theme, save=os.path.join(PLOTS, f"{slug(TITLE)}_{theme}.pdf"))


if __name__ == "__main__":
    main()
