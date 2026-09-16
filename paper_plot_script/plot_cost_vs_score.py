#!/usr/bin/env python3
"""Cost against accuracy: all seven pinned-backend arms on one axis, LCB-100 x 5 passes, 128k."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import palette

from plot_16k_reason_off_5_pass import (
    DOT, EDGE_LW, FIGSIZE, FS_BODY, FS_NOTE, FS_TITLE, MARGINS, PLOTS, THEMES,
    apply_theme, below_panel, pass_ci, ring, slug, wrap_title, write_figure,
)
from plot_cost_5_pass import SOURCES, TOKENS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R4 = f"{ROOT}/runs/4models-1pass-reason-on/results"
RF = f"{ROOT}/runs/fable5-5pass-single/results"
FN = f"{ROOT}/runs/q38-fn-5pass/results"
DS = f"{ROOT}/runs/ds-v41-f-5pass/results"
PASSES = [1, 2, 3, 4, 5]
CAP = 128_000

ARMS = {
    "q38_single":   ("Qwen3.8-27B, single call",    f"{R4}/q38_single_p%d.cap128k.patched.json",  (0.214, 2.550), "q38", "single"),
    "q38_multi":    ("Qwen3.8-27B, with manager",   f"{R4}/q38_multiagent_p%d.patched.json",      (0.214, 2.550), "q38", "manager"),
    "q38fn_single": ("Qwen3.8-Flash-Next, single call",  f"{FN}/q38_fn_single_p%d.patched.json",     (0.150, 0.470), "q38fn", "single"),
    "q38fn_multi":  ("Qwen3.8-Flash-Next, with manager", f"{FN}/q38_fn_multiagent_p%d.patched.json", (0.150, 0.470), "q38fn", "manager"),
    "dsv41_single": ("DeepSeek-V4.1-Flash, single call",  f"{DS}/ds_v41_f_single_p%d.patched.json",     (0.30, 1.20), "dsv41", "single"),
    "dsv41_multi":  ("DeepSeek-V4.1-Flash, with manager", f"{DS}/ds_v41_f_multiagent_p%d.patched.json", (0.30, 1.20), "dsv41", "manager"),
    "luna_single":  ("GPT-5.6-Luna, single call",   f"{R4}/luna_single_p%d.patched.json",         (0.20, 1.20), "luna", "single"),
    "luna_multi":   ("GPT-5.6-Luna, with manager",  f"{R4}/luna_multiagent_p%d.patched.json",     (0.20, 1.20), "luna", "manager"),
    "terra_single": ("GPT-5.6-Terra, single call",  f"{R4}/terra_single_p%d.patched.json",        (2.0, 12.0),  "terra", "single"),
    "terra_multi":  ("GPT-5.6-Terra, with manager", f"{R4}/terra_multiagent_p%d.patched.json",    (2.0, 12.0),  "terra", "manager"),
    "fable_single": ("Fable 5, single call",        f"{RF}/fable5_single_p%d.patched.json",       (10.0, 50.0), "fable", "single"),
}
LABEL = {"q38": "Qwen3.8-27B", "q38fn": "Qwen3.8-Flash-Next", "luna": "GPT-5.6-Luna",
         "terra": "GPT-5.6-Terra", "dsv41": "DeepSeek-V4.1-Flash", "fable": "Claude Fable 5"}
MODEL_ORDER = ("q38", "q38fn", "luna", "terra", "dsv41", "fable")
FILLS = {k: palette.FILLS[k] for k in MODEL_ORDER}

TITLE = ("What accuracy costs — LCB-100, 5 passes, 128k max tokens, reasoning ON, "
         "all eleven pinned-backend arms")

CAPTION = ("\\textbf{Cost against accuracy.} One point per arm; an arrow runs from the "
           "single call to the manager of each model that has both.")

SCATTER_MARGINS = dict(MARGINS, right=0.985, bottom=0.30)
LEGEND_Y = 0.165
# Six models will not fit on one row: at ncol=6 the outer two are pushed off the canvas.
LEGEND_NCOL = 3

# Label placement. The default puts a manager's price above its marker and a single call's
# below, which collides once arms cluster in x -- the $2.73/$3.41/$3.46 group in particular.
# (dx, dy, ha) here overrides that for the points that need it.
LABEL_OFFSETS = {
    "fable_single": ((16, -4), "left"),
    "dsv41_single": ((0, 13), "center"),   # top of the cluster, so it goes above
    # Above (below runs into Terra's label) and right-aligned, so it extends into the gap on
    # its left rather than under DeepSeek's marker on its right.
    "q38fn_single": ((6, 13), "right"),
    "terra_multi": ((0, -20), "center"),   # below, clear of DeepSeek's marker to its right
}


def notes(pts):
    # The cap-matching, billing and price-spread notes are Table~\ref{tab:cost}'s caption
    # now; what stays here is how to read the chart.
    return [
        "x is dollars for one pass over the 100 problems (log scale), y is pass@1; the bar "
        "through each point is the 95% CI across the 5 passes (t, df = 4). Light fill = "
        "single call, dark = with manager; marker shape is the model.",
    ]


# --------------------------------------------------------------------------- data

def compute():
    tok = json.load(open(TOKENS))
    pts = []
    for key, (label, pattern, (ri, ro), mk, arm) in ARMS.items():
        a = np.array(tok[key]["tokens"], float)
        capped = key == "q38_single"
        if capped:
            a = a.copy()
            a[:, :, 1] = np.minimum(a[:, :, 1], CAP)
            a[:, :, 3] = np.minimum(a[:, :, 3], CAP)
        cost = ((a[:, :, 0] + a[:, :, 2]) * ri + (a[:, :, 1] + a[:, :, 3]) * ro) / 1e6
        per_pass = cost.sum(axis=1)
        passed = np.array([[bool(r["passed"]) for r in
                            json.load(open(pattern % p))["lcb"]["records"]] for p in PASSES])
        acc_per_pass = 100 * passed.mean(axis=1)
        pt = dict(key=key, model=mk, arm=arm, label=label, cost=per_pass.mean(),
                  cost_ci=pass_ci(per_pass), acc=acc_per_pass.mean(),
                  acc_ci=pass_ci(acc_per_pass))
        pts.append(pt)
    return pts


# --------------------------------------------------------------------------- plot

def draw(pts, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**SCATTER_MARGINS)
    ax.set_xscale("log")
    ax.set_xlabel("Cost of one pass over the 100 problems (USD, log scale)",
                  fontsize=FS_BODY, color=t["ink2"])
    ax.set_ylabel("Accuracy (pass@1, %)", fontsize=FS_BODY, color=t["ink2"])
    ax.set(xlim=(0.25, 190), ylim=(55, 100))
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100],
                  ["\\$0.50", "\\$1", "\\$2", "\\$5", "\\$10", "\\$20", "\\$50", "\\$100"])
    ax.minorticks_off()
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    by = {p["key"]: p for p in pts}
    # Every model that ran both arms gets an arrow; Fable 5 ran single-only and gets none.
    paired = [mk for mk in MODEL_ORDER
              if f"{mk}_single" in by and f"{mk}_multi" in by]
    for mk in paired:
        s, m = by[f"{mk}_single"], by[f"{mk}_multi"]
        ax.annotate("", xy=(m["cost"], m["acc"]), xytext=(s["cost"], s["acc"]),
                    arrowprops=dict(arrowstyle="-|>", color=FILLS[mk][1], lw=1.4,
                                    alpha=0.55, shrinkA=9, shrinkB=11), zorder=2)

    for p in pts:
        light, dark = FILLS[p["model"]]
        fill = light if p["arm"] == "single" else dark
        edge = ring(fill, theme)
        ax.plot([p["cost_ci"][0], p["cost_ci"][1]], [p["acc"]] * 2, color=edge, lw=1.2,
                alpha=0.8, zorder=3)
        ax.plot([p["cost"]] * 2, [p["acc_ci"][0], p["acc_ci"][1]], color=edge, lw=1.2,
                alpha=0.8, zorder=3)
        ax.scatter([p["cost"]], [p["acc"]], s=DOT, color=fill, zorder=5,
                   marker=palette.MARKER[palette.SLOT[p["model"]]],
                   edgecolor=edge, linewidth=EDGE_LW)
        if p["key"] in LABEL_OFFSETS:
            off, ha = LABEL_OFFSETS[p["key"]]
        else:
            off, ha = (0, 13 if p["arm"] == "manager" else -20), "center"
        ax.annotate(f"{p['acc']:.1f}  \\${p['cost']:.2f}", xy=(p["cost"], p["acc"]),
                    xytext=off, textcoords="offset points", ha=ha,
                    va="bottom", fontsize=FS_NOTE, color=t["ink"])

    pairs, names = [], []
    for mk in MODEL_ORDER:
        light, dark = FILLS[mk]
        cols = (light,) if f"{mk}_multi" not in by else (light, dark)
        pairs.append(tuple(
            Line2D([], [], marker=palette.MARKER[palette.SLOT[mk]], ls="", ms=12,
                   color=c, markeredgecolor=ring(c, theme),
                   markeredgewidth=EDGE_LW) for c in cols))
        names.append(LABEL[mk])

    fig.suptitle(wrap_title(TITLE), x=MARGINS["left"], ha="left", y=0.99, va="top",
                 fontsize=FS_TITLE, fontweight="bold", color=t["ink"], linespacing=1.25)
    below_panel(fig, t, pairs, names, LEGEND_Y, ncol=LEGEND_NCOL)
    if save:
        write_figure(fig, save)
    return fig


def main():
    pts = compute()
    hdr = f"{'arm':32} {'$/pass':>9} {'pass@1':>8}  {'$/point':>9}"
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(pts, key=lambda p: p["cost"]):
        print(f"{p['label']:32} {p['cost']:9.2f} {p['acc']:8.1f}  {p['cost']/p['acc']:9.3f}")
    os.makedirs(PLOTS, exist_ok=True)
    for theme in ("light",):
        draw(pts, theme, save=os.path.join(PLOTS, f"{slug(TITLE)}_{theme}.pdf"))


if __name__ == "__main__":
    main()
