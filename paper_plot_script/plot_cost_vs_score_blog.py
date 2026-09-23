#!/usr/bin/env python3
"""Blog version of the cost-against-accuracy scatter: no number labels, no CI bars,
circle = single call, star = with manager, model name beside the single call, and
cost per task rather than per pass over the set.

The paper figure stays as plot_cost_vs_score.py draws it; this only writes into
blog/img/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_16k_reason_off_5_pass import (
    EDGE_LW, FIGSIZE, FS_BODY, FS_NOTE, FS_TITLE, MARGINS, THEMES,
    apply_theme, below_panel, ring,
)
from plot_cost_vs_score import FILLS, MODEL_ORDER, compute

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG = os.path.join(ROOT, "blog/img")
OUT = os.path.join(IMG, "price-vs-performance.png")

TITLE = "Price and Performance Comparison"

# DeepSeek is dropped here; the paper figure keeps all eleven arms.
DROP = ("dsv41",)
ORDER = tuple(mk for mk in MODEL_ORDER if mk not in DROP)

# compute() prices one pass over the whole set; the axis here is per task.
N_PROBLEMS = 100

# Opus-5 is not plotted -- it never ran in the pinned-backend, five-pass set -- but its
# cost is quoted in a footnote. The numbers come from the OpenRouter one-pass run, which
# logged completion tokens only, so input is estimated from the request text it stored
# (chars / 4); input is ~1% of the single call's bill and ~6% of the manager's.
OPUS_WS = f"{ROOT}/runs/128k-reasoning-on-1pass/ws"
OPUS_RATE = (15.0, 75.0)   # Anthropic list rate, $/M in, $/M out
CHARS_PER_TOKEN = 4


def opus_costs():
    """Opus-5's average USD/task, single call and with manager."""
    out = []
    for name in ("opus_single", "opus_multiagent"):
        cin = cout = 0
        dirs = sorted(os.listdir(f"{OPUS_WS}/{name}"))
        for d_ in dirs:
            with open(f"{OPUS_WS}/{name}/{d_}/transcript.jsonl") as fh:
                for line in fh:
                    d = json.loads(line)
                    if "request" not in d:
                        continue
                    cin += len(json.dumps(d["request"])) / CHARS_PER_TOKEN
                    cout += d.get("completion_tokens") or 0
        ri, ro = OPUS_RATE
        out.append((cin * ri + cout * ro) / 1e6 / len(dirs))
    return out


def usd(v):
    return f"\\${v:.3f}" if v < 0.1 else f"\\${v:.2f}"


def arm_costs(name, single, manager):
    return f"{name}: {usd(single)} single, {usd(manager)} manager."


DOT_SINGLE = 150
DOT_MANAGER = 280          # a star reads smaller than a disc of the same area
# Taller than the paper panel: the legend and the two footnote lines both live under
# the axis here.
FIGSIZE_B = (FIGSIZE[0], FIGSIZE[1] * 1.12)
MARGINS_B = dict(MARGINS, right=0.985, bottom=0.30)
LEGEND_Y = 0.20
FOOTNOTE_Y = 0.03

# Model names beside the single call point, broken at the family/variant boundary so
# each name reads as two short lines rather than one long one.
NAMES = {
    "luna":  "GPT-5.6-\nLuna",
    "q38fn": "Qwen3.8-\nFlash-Next",
    "terra": "GPT-5.6-\nTerra",
    "q38":   "Qwen3.8-\n27B",
    "fable": "Claude\nFable 5",
}

# (dx, dy) in points, then horizontal and vertical alignment of the two-line block.
NAME_OFFSETS = {
    "luna":  ((13, 0), "left", "center"),
    "q38fn": ((0, 12), "center", "bottom"),   # above, out of the $0.027/$0.034 pair
    "terra": ((0, -12), "center", "top"),
    "q38":   ((13, 0), "left", "center"),
    "fable": ((13, 0), "left", "center"),
}


def draw(pts, footnote, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    fig, ax = plt.subplots(figsize=FIGSIZE_B)
    fig.subplots_adjust(**MARGINS_B)
    ax.set_xscale("log")
    ax.set_xlabel("Cost (average USD/task)", fontsize=FS_BODY, color=t["ink2"])
    ax.set_ylabel("Accuracy (%)", fontsize=FS_BODY, color=t["ink2"])
    ax.set(xlim=(0.0025, 1.9), ylim=(55, 100))
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                  ["\\$0.005", "\\$0.01", "\\$0.02", "\\$0.05", "\\$0.10", "\\$0.20",
                   "\\$0.50", "\\$1"])
    ax.minorticks_off()
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    by = {p["key"]: p for p in pts}
    for mk in ORDER:
        if f"{mk}_single" not in by or f"{mk}_multi" not in by:
            continue
        s, m = by[f"{mk}_single"], by[f"{mk}_multi"]
        ax.annotate("", xy=(m["cost"], m["acc"]), xytext=(s["cost"], s["acc"]),
                    arrowprops=dict(arrowstyle="-|>", color=FILLS[mk][1], lw=1.4,
                                    alpha=0.55, shrinkA=9, shrinkB=13), zorder=2)

    for p in pts:
        light, dark = FILLS[p["model"]]
        single = p["arm"] == "single"
        fill = light if single else dark
        edge = ring(fill, theme)
        ax.scatter([p["cost"]], [p["acc"]], s=DOT_SINGLE if single else DOT_MANAGER,
                   color=fill, marker="o" if single else "*", edgecolor=edge,
                   linewidth=EDGE_LW, zorder=5)
        if not single:
            continue
        off, ha, va = NAME_OFFSETS[p["model"]]
        ax.annotate(NAMES[p["model"]], xy=(p["cost"], p["acc"]), xytext=off,
                    textcoords="offset points", ha=ha, va=va, linespacing=1.2,
                    fontsize=FS_NOTE, color=t["ink"])

    # Colour is the model here, so the legend carries shape only: neutral fills.
    handles = [
        Line2D([], [], marker="o", ls="", ms=11, color=t["grid"],
               markeredgecolor=t["muted"], markeredgewidth=EDGE_LW),
        Line2D([], [], marker="*", ls="", ms=18, color=t["ink2"],
               markeredgecolor=t["ink2"], markeredgewidth=EDGE_LW),
    ]
    fig.suptitle(TITLE, x=MARGINS["left"], ha="left", y=0.99, va="top",
                 fontsize=FS_TITLE, fontweight="bold", color=t["ink"], linespacing=1.25)
    below_panel(fig, t, handles, ["single call", "with manager"], LEGEND_Y, ncol=2)

    fig.text(0.5, FOOTNOTE_Y, footnote, ha="center", va="bottom", linespacing=1.4,
             fontsize=FS_NOTE * 0.85, color=t["muted_text"])
    if save:
        fig.savefig(save, dpi=200)
        print("wrote", save)
    return fig


def main():
    allpts = [dict(p, cost=p["cost"] / N_PROBLEMS) for p in compute()]
    for p in sorted(allpts, key=lambda p: p["cost"]):
        print(f"{p['label']:34} {p['cost']:9.4f} {p['acc']:8.1f}")

    # The two arms the chart leaves out, priced in a footnote instead.
    by = {p["key"]: p["cost"] for p in allpts}
    footnote = "\n".join([
        "Not plotted — " + arm_costs("DeepSeek-V4.1-Flash",
                                     by["dsv41_single"], by["dsv41_multi"]),
        arm_costs("Opus 5", *opus_costs()),
    ])
    print("\n" + footnote.replace("\\$", "$"))

    os.makedirs(IMG, exist_ok=True)
    draw([p for p in allpts if p["model"] not in DROP], footnote, "light", save=OUT)


if __name__ == "__main__":
    main()
