#!/usr/bin/env python3
"""Blog version of the manager-vs-single-call bars: no value labels, no CI bars, plus
Opus-5 from the earlier one-pass run.

The paper figure stays as plot_4new_5pass_reason_on.py draws it; this only writes into
blog/img/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_16k_reason_off_5_pass import (
    EDGE_LW, FS_BODY, FS_SUB, FS_TITLE, THEMES, apply_theme, ring,
)
from plot_4new_5pass_reason_on import (
    BAR_W, FIGSIZE_V, FILLS, LEG_Y, M4, PITCH, XLABELS, compute,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG = os.path.join(ROOT, "blog/img")
OUT = os.path.join(IMG, "benchmark-livecodebench.png")

TITLE = "Benchmark (LiveCodeBench)"

# Opus-5 never ran in the pinned-backend, five-pass set. These are its scores from the
# OpenRouter one-pass run on the earlier scaffold, which the paper reports separately.
OPUS = f"{ROOT}/runs/128k-reasoning-on-1pass/results"
# Its palette teal is DeepSeek's teal at a glance, so it takes the magenta that
# plot_128k_reason_on_1_pass.py already gives it.
OPUS_FILL = ("#e8a8c8", "#a8306a")

FILLS_B = dict(FILLS, opus=OPUS_FILL)
XLABELS_B = dict(XLABELS, opus="Claude\nOpus 5")

# The axis runs 50-100 so the bars are easy to tell apart, and reads as a broken one:
# the baseline is labelled 0, the break mark sits between it and 60, and the ticks
# above are the real values.
YLIM = (50, 100)
SINGLE_GREY = {"light": "#d5d4cd", "dark": "#b8b6ae"}
YTICKS = (50, 60, 70, 80, 90, 100)
YTICKLABELS = ("0", "60", "70", "80", "90", "100")
# Break mark position, axes fraction: midway between the baseline and the 60 tick.
BREAK_Y = 0.10


def opus_stat():
    def acc(name):
        recs = json.load(open(f"{OPUS}/{name}.patched.json"))["lcb"]["records"]
        return 100 * sum(bool(r["passed"]) for r in recs) / len(recs)

    return dict(key="opus", label="Claude Opus 5", has_mgr=True,
                single=acc("opus_single"), multi=acc("opus_multiagent"))


def axis_break(ax, t, y=BREAK_Y, size=0.035, gap=0.028):
    """The two slashes low on the y axis that mark it as starting above zero."""
    bb = ax.get_position()
    # Equal lengths on the page, not in axes fraction: the panel is far wider than tall.
    dx = size * (bb.height * FIGSIZE_V[1]) / (bb.width * FIGSIZE_V[0])
    kw = dict(transform=ax.transAxes, clip_on=False, zorder=6, solid_capstyle="butt")
    ax.plot([0, 0], [y - gap, y + gap], color=t["surface"], lw=3.0, **kw)
    for y0 in (y - gap / 2, y + gap / 2):
        ax.plot([-dx / 2, dx / 2], [y0 - size / 2, y0 + size / 2],
                color=t["axis"], lw=1.6, **kw)


def draw(stats, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    stats = sorted(stats, key=lambda s: (s["key"] != "fable", s["single"]))
    fig, ax = plt.subplots(figsize=FIGSIZE_V)
    fig.subplots_adjust(**M4)
    xs = [i * PITCH for i in range(len(stats))]
    ax.set(xlim=(-0.62, xs[-1] + 0.62), ylim=YLIM)
    ax.set_ylabel("Accuracy (%)", fontsize=FS_BODY, color=t["ink2"], loc="top")
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_yticks(YTICKS, YTICKLABELS)
    ax.set_xticks(xs)
    ax.set_xticklabels([XLABELS_B[s["key"]] for s in stats], fontsize=FS_BODY,
                       fontweight="bold", color=t["ink"], linespacing=1.15)
    ax.tick_params(length=0, labelsize=FS_BODY)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])
    axis_break(ax, t)

    fable = next(s for s in stats if s["key"] == "fable")["single"]
    ax.axhline(fable, ls=(0, (6, 4)), lw=1.6, color=FILLS_B["fable"][1], alpha=0.85,
               zorder=2)

    # Every single call is the same grey, so colour marks the manager arm alone.
    grey = SINGLE_GREY[theme]
    for x, s in zip(xs, stats):
        arms = [("single", grey, -BAR_W / 2 if s["has_mgr"] else 0.0)]
        if s["has_mgr"]:
            arms.append(("multi", FILLS_B[s["key"]][1], BAR_W / 2))
        for arm, fill, dx in arms:
            ax.bar(x + dx, s[arm], BAR_W, color=fill, edgecolor=ring(fill, theme),
                   linewidth=EDGE_LW, zorder=3)

    def swatch(colour):
        return Line2D([], [], marker="o", ls="", ms=12, color=colour,
                      markeredgecolor=ring(colour, theme), markeredgewidth=EDGE_LW)

    deep = "#6f6d67" if theme == "light" else "#5e5c57"
    key = [(swatch(grey), "single call"),
           (swatch(deep), "with manager"),
           (Line2D([], [], ls=(0, (6, 4)), lw=1.6, color=FILLS_B["fable"][1]),
            f"Fable 5 single call, {fable:.1f}%")]

    fig.suptitle(TITLE, x=M4["left"], ha="left", y=0.99, va="top",
                 fontsize=FS_TITLE, fontweight="bold", color=t["ink"], linespacing=1.25)
    leg = fig.legend([h for h, _ in key], [n for _, n in key], loc="lower center",
                     bbox_to_anchor=(0.5, LEG_Y), ncol=len(key), frameon=False,
                     fontsize=FS_SUB, labelcolor=t["ink2"], handletextpad=0.7,
                     handlelength=2.2, columnspacing=2.8)
    fig.add_artist(leg)
    if save:
        fig.savefig(save, dpi=200)
        print("wrote", save)
    return fig


def main():
    stats = compute() + [opus_stat()]
    for s in sorted(stats, key=lambda s: (s["key"] != "fable", s["single"])):
        mgr = f"{s['multi']:6.1f}" if s["has_mgr"] else "     -"
        print(f"{s['label']:22} {s['single']:6.1f} {mgr}")
    os.makedirs(IMG, exist_ok=True)
    draw(stats, "light", save=OUT)


if __name__ == "__main__":
    main()
