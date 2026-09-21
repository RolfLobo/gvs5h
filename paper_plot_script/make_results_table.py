#!/usr/bin/env python3
"""Write paper/tab-openrouter.tex -- the LCB-100 single-to-manager table.

This table used to be typed by hand, which meant it drifted from the figures whenever the
grading changed: the captions are generated from the run data (make_figures_tex.py) but the
table was not, so a re-grade updated one and silently left the other stale.

Numbers come from runs/patched_verdicts.json, the per-problem verdicts of the patched
grader (special judges for the four problems LiveCodeBench string-compares wrongly, and a
real subprocess so sys.stdout.buffer works). That file also carries each arm's prior
.regraded.json score as `old_pass@1`, so --diff can show what the re-grade moved.

    uv run --with numpy --with scipy python paper_plot_script/make_results_table.py
    uv run ... python paper_plot_script/make_results_table.py --diff   # old vs new

Run it after the plot scripts; paper.tex \\inputs what it writes.
"""
import os
import sys
import json
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "paper")
VERDICTS = os.path.join(ROOT, "runs", "patched_verdicts.json")

# Rows in table order, with the parameter counts the paper quotes.
MODELS = [
    ("opus", "Opus-5", "n/a"),
    ("kimi", "Kimi-K3", r"${\sim}2.8$T"),
    ("mm3", "MiniMax-M3", "428B"),
    ("q35", "Qwen3.6-35B", "35B"),
    ("q9", "Qwen3.5-9B", "9B"),
]

# Columns: (run directory, header, decimals, n_passes)
COLUMNS = [
    ("128k-reasoning-on-1pass", r"\textbf{128k $\cdot$ ON (1 pass)}", 0, 1),
    ("128k-reasoning-off-1pass", r"\textbf{128k $\cdot$ OFF (1 pass)}", 0, 1),
    ("16k-reasoning-off-5pass", r"\textbf{16k $\cdot$ OFF ($\times$5)}", 1, 5),
]

# Cells the data cannot supply, and why. Opus has no 128k reasoning-off arm (the cell was
# always a {"skipped": ...} placeholder) and its 16k arm was removed as unused; Qwen3.5-9B
# with reasoning on returns reasoning-only replies.
BLANK = {
    ("opus", "128k-reasoning-off-1pass"): "--",
    ("opus", "16k-reasoning-off-5pass"): "--",
    ("q9", "128k-reasoning-on-1pass"): r"\emph{unusable}",
}

# A delta is bolded when it clears the pass-to-pass noise band the paper quotes for a
# one-pass delta (~4.5pp, Section 3.3) in the direction of the scaffold. Eyeballing which
# numbers looked notable is what the hand-typed table did; this is the rule that replaces it.
NOISE_BAND = 4.5


def arm_scores(data, run, model, engine, n_passes):
    """Mean patched pass@1 for one model/engine cell, or None if the arm is absent."""
    arms = data.get(run, {})
    if n_passes == 1:
        rec = arms.get(f"{model}_{engine}")
        return None if rec is None else rec["new_pass@1"]
    vals = [arms[f"{model}_{engine}_p{p}"]["new_pass@1"]
            for p in range(1, n_passes + 1) if f"{model}_{engine}_p{p}" in arms]
    return st.mean(vals) if vals else None


def fmt_cell(single, manager, dec):
    if single is None or manager is None:
        return None
    d = manager - single
    num = f"%.{dec}f"
    sign = f"$-${abs(d):.{dec}f}" if d < 0 else f"+{d:.{dec}f}"
    body = f"({sign})" if not (d >= NOISE_BAND) else rf"(\textbf{{{sign}}})"
    return f"{num % single} $\\to$ {num % manager} {body}"


def build(data):
    lines = []
    for key, label, params in MODELS:
        cells = []
        for run, _, dec, npass in COLUMNS:
            if (key, run) in BLANK:
                cells.append(BLANK[(key, run)])
                continue
            s = arm_scores(data, run, key, "single", npass)
            m = arm_scores(data, run, key, "multiagent", npass)
            cells.append(fmt_cell(s, m, dec) or "--")
        lines.append((label, params, cells))
    return lines


CAPTION = r"""\textbf{LCB-100 pass@1 (\%), single $\to$ manager.} For the OpenRouter-served models on the
original scaffold. ``--'' = not run; Qwen3.5-9B reasoning-on returns reasoning-only replies and is
unusable. Scores are from the corrected grader (Appendix~\ref{app:evaluator}): the four LCB-100 problems
whose reference answer is not unique are judged by the original contest rule, and submissions run
as real subprocesses so \texttt{sys.stdout.buffer} behaves as it does on the contest judge. A
bolded delta clears the ${\sim}4.5$ point pass-to-pass band. The eleven pinned-backend, five-pass
arms are reported separately in Section~\ref{sec:results} and are not pooled here, because both the
serving path and the scaffold version differ (Section~\ref{sec:setup})."""


def write_tex(rows):
    out = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + CAPTION + "}",
           r"\label{tab:openrouter}", r"\small",
           r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
           r"\textbf{Model} & \textbf{Params} & " + " &\n".join(c[1] for c in COLUMNS) + r" \\",
           r"\midrule"]
    width = max(len(r[0]) for r in rows)
    for label, params, cells in rows:
        out.append(f"{label:<{width}} & {params:<14} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(PAPER, "tab-openrouter.tex")
    with open(path, "w") as f:
        f.write("% Generated by paper_plot_script/make_results_table.py -- do not edit.\n")
        f.write("\n".join(out) + "\n")
    return path


def main():
    data = json.load(open(VERDICTS))
    rows = build(data)
    if "--diff" in sys.argv:
        print(f"{'model':12s} {'condition':26s} {'was':>16s} {'now':>16s}")
        for key, label, _ in MODELS:
            for run, _, dec, npass in COLUMNS:
                if (key, run) in BLANK:
                    continue
                for eng in ("single", "multiagent"):
                    arms = data.get(run, {})
                    names = ([f"{key}_{eng}"] if npass == 1
                             else [f"{key}_{eng}_p{p}" for p in range(1, npass + 1)])
                    have = [arms[n] for n in names if n in arms]
                    if not have:
                        continue
                    o = st.mean(x["old_pass@1"] for x in have)
                    n = st.mean(x["new_pass@1"] for x in have)
                    print(f"{label:12s} {run + '/' + eng:26s} {o:16.1f} {n:16.1f}")
        return
    path = write_tex(rows)
    for label, params, cells in rows:
        print(f"{label:12s} {params:14s} " + " | ".join(cells))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
