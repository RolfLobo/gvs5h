#!/usr/bin/env python3
"""Write paper/tab-qwen-caps.tex and paper/tab-emitted.tex.

Both were typed by hand and both quote pass@1, so both went stale the moment the grading
changed. They are computed here from the same *.patched.json files the charts read, so all
three tables and every figure now come off one grading.

  tab:qwen-caps  Qwen3.8-27B's single arm at each cap: score, how often the generation hit
                 the cap (finish_reason=length), how often nothing gradable came back.
  tab:emitted    Each pinned-backend single arm re-scored over only the problem-passes
                 where it emitted code -- the ceiling an emit-rate fix could reach.

    uv run --with numpy --with scipy python paper_plot_script/make_arm_tables.py
"""
import os
import json
import statistics as st

from plot_16k_reason_off_5_pass import pass_ci
from plot_4new_5pass_reason_on import compute as pinned_stats, p_tex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "paper")
R4 = os.path.join(ROOT, "runs/firstparty-128k-reasoning-on-5pass/results")
RF = os.path.join(ROOT, "runs/fable5-128k-reasoning-on-5pass/results")
FN = os.path.join(ROOT, "runs/q38-fn-5pass/results")
DS = os.path.join(ROOT, "runs/ds-v41-f-5pass/results")
PASSES = 5


def load(pattern):
    """-> list of per-pass record lists."""
    return [json.load(open(pattern % p))["lcb"]["records"] for p in range(1, PASSES + 1)]


def score(passes):
    return [100.0 * sum(bool(r["passed"]) for r in recs) / len(recs) for recs in passes]


def emitted(recs):
    return [r for r in recs if (r.get("code") or "").strip()]


def qwen_caps():
    # "Cut off by the cap" means a different test per row. The 250k row is the generation as
    # it ran, so its cap hits are finish_reason=length against the 250k ceiling. The 128k row
    # is a replay of those same generations truncated at 128k, and capmatch_q38.py does not
    # rewrite finish_reason -- what was cut there is whatever ran past 128,000 output tokens.
    rows = []
    for label, pat, cut_hit in (
            (r"128k (cap-matched, App.~\ref{app:protocol})",
             f"{R4}/q38_single_p%d.cap128k.patched.json",
             lambda r: (r.get("completion_tokens") or 0) >= 128_000),
            ("250k (as generated)", f"{R4}/q38_single_p%d.patched.json",
             lambda r: r.get("finish_reason") == "length")):
        passes = load(pat)
        flat = [r for recs in passes for r in recs]
        n = len(flat)
        lo, hi = pass_ci(score(passes))
        mean, half = (lo + hi) / 2, (hi - lo) / 2
        cut = sum(1 for r in flat if cut_hit(r))
        nocode = n - len(emitted(flat))
        rows.append((label, rf"${mean:.1f} \pm {half:.1f}$",
                     rf"{cut} / {n} \ ({100.0 * cut / n:.1f}\,\%)",
                     rf"{nocode} / {n} \ ({100.0 * nocode / n:.1f}\,\%)"))
    return rows


def emitted_table():
    arms = [("GPT-5.6-Terra", f"{R4}/terra_single_p%d.patched.json"),
            ("GPT-5.6-Luna", f"{R4}/luna_single_p%d.patched.json"),
            ("Qwen3.8-27B", f"{R4}/q38_single_p%d.cap128k.patched.json"),
            ("Claude Fable 5", f"{RF}/fable5_single_p%d.patched.json")]
    rows = []
    for label, pat in arms:
        passes = load(pat)
        flat = [r for recs in passes for r in recs]
        n = len(flat)
        as_scored = st.mean(score(passes))
        emit = emitted(flat)
        restricted = 100.0 * sum(bool(r["passed"]) for r in emit) / len(emit)
        d = restricted - as_scored
        tail = "" if len(emit) == n else rf" (${'+' if d >= 0 else '-'}{abs(d):.1f}$)"
        rows.append((label, f"{as_scored:.1f}", f"{len(emit)}/{n}", f"{restricted:.1f}{tail}"))
    return rows


def agreement_table():
    """Problem-passes won by exactly one arm. Read off the same compute() that makes
    Table~\\ref{tab:main} and the bars, so the two cannot disagree about a run."""
    return [(s["label"], str(s["mgr_only"]), str(s["sgl_only"]), str(s["n_pp"]),
             p_tex(s["mcnemar_p"]))
            for s in pinned_stats() if s["has_mgr"]]


CAP_ARMS = [
    ("GPT-5.6-Terra", f"{R4}/terra_single_p%d.patched.json",
     f"{R4}/terra_multiagent_p%d.patched.json", "128k"),
    ("GPT-5.6-Luna", f"{R4}/luna_single_p%d.patched.json",
     f"{R4}/luna_multiagent_p%d.patched.json", "128k"),
    ("Qwen3.8-27B", f"{R4}/q38_single_p%d.cap128k.patched.json",
     f"{R4}/q38_multiagent_p%d.patched.json", "128k\\textsuperscript{a}"),
    # The same generations read at the cap they were made under. There is no 250k manager
    # arm to pair it with -- that arm ran natively at 128k -- so its manager cells are empty.
    ("Qwen3.8-27B (250k)", f"{R4}/q38_single_p%d.patched.json", None, "250k\\textsuperscript{a}"),
    ("Qwen3.8-Flash-Next", f"{FN}/q38_fn_single_p%d.patched.json",
     f"{FN}/q38_fn_multiagent_p%d.patched.json", "128k"),
    ("DeepSeek-V4.1-Flash", f"{DS}/ds_v41_f_single_p%d.patched.json",
     f"{DS}/ds_v41_f_multiagent_p%d.patched.json", "128k"),
    ("Claude Fable 5", f"{RF}/fable5_single_p%d.patched.json", None,
     "128k"),
]


def caps_table():
    """Cap hits and empty solutions per arm.

    The single arm makes one call per problem-pass, so its cap hits are out of 500. The
    manager makes several, and truncated_calls counts every one of them that stopped at the
    limit (multiagent.py sums finish_reason == length over the whole transcript), so its
    denominator is the arm's total call count, not 500.

    Qwen3.8-27B's single arm is the 128k replay of a 250k generation and capmatch_q38.py does
    not rewrite finish_reason, so a cap hit there is a completion at or past 128,000 tokens.
    """
    def scores(pat):
        """pass@1 as scored, and re-scored over only the problem-passes that emitted code."""
        passes = load(pat)
        flat = [r for recs in passes for r in recs]
        emit = emitted(flat)
        return (st.mean(score(passes)),
                100.0 * sum(bool(r["passed"]) for r in emit) / len(emit))

    rows = []
    for label, s_pat, m_pat, cap in CAP_ARMS:
        s = [r for recs in load(s_pat) for r in recs]
        if "cap128k" in s_pat:
            s_hits = sum(1 for r in s if (r.get("completion_tokens") or 0) >= 128_000)
        else:
            s_hits = sum(1 for r in s if r.get("finish_reason") == "length")
        s_nocode = len(s) - len(emitted(s))
        refusals = sum(1 for r in s if r.get("finish_reason") == "refusal")
        s_score, s_emit_score = scores(s_pat)
        if m_pat is None:
            m_hits, m_nocode = "---", "---"
            score_cell = f"{s_score:.1f} / ---"
            emit_cell = f"{s_emit_score:.1f} / ---"
        else:
            m = [r for recs in load(m_pat) for r in recs]
            m_calls = sum(r.get("n_calls") or 0 for r in m)
            m_hits = f"{sum(r.get('truncated_calls') or 0 for r in m)} / {m_calls:,}"
            m_nocode = str(len(m) - len(emitted(m)))
            m_score, m_emit_score = scores(m_pat)
            score_cell = f"{s_score:.1f} / {m_score:.1f}"
            emit_cell = f"{s_emit_score:.1f} / {m_emit_score:.1f}"
        rows.append((label, f"{s_hits} / {len(s)}", m_hits,
                     f"{s_nocode} / {m_nocode}", str(refusals),
                     score_cell, emit_cell, cap))
    return rows


def write(path, caption, label, header, rows, colspec="@{}lccc@{}", footer=None):
    out = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}",
           rf"\label{{{label}}}", r"\small", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
           header, r"\midrule"]
    width = max(len(r[0]) for r in rows)
    for r in rows:
        out.append(f"{r[0]:<{width}} & " + " & ".join(r[1:]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if footer:
        out += [r"\vspace{0.6ex}", r"\begin{minipage}{0.92\linewidth}", r"\footnotesize",
                footer, r"\end{minipage}"]
    out += [r"\end{table}"]
    full = os.path.join(PAPER, path)
    with open(full, "w") as f:
        f.write("% Generated by paper_plot_script/make_arm_tables.py -- do not edit.\n")
        f.write("\n".join(out) + "\n")
    return full


def main():
    caps = qwen_caps()
    write("tab-qwen-caps.tex",
          r"\textbf{Qwen3.8-27B's single arm read at both caps.} Cap-matched to 128k, and as "
          r"generated at 250k. Scores are from the corrected grader (Appendix~\ref{app:evaluator}); "
          r"$\pm$ is a 95\% $t$ interval across the five passes.",
          "tab:qwen-caps",
          r"\textbf{Qwen3.8-27B single} & \textbf{pass@1} & \textbf{Cut off by the cap} &"
          "\n" r"\textbf{No code at all} \\", caps)
    emit = emitted_table()
    write("tab-emitted.tex",
          r"\textbf{Each single arm re-scored} over only the problem-passes where it emitted "
          r"code. Scores are from the corrected grader (Appendix~\ref{app:evaluator}).",
          "tab:emitted",
          r"\textbf{Model (single arm)} & \textbf{As scored} & \textbf{Emitted code} &"
          "\n" r"\textbf{Restricted to those} \\", emit)
    caps_rows = caps_table()
    write("tab-caps.tex",
          r"\textbf{Cap hits and empty solutions per arm.} A \emph{cap hit} is a generation "
          r"that stopped at the token limit (\texttt{finish\_reason=length}).  \emph{No code} counts  "
          r"any run containing no code, due to a cap hit or other reason.",
          "tab:caps",
          # Each of these headers is wider than the column under it; stacked they fit the line.
          r"\textbf{Model} & \textbf{\shortstack{Cap hits\\single}} &"
          "\n" r"\textbf{\shortstack{Cap hits\\manager}} & \textbf{\shortstack{No code\\s / m}} &"
          "\n" r"\textbf{\shortstack{of which\\refusals}} & \textbf{\shortstack{pass@1\\s / m}} &"
          "\n" r"\textbf{\shortstack{Emitted only\\s / m}} & \textbf{Cap} \\",
          caps_rows,
          # Eight columns overrun the line at the default \tabcolsep; 0.45em gaps fit.
          colspec="@{}l" + "@{\\hspace{0.45em}}c" * 7 + "@{}",
          footer=(r"\textsuperscript{a}~ Qwen3.8-27B's single "
                  r"arm was run at 250k and cap-matched back to 128k, where a "
                  r"cap hit is a completion reaching 128,000 output tokens."
          )
        )
    agree = agreement_table()
    write("tab-agreement.tex",
          r"\textbf{Problem-passes won by one arm alone.} Of the 500 problem-passes per "
          r"model (100 problems $\times$ 5 passes), the number the manager arm solved and "
          r"the single call did not, against the number the single call solved and the "
          r"manager did not. The gains are not compensating wins and losses. The test is "
          r"exact McNemar on those discordant pairs (Section~\ref{sec:setup}).",
          "tab:agreement",
          r"\textbf{Model} & \textbf{Manager only} & \textbf{Single call only} &"
          "\n" r"\textbf{Problem-passes} & \textbf{Exact McNemar $p$} \\", agree,
          colspec="@{}lcccc@{}")
    for r in caps + emit + agree:
        print(" | ".join(r))
    print(f"\nWrote {PAPER}/tab-qwen-caps.tex, tab-emitted.tex and tab-agreement.tex")


if __name__ == "__main__":
    main()
