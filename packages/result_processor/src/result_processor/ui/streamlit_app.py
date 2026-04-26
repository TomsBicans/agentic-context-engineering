"""Streamlit dashboard for inspecting experiment + analysis results.

Long-running operations (analyze, visualize) are dispatched as `result-processor`
subprocesses so the UI is a thin wrapper that always uses the same code path
as the terminal / Makefile entry points.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from result_processor.visualization.loader import (
    build_dataframe,
    load_analyses,
    load_runs,
)
from result_processor.visualization.plots import ALL_PLOTS


def _default_dir(env_key: str, fallback: str) -> str:
    return os.environ.get(env_key) or fallback


def _refresh_data(experiment_dir: Path, analysis_dir: Path) -> dict:
    """Read the JSONL files from disk. Cached via st.cache_data."""
    runs = load_runs(experiment_dir)
    analyses = load_analyses(analysis_dir)
    df = build_dataframe(experiment_dir, analysis_dir)
    return {"runs": runs, "analyses": analyses, "df": df}


# Wrap at module level so st.cache_data can hash by argument values.
_load_data = st.cache_data(show_spinner=False)(_refresh_data)


def _run_subprocess(args: list[str], status_label: str) -> int:
    """Stream a subprocess to a code block and return its exit code."""
    with st.status(status_label, expanded=True) as status:
        st.code(" ".join(args), language="bash")
        log = st.empty()
        lines: list[str] = []
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip("\n"))
            log.code("\n".join(lines[-200:]))
        rc = process.wait()
        if rc == 0:
            status.update(label=f"{status_label} ✓", state="complete")
        else:
            status.update(label=f"{status_label} (exit {rc})", state="error")
        return rc


def _cli_path() -> list[str]:
    """Invoke the CLI via the current Python interpreter to avoid PATH issues."""
    return [sys.executable, "-m", "result_processor.main"]


def _sidebar() -> dict:
    st.sidebar.header("Configuration")
    experiment_dir = st.sidebar.text_input(
        "experiment_results_dir",
        value=_default_dir("RP_EXPERIMENT_RESULTS_DIR", "./data/experiment_results"),
    )
    analysis_dir = st.sidebar.text_input(
        "analysis_results_dir",
        value=_default_dir("RP_ANALYSIS_RESULTS_DIR", "./data/analysis_results"),
    )
    corpora_root = st.sidebar.text_input(
        "path_to_corpora",
        value="./corpora/scraped_data",
    )
    examiner_model = st.sidebar.text_input("examiner_model", value="qwen3:4b")
    num_ctx = st.sidebar.number_input("num_ctx", min_value=1024, max_value=131072, value=8192, step=1024)

    if st.sidebar.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return {
        "experiment_dir": Path(experiment_dir),
        "analysis_dir": Path(analysis_dir),
        "corpora_root": corpora_root,
        "examiner_model": examiner_model,
        "num_ctx": int(num_ctx),
    }


def _tab_overview(df: pd.DataFrame, runs, analyses) -> None:
    st.subheader("Overview")
    if df.empty:
        st.info("No runs found yet. Generate some with `experiment-runner run`.")
        return

    n_runs = len(runs)
    n_analyzed = sum(1 for r in runs if r.run_id in analyses)
    coverage = n_analyzed / n_runs if n_runs else 0

    cols = st.columns(4)
    cols[0].metric("Total runs", n_runs)
    cols[1].metric("Analyzed", n_analyzed)
    cols[2].metric("Coverage", f"{coverage * 100:.0f}%")
    cols[3].metric("Systems", df["system_name"].nunique())

    st.markdown("**Per-system support rate (mean):**")
    grouped = (
        df.dropna(subset=["support_rate"])
        .groupby("system_name", as_index=False)["support_rate"]
        .mean()
        .round(3)
    )
    st.dataframe(grouped, use_container_width=True, hide_index=True)


def _tab_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    st.subheader("Runs")
    if df.empty:
        st.info("Empty dataset.")
        return df, []

    cols = st.columns(4)
    systems = cols[0].multiselect("System", sorted(df["system_name"].unique()))
    corpora = cols[1].multiselect("Corpus", sorted(df["corpus"].unique()))
    levels_raw = sorted(df["level"].dropna().unique().tolist())
    levels = cols[2].multiselect("Level", [int(x) for x in levels_raw])
    only_unanalyzed = cols[3].checkbox("Unanalyzed only", value=False)

    filtered = df.copy()
    if systems:
        filtered = filtered[filtered["system_name"].isin(systems)]
    if corpora:
        filtered = filtered[filtered["corpus"].isin(corpora)]
    if levels:
        filtered = filtered[filtered["level"].isin(levels)]
    if only_unanalyzed:
        filtered = filtered[filtered["support_rate"].isna()]

    display_cols = [
        "run_id",
        "system_name",
        "corpus",
        "level",
        "question_id",
        "model",
        "execution_time_s",
        "tool_call_count",
        "support_rate",
        "verdict",
        "helpfulness_rating",
    ]
    available = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[available], use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered)} run(s) match filters")

    return filtered, list(filtered["run_id"])


def _tab_charts(df: pd.DataFrame) -> None:
    st.subheader("Charts")
    if df.empty or df["support_rate"].dropna().empty:
        st.info("Run analyze first to populate charts.")
        return

    plot_names = list(ALL_PLOTS.keys())
    selected = st.multiselect("Charts to show", plot_names, default=plot_names[:4])
    for name in selected:
        st.plotly_chart(ALL_PLOTS[name](df), use_container_width=True)


def _tab_run_details(df: pd.DataFrame, analyses) -> None:
    st.subheader("Run details")
    if df.empty:
        st.info("No runs.")
        return

    run_id = st.selectbox("Run ID", df["run_id"].tolist())
    row = df[df["run_id"] == run_id].iloc[0]

    cols = st.columns(3)
    cols[0].metric("System", row["system_name"])
    cols[1].metric("Corpus", row["corpus"])
    cols[2].metric("Level", row["level"] if pd.notna(row["level"]) else "—")

    st.markdown(f"**Question:** {row['question_text']}")
    with st.expander("Answer", expanded=True):
        st.write(row["answer_text"] or "(empty)")

    analysis = analyses.get(run_id)
    if analysis is None:
        st.warning("Not yet analyzed.")
        return

    cols = st.columns(4)
    cols[0].metric("Support", f"{analysis.support_rate:.2f}")
    cols[1].metric("Error", f"{analysis.error_rate:.2f}")
    cols[2].metric("Verdict", analysis.verdict.value)
    cols[3].metric("Helpfulness", analysis.helpfulness_rating or "—")

    if analysis.examiner_notes:
        st.markdown(f"**Examiner notes:** {analysis.examiner_notes}")

    st.markdown(f"**Per-claim verdicts ({len(analysis.claims)})**")
    claim_rows = [
        {
            "status": c.status.value,
            "statement": c.statement,
            "file": c.cited_file or "—",
            "lines": (
                f"{c.cited_line_start}-{c.cited_line_end}"
                if c.cited_line_start is not None
                else "—"
            ),
            "justification": c.justification,
        }
        for c in analysis.claims
    ]
    if claim_rows:
        st.dataframe(pd.DataFrame(claim_rows), use_container_width=True, hide_index=True)


def _tab_actions(cfg: dict, filtered: pd.DataFrame) -> None:
    st.subheader("Actions")
    st.caption("Both actions shell out to the `result-processor` CLI so behaviour matches Makefile/terminal usage.")

    st.markdown("### Analyze")
    cols = st.columns([1, 1, 2])
    resume = cols[0].checkbox("Resume (skip cached)", value=True)
    only_filtered = cols[1].checkbox("Only filtered runs", value=False)

    if cols[2].button("▶ Run analyze", type="primary", use_container_width=True):
        args = _cli_path() + [
            "analyze",
            "--experiment-results-dir", str(cfg["experiment_dir"]),
            "--output-dir", str(cfg["analysis_dir"]),
            "--path-to-corpora", cfg["corpora_root"],
            "--examiner-model", cfg["examiner_model"],
            "--num-ctx", str(cfg["num_ctx"]),
        ]
        if not resume:
            args.append("--no-resume")
        if only_filtered and not filtered.empty:
            input_files = sorted({_locate_source_file(cfg["experiment_dir"], rid) for rid in filtered["run_id"]})
            input_files = [f for f in input_files if f]
            if input_files:
                args.append("--input-files")
                args.extend(input_files)
        rc = _run_subprocess(args, "Running analyze")
        if rc == 0:
            st.cache_data.clear()

    st.markdown("### Visualize")
    cols = st.columns([3, 1])
    output_dir = cols[0].text_input("Output directory", value="./data/figures")
    formats = cols[1].multiselect("Formats", ["html", "png", "pdf", "svg"], default=["html"])
    if st.button("▶ Run visualize", type="primary", use_container_width=True):
        args = _cli_path() + [
            "visualize",
            "--experiment-results-dir", str(cfg["experiment_dir"]),
            "--analysis-results-dir", str(cfg["analysis_dir"]),
            "--output-dir", output_dir,
            "--formats", *formats,
        ]
        rc = _run_subprocess(args, "Running visualize")
        if rc == 0:
            st.success(f"Figures written to {output_dir}/")


def _locate_source_file(experiment_dir: Path, run_id: str) -> str | None:
    """Find which JSONL file contains a given run_id (for --input-files)."""
    for jsonl in experiment_dir.glob("*.jsonl"):
        try:
            with jsonl.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        if json.loads(line).get("run_id") == run_id:
                            return str(jsonl)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return None


def main() -> None:
    st.set_page_config(page_title="Result processor", layout="wide")
    st.title("Experiment result processor")

    cfg = _sidebar()

    try:
        data = _load_data(cfg["experiment_dir"], cfg["analysis_dir"])
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    df: pd.DataFrame = data["df"]
    runs = data["runs"]
    analyses = data["analyses"]

    overview_tab, runs_tab, charts_tab, details_tab, actions_tab = st.tabs(
        ["Overview", "Runs", "Charts", "Run details", "Actions"]
    )

    with overview_tab:
        _tab_overview(df, runs, analyses)

    with runs_tab:
        filtered, _ = _tab_runs(df)

    with charts_tab:
        _tab_charts(df)

    with details_tab:
        _tab_run_details(df, analyses)

    with actions_tab:
        _tab_actions(cfg, filtered)


if __name__ == "__main__" or __name__ == "__page__":
    main()
