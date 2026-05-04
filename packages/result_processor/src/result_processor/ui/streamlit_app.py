"""Streamlit dashboard for inspecting experiment + analysis results.

Long-running operations (analyze, visualize) are dispatched as `result-processor`
subprocesses so the UI is a thin wrapper that always uses the same code path
as the terminal / Makefile entry points.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from experiment_runner.commands.suite import (
    build_suite_tasks,
    load_suite_config,
    load_suite_state,
    save_suite_config,
    summarize_suite_state,
)
from experiment_runner.models.enums import AutomationLevel, Corpus, SystemName
from experiment_runner.models.result import RunResult
from experiment_runner.models.suite import (
    ExperimentSuiteConfig,
    SuiteCorpusSelection,
    SuiteTaskStatus,
    default_state_path,
    suite_slug,
)
from experiment_runner.runners.registry import DISABLED_SYSTEMS, SYSTEM_AUTOMATION_LEVELS
from result_processor.visualization.loader import (
    build_dataframe,
    load_analyses,
    load_runs,
)
from result_processor.visualization.plots import ALL_PLOTS


_AUTOMATION_BADGE: dict[AutomationLevel, str] = {
    AutomationLevel.FULL: "⚙ automated",
    AutomationLevel.PARTIAL: "◑ partial",
    AutomationLevel.MANUAL: "✋ manual",
}

_ALL_SYSTEMS: list[SystemName] = [s for s in SYSTEM_AUTOMATION_LEVELS if s not in DISABLED_SYSTEMS]
_AUTOMATED_SYSTEMS: list[SystemName] = [
    s for s in _ALL_SYSTEMS if SYSTEM_AUTOMATION_LEVELS[s] == AutomationLevel.FULL
]


def _system_label(system: SystemName) -> str:
    badge = _AUTOMATION_BADGE.get(SYSTEM_AUTOMATION_LEVELS[system], "?")
    return f"{system.value}  [{badge}]"


CORPUS_DEFAULTS: dict[str, tuple[str, str]] = {
    "solar_system_wiki": (
        "./corpora/questions/solar_system.json",
        "./corpora/scraped_data/solar_system_wiki",
    ),
    "oblivion_wiki": (
        "./corpora/questions/oblivion.json",
        "./corpora/scraped_data/oblivion_wiki",
    ),
    "scipy": (
        "./corpora/questions/scipy.json",
        "./corpora/scraped_data/scipy_repo",
    ),
}

_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+)$")
_RESULT_RE = re.compile(r"^results\s*(?:→|->)\s*(.+)$")
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_SUITE_DIR = Path("./data/experiment_suites")


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


def _experiment_runner_cli() -> list[str]:
    return [sys.executable, "-m", "experiment_runner.main"]


def _build_experiment_run_args(
    *,
    system: str,
    corpus: str,
    questions_file: str,
    output_dir: str,
    model: str,
    num_ctx: int,
    path_to_corpora: str,
    automation_level: str,
    selected_ids: list[str],
    reasoning_enabled: bool,
    no_trace: bool,
    dry_run: bool,
) -> list[str]:
    args = _experiment_runner_cli() + [
        "run",
        "--system", system,
        "--corpus", corpus,
        "--questions-file", questions_file,
        "--output-dir", output_dir,
        "--model", model,
        "--num-ctx", str(num_ctx),
        "--path-to-corpora", path_to_corpora,
        "--automation-level", automation_level,
    ]
    if selected_ids:
        args.append("--question-ids")
        args.extend(selected_ids)
    if reasoning_enabled:
        args.append("--reasoning-enabled")
    if no_trace:
        args.append("--no-trace")
    if dry_run:
        args.append("--dry-run")
    return args


def _build_suite_run_args(config_path: str, state_path: str) -> list[str]:
    return _experiment_runner_cli() + [
        "suite",
        "run",
        "--config",
        config_path,
        "--state",
        state_path,
    ]


def _build_suite_cancel_args(state_path: str) -> list[str]:
    return _experiment_runner_cli() + [
        "suite",
        "cancel",
        "--state",
        state_path,
    ]


def _shell_command(args: list[str]) -> str:
    return shlex.join(args)


def _render_shell_command_preview(command: str) -> None:
    st.markdown("Shell command")
    st.code(command, language="bash")


def _parse_ollama_list(output: str) -> list[str]:
    models: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("NAME "):
            continue
        models.append(stripped.split()[0])
    return models


def _query_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return _parse_ollama_list(result.stdout)


_load_ollama_models = st.cache_data(show_spinner=False, ttl=60)(_query_ollama_models)


def _model_options(default: str = DEFAULT_MODEL) -> list[str]:
    models = _load_ollama_models()
    if default not in models:
        return [default, *models]
    return models


def _select_model(label: str, default: str = DEFAULT_MODEL, *, key: str) -> str:
    options = _model_options(default)
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, key=key)


def _render_run_errors(result_path: str) -> None:
    """Parse a result JSONL and show any answer_error entries immediately."""
    import json as _json

    path = Path(result_path)
    if not path.exists():
        return

    errors: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = _json.loads(line)
                if record.get("answer_error"):
                    errors.append({
                        "question_id": record.get("question_id", "?"),
                        "question": record.get("question_text", "")[:80],
                        "error": record["answer_error"],
                    })
    except Exception:
        return

    if not errors:
        return

    with st.expander(f"⚠ {len(errors)} error(s) in last run", expanded=True):
        for e in errors:
            st.error(f"**{e['question_id']}** — {e['question']}\n\n`{e['error']}`")


def _run_experiment_subprocess(
    args: list[str],
    expected_total: int,
    status_label: str = "Running experiment",
) -> tuple[int, str | None]:
    """Run experiment-runner with a live progress bar driven by `[i/total]` lines.

    Returns (exit_code, result_path). `result_path` is the JSONL file the runner
    wrote (parsed from the trailing `results → <path>` line), or None on failure
    or dry-run.
    """
    with st.status(status_label, expanded=True) as status:
        st.code(" ".join(args), language="bash")
        progress = st.progress(0.0, text="Starting…")
        log_box = st.empty()

        # Force unbuffered output from the child so `[i/total]` lines arrive
        # immediately rather than being held until the buffer fills.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None

        lines: list[str] = []
        current = 0
        total = max(expected_total, 1)
        result_path: str | None = None

        for line in process.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            log_box.code("\n".join(lines[-200:]))

            m_prog = _PROGRESS_RE.match(line)
            if m_prog:
                current = int(m_prog.group(1))
                total = int(m_prog.group(2))
                progress.progress(
                    min(current / max(total, 1), 1.0),
                    text=f"[{current}/{total}] {m_prog.group(3)[:100]}",
                )
                continue

            m_res = _RESULT_RE.match(line)
            if m_res:
                result_path = m_res.group(1).strip()

        rc = process.wait()
        if rc == 0:
            progress.progress(1.0, text=f"Done — {current}/{total}")
            label = f"{status_label} ✓"
            if result_path:
                label += f" — {result_path}"
            status.update(label=label, state="complete")
        else:
            status.update(label=f"{status_label} (exit {rc})", state="error")

        # Persist log in session state so it survives the st.rerun() that
        # follows a successful run (rerun wipes all widget state).
        st.session_state["_last_run_log"] = "\n".join(lines)
        st.session_state["_last_run_rc"] = rc

        return rc, result_path


def _start_background_subprocess(args: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        args,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    log.close()
    return process.pid


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
    suite_dir = st.sidebar.text_input(
        "experiment_suites_dir",
        value=str(DEFAULT_SUITE_DIR),
    )
    corpora_root = st.sidebar.text_input(
        "path_to_corpora",
        value="./corpora/scraped_data",
    )
    model_options = _model_options()
    default_model_index = model_options.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_options else 0
    examiner_model = st.sidebar.selectbox(
        "examiner_model",
        model_options,
        index=default_model_index,
        key="examiner_model",
    )
    num_ctx = st.sidebar.number_input("num_ctx", min_value=1024, max_value=131072, value=8192, step=1024)

    if st.sidebar.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    return {
        "experiment_dir": Path(experiment_dir),
        "analysis_dir": Path(analysis_dir),
        "suite_dir": Path(suite_dir),
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
    st.dataframe(grouped, width="stretch", hide_index=True)


def _tab_runs(df: pd.DataFrame, analyses, runs) -> tuple[pd.DataFrame, list[str]]:
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
    if "created_at" in filtered.columns:
        filtered = filtered.sort_values(
            "created_at",
            ascending=False,
            na_position="last",
        )

    display_cols = [
        "run_date",
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
    event = st.dataframe(
        filtered[available],
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="runs_table",
    )
    st.caption(f"{len(filtered)} run(s) match filters")

    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        selected_row = filtered.iloc[selected_rows[0]]
        st.divider()
        _render_run_details(filtered, analyses, runs, selected_row["run_id"])
    else:
        st.info("Select a run row to view its details here.")

    return filtered, list(filtered["run_id"])


def _latest_runs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    latest = df.copy()
    if "created_at" in latest.columns:
        latest = latest.sort_values("created_at", ascending=False, na_position="last")
    return latest


def _render_latest_runs_panel(df: pd.DataFrame, analyses, runs) -> None:
    with st.expander("Runs", expanded=True):
        latest = _latest_runs_dataframe(df)
        if latest.empty:
            st.info("No experiment runs yet.")
            return

        display_cols = [
            "run_date",
            "system_name",
            "corpus",
            "level",
            "question_id",
            "model",
            "verdict",
            "support_rate",
            "helpfulness_rating",
        ]
        available = [c for c in display_cols if c in latest.columns]
        event = st.dataframe(
            latest[available],
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="latest_runs_table",
        )
        st.caption(f"{len(latest)} run(s)")

        selected_rows = event.selection.rows if event.selection else []
        if selected_rows:
            selected_row = latest.iloc[selected_rows[0]]
            st.divider()
            _render_run_details(latest, analyses, runs, selected_row["run_id"])
        else:
            st.info("Select a run row to inspect it here.")


def _tab_charts(df: pd.DataFrame) -> None:
    st.subheader("Charts")
    if df.empty or df["support_rate"].dropna().empty:
        st.info("Run analyze first to populate charts.")
        return

    plot_names = list(ALL_PLOTS.keys())
    selected = st.multiselect("Charts to show", plot_names, default=plot_names[:4])
    for name in selected:
        st.plotly_chart(ALL_PLOTS[name](df), width="stretch")


def _render_trace_steps(run) -> None:
    """Render trace steps (reasoning, tool calls, messages) for a run."""
    if run is None or run.trace is None or not run.trace.steps:
        return
    steps = run.trace.steps
    with st.expander(f"Execution trace ({len(steps)} steps)", expanded=False):
        for i, step in enumerate(steps, 1):
            label = f"Step {i} — `{step.type}`"
            if step.type == "reasoning":
                with st.expander(label, expanded=False):
                    st.text(step.content or "")
            elif step.type == "tool_call":
                with st.expander(label + (f" `{step.name}`" if step.name else ""), expanded=False):
                    if step.input:
                        st.code(step.input, language="json")
            elif step.type == "tool_result":
                with st.expander(label + (f" `{step.name}`" if step.name else ""), expanded=False):
                    st.text(step.output or "")
            else:
                with st.expander(label, expanded=step.type == "agent_message"):
                    st.write(step.content or "")


def _render_run_details(df: pd.DataFrame, analyses, runs, run_id: str) -> None:
    st.subheader("Run details")
    if df.empty:
        st.info("No runs.")
        return

    row = df[df["run_id"] == run_id].iloc[0]
    run = next((r for r in runs if r.run_id == run_id), None)

    runtime = row.get("execution_time_s")
    runtime_label = f"{runtime:.2f}s" if pd.notna(runtime) else "—"
    tool_calls = row.get("tool_call_count")
    tool_calls_label = int(tool_calls) if pd.notna(tool_calls) else "—"

    cols = st.columns(6)
    cols[0].metric("System", row["system_name"])
    cols[1].metric("Corpus", row["corpus"])
    cols[2].metric("Level", row["level"] if pd.notna(row["level"]) else "—")
    cols[3].metric("Model", row.get("model", "—"))
    cols[4].metric("Runtime", runtime_label)
    cols[5].metric("Tool calls", tool_calls_label)
    st.caption(f"Run date: {_format_run_date(row.get('created_at'))}")

    st.markdown(f"**Question:** {row['question_text']}")
    with st.expander("Answer", expanded=True):
        st.write(row["answer_text"] or "(empty)")

    _render_trace_steps(run)

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
        st.dataframe(pd.DataFrame(claim_rows), width="stretch", hide_index=True)


def _tab_run_details(df: pd.DataFrame, analyses, runs) -> None:
    if df.empty:
        st.subheader("Run details")
        st.info("No runs.")
        return

    run_id = st.selectbox("Run ID", df["run_id"].tolist())
    _render_run_details(df, analyses, runs, run_id)


def _format_run_date(value) -> str:
    if pd.isna(value):
        return "—"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_question_meta(questions_file: str) -> tuple[list[str], dict[str, str]]:
    """Return (ordered_ids, id→question_text) parsed from a questions JSON file."""
    path = Path(questions_file)
    if not path.is_file():
        return [], {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], {}
    ids: list[str] = []
    meta: dict[str, str] = {}
    for q in raw:
        qid = q.get("id")
        if not qid:
            continue
        ids.append(qid)
        meta[qid] = q.get("question", "")
    return ids, meta


def _load_question_rows(questions_file: str) -> list[dict]:
    path = Path(questions_file)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [row for row in raw if row.get("id")]


def _suite_config_paths(suite_dir: Path) -> list[Path]:
    if not suite_dir.exists():
        return []
    return sorted(p for p in suite_dir.glob("*.json") if not p.name.endswith(".state.json"))


def _suite_state_paths(suite_dir: Path) -> list[Path]:
    if not suite_dir.exists():
        return []
    return sorted(p for p in suite_dir.glob("*.state.json"))


def _suite_paths(suite_dir: Path, config: ExperimentSuiteConfig) -> tuple[Path, Path, Path]:
    config_path = suite_dir / f"{suite_slug(config.name)}.json"
    state_path = default_state_path(config_path, config)
    launcher_log_path = state_path.with_suffix(".launcher.log")
    return config_path, state_path, launcher_log_path


def _result_files_from_suite_states(state_paths: list[Path]) -> list[str]:
    paths: set[str] = set()
    for state_path in state_paths:
        try:
            state = load_suite_state(state_path)
        except Exception:
            continue
        for task in state.tasks:
            if task.result_path and Path(task.result_path).is_file():
                paths.add(str(Path(task.result_path).resolve()))
    return sorted(paths)


def _run_id_from_result_path(result_path: str | None) -> str | None:
    run = _run_from_result_path(result_path)
    return run.run_id if run else None


def _run_from_result_path(result_path: str | None) -> RunResult | None:
    if not result_path:
        return None
    path = Path(result_path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    return RunResult.model_validate_json(line)
                except Exception:
                    continue
    except OSError:
        return None
    return None


def _dataframe_for_single_run(run: RunResult) -> pd.DataFrame:
    tokens_total = (
        run.metrics.tokens.total
        if run.metrics and run.metrics.tokens
        else None
    )
    return pd.DataFrame.from_records([
        {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "run_date": run.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "system_name": run.system_name.value,
            "corpus": run.corpus.value,
            "model": run.model,
            "reasoning_enabled": run.reasoning_enabled,
            "question_id": run.question_id,
            "question_text": run.question_text,
            "level": _parse_question_level(run.question_id),
            "automation_level": run.automation_level.value,
            "answer_text": run.answer_text,
            "execution_time_s": run.metrics.execution_time_s if run.metrics else None,
            "step_count": run.metrics.step_count if run.metrics else None,
            "tool_call_count": run.metrics.tool_call_count if run.metrics else None,
            "tokens_total": tokens_total,
            "corpus_used": run.metrics.corpus_used if run.metrics else None,
        }
    ])


def _parse_question_level(question_id: str) -> int | None:
    match = re.search(r"_L(\d+)_", question_id or "")
    return int(match.group(1)) if match else None


def _suite_state_dataframe(state_path: Path) -> tuple[dict, pd.DataFrame]:
    if not state_path.is_file():
        return {}, pd.DataFrame()
    state = load_suite_state(state_path)
    summary = summarize_suite_state(state)
    rows = [
        {
            "index": task.index,
            "status": task.status.value,
            "model": task.model,
            "corpus": task.corpus.value,
            "level": task.level,
            "question_id": task.question_id,
            "system": task.system.value,
            "result_path": task.result_path,
            "run_id": _run_id_from_result_path(task.result_path),
            "return_code": task.return_code,
            "error": task.error,
        }
        for task in state.tasks
    ]
    return summary, pd.DataFrame(rows)


def _render_suite_progress(state_path: Path, df: pd.DataFrame, analyses, runs) -> None:
    summary, tasks_df = _suite_state_dataframe(state_path)
    if not summary:
        st.info("No persisted suite state yet.")
        return

    total = int(summary["total"])
    completed = int(summary["completed"])
    progress = completed / total if total else 0.0
    st.progress(progress, text=f"{completed}/{total} task(s) completed")

    cols = st.columns(6)
    cols[0].metric("Total", total)
    cols[1].metric("Succeeded", summary[SuiteTaskStatus.SUCCEEDED.value])
    cols[2].metric("Failed", summary[SuiteTaskStatus.FAILED.value])
    cols[3].metric("Running", summary[SuiteTaskStatus.RUNNING.value])
    cols[4].metric("Pending", summary[SuiteTaskStatus.PENDING.value])
    cols[5].metric("Cancelled", summary[SuiteTaskStatus.CANCELLED.value])

    if summary["cancel_requested"]:
        st.warning("Cancellation has been requested. The suite stops before starting the next task.")

    if not tasks_df.empty:
        event = st.dataframe(
            tasks_df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="suite_tasks_table",
        )
        selected_rows = event.selection.rows if event.selection else []
        if selected_rows:
            selected = tasks_df.iloc[selected_rows[0]]
            run_id = selected.get("run_id")
            if run_id and not df.empty and run_id in set(df["run_id"]):
                st.divider()
                _render_run_details(df, analyses, runs, run_id)
            elif selected.get("result_path"):
                run = _run_from_result_path(selected.get("result_path"))
                if run is None:
                    st.warning("The selected task result file could not be read.")
                else:
                    st.divider()
                    _render_run_details(
                        _dataframe_for_single_run(run),
                        analyses,
                        [run],
                        run.run_id,
                    )
            else:
                st.info("The selected task has not produced a result file yet.")


def _suite_task_preview(config: ExperimentSuiteConfig) -> pd.DataFrame:
    tasks = build_suite_tasks(config)
    return pd.DataFrame(
        [
            {
                "index": task.index,
                "model": task.model,
                "corpus": task.corpus.value,
                "level": task.level,
                "question_id": task.question_id,
                "system": task.system.value,
                "command": _shell_command(task.command),
            }
            for task in tasks
        ]
    )


def _compact_names(values: list[str], *, max_items: int = 3) -> str:
    if not values:
        return "none"
    if len(values) <= max_items:
        return "-".join(values)
    return "-".join(values[:max_items]) + f"-plus{len(values) - max_items}"


def _suggest_suite_name(
    *,
    systems: list[SystemName],
    models: list[str],
    corpus_selections: list[SuiteCorpusSelection],
) -> str:
    corpus_parts: list[str] = []
    for selection in corpus_selections:
        part = selection.corpus.value.replace("_", "-")
        if selection.levels:
            part += "-l" + "-".join(str(level) for level in sorted(selection.levels))
        if selection.question_ids:
            part += f"-q{len(selection.question_ids)}"
        corpus_parts.append(part)

    raw_name = "-".join(
        [
            _compact_names(corpus_parts, max_items=2),
            _compact_names(models, max_items=2),
            _compact_names([system.value for system in systems], max_items=3),
        ]
    )
    return suite_slug(raw_name)


def _tab_experiment_suite(cfg: dict, df: pd.DataFrame, analyses, runs) -> None:
    st.subheader("Experiment suite")
    st.caption(
        "Configure a persisted multi-system benchmark suite. The suite expands to one "
        "`experiment-runner run` command per question/system/model task."
    )

    suite_dir: Path = cfg["suite_dir"]
    suite_dir.mkdir(parents=True, exist_ok=True)
    saved_configs = _suite_config_paths(suite_dir)
    selected_config_path: Path | None = None
    loaded_config: ExperimentSuiteConfig | None = None

    if saved_configs:
        selected_config_path = st.selectbox(
            "Saved suite config",
            saved_configs,
            format_func=lambda p: p.name,
            index=0,
        )
        if selected_config_path and st.button("Load selected suite", width="stretch"):
            st.session_state["loaded_suite_config_path"] = str(selected_config_path)
            st.rerun()

    loaded_path = st.session_state.get("loaded_suite_config_path")
    if loaded_path and Path(loaded_path).is_file():
        try:
            loaded_config = load_suite_config(loaded_path)
            st.info(f"Loaded suite config: `{loaded_path}`")
        except Exception as exc:
            st.warning(f"Could not load suite config: {exc}")

    if "suite_name_input" not in st.session_state:
        st.session_state["suite_name_input"] = loaded_config.name if loaded_config else "thesis-smoke-suite"
    if loaded_config and st.session_state.get("suite_loaded_name_for") != loaded_path:
        st.session_state["suite_name_input"] = loaded_config.name
        st.session_state["suite_loaded_name_for"] = loaded_path
    if "suite_name_pending" in st.session_state:
        st.session_state["suite_name_input"] = st.session_state.pop("suite_name_pending")

    suite_name_container = st.container()

    default_systems = loaded_config.systems if loaded_config else [SystemName.ACE]
    selected_systems = st.multiselect(
        "systems",
        _AUTOMATED_SYSTEMS,
        default=[s for s in default_systems if s in _AUTOMATED_SYSTEMS],
        format_func=_system_label,
    )

    model_options = _model_options()
    default_models = loaded_config.models if loaded_config else [DEFAULT_MODEL]
    selected_models = st.multiselect(
        "models",
        model_options,
        default=[m for m in default_models if m in model_options] or [model_options[0]],
    )

    default_corpora = [c.corpus.value for c in loaded_config.corpora] if loaded_config else ["solar_system_wiki"]
    selected_corpora = st.multiselect(
        "corpora",
        list(CORPUS_DEFAULTS.keys()),
        default=[c for c in default_corpora if c in CORPUS_DEFAULTS],
    )

    loaded_by_corpus = {c.corpus.value: c for c in loaded_config.corpora} if loaded_config else {}
    corpus_selections: list[SuiteCorpusSelection] = []
    for corpus in selected_corpora:
        defaults = loaded_by_corpus.get(corpus)
        default_q_file, default_corpus_path = CORPUS_DEFAULTS[corpus]
        with st.expander(f"{corpus} questions", expanded=True):
            cols = st.columns(2)
            questions_file = cols[0].text_input(
                f"{corpus} questions_file",
                value=defaults.questions_file if defaults else default_q_file,
                key=f"suite_qf_{corpus}",
            )
            path_to_corpora = cols[1].text_input(
                f"{corpus} path_to_corpora",
                value=defaults.path_to_corpora if defaults else default_corpus_path,
                key=f"suite_corpora_{corpus}",
            )
            rows = _load_question_rows(questions_file)
            levels = sorted({int(row.get("level", 0)) for row in rows if row.get("level") is not None})
            selected_levels = st.multiselect(
                f"{corpus} levels (empty = all)",
                levels,
                default=defaults.levels if defaults else [],
                key=f"suite_levels_{corpus}",
            )
            ids = [row["id"] for row in rows]
            questions_meta = {row["id"]: row.get("question", "") for row in rows}
            selected_ids = st.multiselect(
                f"{corpus} question_ids (empty = all selected levels)",
                ids,
                default=defaults.question_ids if defaults else [],
                format_func=lambda i: f"{i} — {questions_meta.get(i, '')[:80]}",
                key=f"suite_ids_{corpus}",
            )
            corpus_selections.append(
                SuiteCorpusSelection(
                    corpus=Corpus(corpus),
                    questions_file=questions_file,
                    path_to_corpora=path_to_corpora,
                    levels=selected_levels,
                    question_ids=selected_ids,
                )
            )

    suggested_name = _suggest_suite_name(
        systems=selected_systems,
        models=selected_models,
        corpus_selections=corpus_selections,
    )
    with suite_name_container:
        suite_name = st.text_input("suite_name", key="suite_name_input")
        cols = st.columns([1, 3])
        if cols[0].button("Generate suite name", width="stretch"):
            st.session_state["suite_name_pending"] = suggested_name
            st.rerun()
        cols[1].caption(f"Suggested: `{suggested_name}`")

    cols = st.columns(3)
    num_ctx = cols[0].number_input(
        "suite num_ctx",
        min_value=1024,
        max_value=131072,
        value=loaded_config.num_ctx if loaded_config else 8192,
        step=1024,
    )
    reasoning_enabled = cols[1].checkbox(
        "suite reasoning_enabled",
        value=loaded_config.reasoning_enabled if loaded_config else False,
    )
    no_trace = cols[2].checkbox(
        "suite no_trace",
        value=loaded_config.no_trace if loaded_config else True,
    )

    output_dir = st.text_input(
        "suite output_dir",
        value=loaded_config.output_dir if loaded_config else str(cfg["experiment_dir"]),
    )

    config = ExperimentSuiteConfig(
        name=suite_name,
        systems=selected_systems,
        models=selected_models,
        corpora=corpus_selections,
        output_dir=output_dir,
        num_ctx=int(num_ctx),
        reasoning_enabled=reasoning_enabled,
        no_trace=no_trace,
    )
    if loaded_config:
        config.suite_id = loaded_config.suite_id

    config_path, state_path, launcher_log_path = _suite_paths(suite_dir, config)
    run_args = _build_suite_run_args(str(config_path), str(state_path))
    cancel_args = _build_suite_cancel_args(str(state_path))

    st.markdown("### Generated task commands")
    try:
        preview = _suite_task_preview(config)
        st.dataframe(preview, width="stretch", hide_index=True)
        st.caption(f"{len(preview)} task command(s) generated.")
    except Exception as exc:
        preview = pd.DataFrame()
        st.error(f"Could not build suite preview: {exc}")

    st.markdown("### Suite control")
    st.code(_shell_command(run_args), language="bash")
    cols = st.columns(4)
    if cols[0].button("Save config", width="stretch", disabled=preview.empty):
        save_suite_config(config_path, config)
        st.success(f"Saved {config_path}")
    if cols[1].button("Run / resume suite", type="primary", width="stretch", disabled=preview.empty):
        save_suite_config(config_path, config)
        pid = _start_background_subprocess(run_args, launcher_log_path)
        st.success(f"Started suite process pid={pid}.")
        st.rerun()
    if cols[2].button("Cancel suite", width="stretch", disabled=not state_path.exists()):
        result = subprocess.run(cancel_args, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            st.warning("Cancellation requested.")
        else:
            st.error(result.stderr or result.stdout or f"Cancel failed with {result.returncode}")
    if cols[3].button("Refresh suite status", width="stretch"):
        st.rerun()

    st.caption(f"Config: `{config_path}`")
    st.caption(f"State: `{state_path}`")
    _render_suite_progress(state_path, df, analyses, runs)

    if launcher_log_path.exists():
        with st.expander("Suite launcher output", expanded=False):
            st.code(launcher_log_path.read_text(encoding="utf-8")[-8000:], language="text")


def _tab_run_experiment(cfg: dict, df: pd.DataFrame, analyses, runs) -> None:
    st.subheader("Run experiment")
    st.caption(
        "Launches `experiment-runner run` as a subprocess (same code path as "
        "`make ace_experiment_single`). Live progress is parsed from the "
        "`[i/total]` markers it writes to stderr."
    )

    message = st.session_state.pop("last_experiment_message", None)
    if message:
        st.success(message)

    form_col, runs_col = st.columns([1, 1], gap="large")
    with form_col:
        _render_run_experiment_form(cfg)
    with runs_col:
        _render_latest_runs_panel(df, analyses, runs)


def _render_run_experiment_form(cfg: dict) -> None:
    kind_filter = st.radio(
        "Filter by runner type",
        options=["all", "automated", "manual"],
        format_func=lambda v: {
            "all": "All systems",
            "automated": "⚙ Automated only",
            "manual": "✋ Manual only",
        }[v],
        index=1,
        horizontal=True,
        key="system_kind_filter",
    )
    if kind_filter == "automated":
        visible = [s for s in _ALL_SYSTEMS if SYSTEM_AUTOMATION_LEVELS[s] == AutomationLevel.FULL]
    elif kind_filter == "manual":
        visible = [s for s in _ALL_SYSTEMS if SYSTEM_AUTOMATION_LEVELS[s] == AutomationLevel.MANUAL]
    else:
        visible = _ALL_SYSTEMS

    cols = st.columns(2)
    system_obj = cols[0].selectbox(
        "system",
        visible,
        format_func=_system_label,
        key="system_select",
    )
    system = system_obj.value
    corpus = cols[1].selectbox("corpus", list(CORPUS_DEFAULTS.keys()), index=0)

    default_q_file, default_corpora_root = CORPUS_DEFAULTS[corpus]

    cols = st.columns(2)
    questions_file = cols[0].text_input(
        "questions_file",
        value=default_q_file,
        key=f"qf_{corpus}",
    )
    path_to_corpora = cols[1].text_input(
        "path_to_corpora",
        value=default_corpora_root,
        key=f"corpora_{corpus}",
    )

    available_ids, questions_meta = _load_question_meta(questions_file)
    if questions_file and not available_ids:
        st.warning(f"Could not load any question IDs from {questions_file}.")

    selected_ids = st.multiselect(
        f"question_ids ({len(available_ids)} available — empty = run all)",
        available_ids,
        default=[],
        format_func=lambda i: f"{i} — {questions_meta.get(i, '')[:80]}",
    )

    cols = st.columns(3)
    with cols[0]:
        model = _select_model("model", key="experiment_model")
    num_ctx = cols[1].number_input(
        "num_ctx",
        min_value=1024,
        max_value=131072,
        value=8192,
        step=1024,
    )
    automation_level = cols[2].selectbox(
        "automation_level",
        ["full", "partial", "manual"],
        index=0,
    )

    cols = st.columns(3)
    output_dir = cols[0].text_input("output_dir", value=str(cfg["experiment_dir"]))
    reasoning_enabled = cols[1].checkbox("reasoning_enabled", value=False)
    no_trace = cols[2].checkbox("no_trace", value=False)

    dry_run = st.checkbox("dry_run (validate without executing)", value=False)

    n_to_run = len(selected_ids) if selected_ids else len(available_ids)
    args = _build_experiment_run_args(
        system=system,
        corpus=corpus,
        questions_file=questions_file,
        output_dir=output_dir,
        model=model,
        num_ctx=int(num_ctx),
        path_to_corpora=path_to_corpora,
        automation_level=automation_level,
        selected_ids=selected_ids,
        reasoning_enabled=reasoning_enabled,
        no_trace=no_trace,
        dry_run=dry_run,
    )
    _render_shell_command_preview(_shell_command(args))

    cols = st.columns([3, 1])
    cols[0].caption(
        f"Will run **{n_to_run}** question(s) with `{system}` on corpus `{corpus}`."
    )
    if cols[1].button("▶ Run experiment", type="primary", width="stretch", disabled=n_to_run == 0):
        rc, result_path = _run_experiment_subprocess(
            args,
            expected_total=n_to_run,
            status_label="Dry run" if dry_run else "Running experiment",
        )
        if rc == 0 and not dry_run:
            st.cache_data.clear()
            if result_path:
                st.session_state["last_experiment_message"] = f"Wrote {result_path}."
                st.session_state["_last_result_path"] = result_path
            else:
                st.session_state["last_experiment_message"] = "Run complete."
                st.session_state.pop("_last_result_path", None)
            st.rerun()

    if "_last_run_log" in st.session_state:
        rc = st.session_state.get("_last_run_rc", 0)
        st.expander("Last run output", expanded=rc != 0).code(
            st.session_state["_last_run_log"], language="text"
        )

    if "_last_result_path" in st.session_state:
        _render_run_errors(st.session_state["_last_result_path"])


def _tab_actions(cfg: dict, filtered: pd.DataFrame) -> None:
    st.subheader("Actions")
    st.caption("Both actions shell out to the `result-processor` CLI so behaviour matches Makefile/terminal usage.")

    st.markdown("### Analyze")
    cols = st.columns([1, 1, 2])
    resume = cols[0].checkbox("Resume (skip cached)", value=True)
    only_filtered = cols[1].checkbox("Only filtered runs", value=False)

    if cols[2].button("▶ Run analyze", type="primary", width="stretch"):
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

    st.markdown("### Analyze experiment suites")
    suite_states = _suite_state_paths(cfg["suite_dir"])
    if not suite_states:
        st.info("No suite state files found yet.")
    else:
        selected_states = st.multiselect(
            "Suite states",
            suite_states,
            default=suite_states[:1],
            format_func=lambda p: p.name,
        )
        default_analysis_name = f"{cfg['examiner_model'].replace(':', '-')}-suite-analysis"
        analysis_name = st.text_input("Analysis run name", value=default_analysis_name)
        suite_resume = st.checkbox("Resume suite analysis (skip cached)", value=True)
        result_files = _result_files_from_suite_states(selected_states)
        st.caption(f"{len(result_files)} result file(s) found across selected suite state(s).")

        if st.button(
            "▶ Analyze selected suites",
            type="primary",
            width="stretch",
            disabled=not result_files or not analysis_name.strip(),
        ):
            output_dir = cfg["analysis_dir"] / suite_slug(analysis_name)
            args = _cli_path() + [
                "analyze",
                "--experiment-results-dir", str(cfg["experiment_dir"]),
                "--output-dir", str(output_dir),
                "--path-to-corpora", cfg["corpora_root"],
                "--examiner-model", cfg["examiner_model"],
                "--num-ctx", str(cfg["num_ctx"]),
                "--input-files",
                *result_files,
            ]
            if not suite_resume:
                args.append("--no-resume")
            rc = _run_subprocess(args, "Running suite analysis")
            if rc == 0:
                st.cache_data.clear()
                st.success(f"Analysis written to {output_dir}/")

    st.markdown("### Visualize")
    cols = st.columns([3, 1])
    output_dir = cols[0].text_input("Output directory", value="./data/figures")
    formats = cols[1].multiselect("Formats", ["html", "png", "pdf", "svg"], default=["html"])
    if st.button("▶ Run visualize", type="primary", width="stretch"):
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

    (
        suite_tab,
        run_tab,
        overview_tab,
        runs_tab,
        charts_tab,
        details_tab,
        actions_tab,
    ) = st.tabs(
        ["Experiment suite", "Run experiment", "Overview", "Runs", "Charts", "Run details", "Actions"]
    )

    with suite_tab:
        _tab_experiment_suite(cfg, df, analyses, runs)

    with run_tab:
        _tab_run_experiment(cfg, df, analyses, runs)

    with overview_tab:
        _tab_overview(df, runs, analyses)

    with runs_tab:
        filtered, _ = _tab_runs(df, analyses, runs)

    with charts_tab:
        _tab_charts(df)

    with details_tab:
        _tab_run_details(df, analyses, runs)

    with actions_tab:
        _tab_actions(cfg, filtered)


if __name__ == "__main__" or __name__ == "__page__":
    main()
