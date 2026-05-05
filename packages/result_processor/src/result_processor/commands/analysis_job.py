from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiment_runner.models.result import RunResult

from result_processor.analysis.io import iter_run_results, load_existing_run_ids
from result_processor.analysis.pipeline import analyze_directory
from result_processor.models.analysis_job import (
    AnalysisJobState,
    AnalysisJobTask,
    AnalysisTaskStatus,
)


def load_analysis_job_state(path: str | Path) -> AnalysisJobState:
    return AnalysisJobState.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_analysis_job_state(path: str | Path, state: AnalysisJobState) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(timezone.utc)
    out.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _output_file_for_run_file(output_dir: str | Path, source_file: str | Path) -> str:
    return str((Path(output_dir) / Path(source_file).name).resolve())


def _task_key(source_file: str, run_id: str) -> tuple[str, str]:
    return str(Path(source_file).resolve()), run_id


def build_analysis_job_state(
    *,
    job_name: str,
    experiment_results_dir: str,
    output_dir: str,
    path_to_corpora: str,
    examiner_model: str,
    num_ctx: int,
    input_files: list[str],
    resume: bool,
    log_path: str | None = None,
) -> AnalysisJobState:
    tasks: list[AnalysisJobTask] = []
    for input_file in input_files:
        source = Path(input_file).resolve()
        output_file = _output_file_for_run_file(output_dir, source)
        existing = load_existing_run_ids(Path(output_file)) if resume else set()
        for run in iter_run_results(source):
            status = AnalysisTaskStatus.SKIPPED if run.run_id in existing else AnalysisTaskStatus.PENDING
            tasks.append(
                AnalysisJobTask(
                    run_id=run.run_id,
                    source_file=str(source),
                    output_file=output_file,
                    status=status,
                )
            )
    return AnalysisJobState(
        job_name=job_name,
        experiment_results_dir=experiment_results_dir,
        output_dir=output_dir,
        path_to_corpora=path_to_corpora,
        examiner_model=examiner_model,
        num_ctx=num_ctx,
        input_files=[str(Path(f).resolve()) for f in input_files],
        resume=resume,
        log_path=log_path,
        tasks=tasks,
    )


def reconcile_analysis_job_state(state: AnalysisJobState) -> AnalysisJobState:
    previous = {
        _task_key(task.source_file, task.run_id): task
        for task in state.tasks
    }
    rebuilt = build_analysis_job_state(
        job_name=state.job_name,
        experiment_results_dir=state.experiment_results_dir,
        output_dir=state.output_dir,
        path_to_corpora=state.path_to_corpora,
        examiner_model=state.examiner_model,
        num_ctx=state.num_ctx,
        input_files=state.input_files,
        resume=state.resume,
        log_path=state.log_path,
    )
    for task in rebuilt.tasks:
        old = previous.get(_task_key(task.source_file, task.run_id))
        if old and old.status in {AnalysisTaskStatus.ANALYZED, AnalysisTaskStatus.FAILED, AnalysisTaskStatus.CANCELLED}:
            task.status = old.status
            task.error = old.error
            task.started_at = old.started_at
            task.finished_at = old.finished_at
    state.tasks = rebuilt.tasks
    return state


def summarize_analysis_job_state(state: AnalysisJobState) -> dict[str, int | str | bool]:
    counts = {status.value: 0 for status in AnalysisTaskStatus}
    for task in state.tasks:
        counts[task.status.value] += 1
    completed = (
        counts[AnalysisTaskStatus.ANALYZED.value]
        + counts[AnalysisTaskStatus.SKIPPED.value]
        + counts[AnalysisTaskStatus.FAILED.value]
        + counts[AnalysisTaskStatus.CANCELLED.value]
    )
    return {
        "job_name": state.job_name,
        "total": len(state.tasks),
        "completed": completed,
        "cancel_requested": state.cancel_requested,
        **counts,
    }


def run_analysis_job(state_path: str | Path) -> AnalysisJobState:
    path = Path(state_path)
    state = reconcile_analysis_job_state(load_analysis_job_state(path))
    state.cancel_requested = False
    save_analysis_job_state(path, state)

    log_path = Path(state.log_path or path.with_suffix(".log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    input_files = state.input_files

    def should_cancel() -> bool:
        return load_analysis_job_state(path).cancel_requested

    def mark(status: str, run: RunResult, source: Path, error: str | None) -> None:
        current = load_analysis_job_state(path)
        key = _task_key(str(source), run.run_id)
        task = next((t for t in current.tasks if _task_key(t.source_file, t.run_id) == key), None)
        if task is None:
            return
        if status == "running":
            task.status = AnalysisTaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)
            task.error = None
        elif status == "analyzed":
            task.status = AnalysisTaskStatus.ANALYZED
            task.finished_at = datetime.now(timezone.utc)
        elif status == "skipped":
            task.status = AnalysisTaskStatus.SKIPPED
            task.finished_at = datetime.now(timezone.utc)
        elif status == "failed":
            task.status = AnalysisTaskStatus.FAILED
            task.error = error
            task.finished_at = datetime.now(timezone.utc)
        save_analysis_job_state(path, current)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== analysis job {state.job_name} ===\n")
        log.flush()
        stdout = sys.stdout
        stderr = sys.stderr
        try:
            sys.stdout = log
            sys.stderr = log
            analyze_directory(
                experiment_results_dir=state.experiment_results_dir,
                output_dir=state.output_dir,
                path_to_corpora=state.path_to_corpora,
                examiner_model=state.examiner_model,
                num_ctx=state.num_ctx,
                input_files=input_files,
                resume=state.resume,
                progress_callback=mark,
                should_cancel=should_cancel,
                continue_on_error=True,
            )
        finally:
            sys.stdout = stdout
            sys.stderr = stderr

    state = load_analysis_job_state(path)
    if state.cancel_requested:
        for task in state.tasks:
            if task.status in {AnalysisTaskStatus.PENDING, AnalysisTaskStatus.RUNNING}:
                task.status = AnalysisTaskStatus.CANCELLED
                task.finished_at = datetime.now(timezone.utc)
    save_analysis_job_state(path, state)
    return state


def run_analysis_job_run(args: argparse.Namespace) -> None:
    state = load_analysis_job_state(args.state)
    state.active_pid = os.getpid()
    save_analysis_job_state(args.state, state)
    try:
        final = run_analysis_job(args.state)
    finally:
        state = load_analysis_job_state(args.state)
        state.active_pid = None
        save_analysis_job_state(args.state, state)
    sys.stdout.write(json.dumps(summarize_analysis_job_state(final), indent=2) + "\n")


def run_analysis_job_status(args: argparse.Namespace) -> None:
    state = load_analysis_job_state(args.state)
    sys.stdout.write(json.dumps(summarize_analysis_job_state(state), indent=2) + "\n")


def run_analysis_job_cancel(args: argparse.Namespace) -> None:
    state = load_analysis_job_state(args.state)
    state.cancel_requested = True
    save_analysis_job_state(args.state, state)
    sys.stdout.write(f"cancel requested for {args.state}\n")
