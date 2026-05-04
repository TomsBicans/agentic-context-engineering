from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from experiment_runner.commands.run import _load_questions
from experiment_runner.models.enums import AutomationLevel, SystemName
from experiment_runner.models.question import Question
from experiment_runner.models.suite import (
    ExperimentSuiteConfig,
    ExperimentSuiteState,
    SuiteTask,
    SuiteTaskStatus,
    default_state_path,
)
from experiment_runner.runners.registry import DISABLED_SYSTEMS, SYSTEM_AUTOMATION_LEVELS


_RESULT_RE_PREFIXES = ("results → ", "results -> ")


def load_suite_config(path: str | Path) -> ExperimentSuiteConfig:
    return ExperimentSuiteConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_suite_config(path: str | Path, config: ExperimentSuiteConfig) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_suite_state(path: str | Path) -> ExperimentSuiteState:
    return ExperimentSuiteState.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_suite_state(path: str | Path, state: ExperimentSuiteState) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(timezone.utc)
    out.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")


def validate_suite_config(config: ExperimentSuiteConfig) -> None:
    if not config.systems:
        raise ValueError("suite must include at least one system")
    if not config.models:
        raise ValueError("suite must include at least one model")
    if not config.corpora:
        raise ValueError("suite must include at least one corpus")

    invalid: list[str] = []
    for system in config.systems:
        if system in DISABLED_SYSTEMS:
            invalid.append(f"{system.value} is disabled")
        elif SYSTEM_AUTOMATION_LEVELS.get(system) != AutomationLevel.FULL:
            invalid.append(f"{system.value} is not fully automated")
    if invalid:
        raise ValueError("suite can only run enabled automated systems: " + "; ".join(invalid))


def _task_id(model: str, corpus: str, question_id: str, system: str) -> str:
    raw = f"{model}::{corpus}::{question_id}::{system}"
    return "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-").lower()


def _build_task_command(
    *,
    system: SystemName,
    corpus: str,
    questions_file: str,
    output_dir: str,
    model: str,
    num_ctx: int,
    path_to_corpora: str,
    question_id: str,
    reasoning_enabled: bool,
    no_trace: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiment_runner.main",
        "run",
        "--system",
        system.value,
        "--corpus",
        corpus,
        "--questions-file",
        questions_file,
        "--output-dir",
        output_dir,
        "--model",
        model,
        "--num-ctx",
        str(num_ctx),
        "--path-to-corpora",
        path_to_corpora,
        "--automation-level",
        AutomationLevel.FULL.value,
        "--question-ids",
        question_id,
    ]
    if reasoning_enabled:
        command.append("--reasoning-enabled")
    if no_trace:
        command.append("--no-trace")
    return command


def _selected_questions(selection, questions: list[Question]) -> list[Question]:
    selected = questions
    if selection.levels:
        allowed_levels = set(selection.levels)
        selected = [q for q in selected if q.level in allowed_levels]
    if selection.question_ids:
        allowed_ids = set(selection.question_ids)
        selected = [q for q in selected if q.id in allowed_ids]
        missing = allowed_ids - {q.id for q in selected}
        if missing:
            raise ValueError(
                f"Question IDs not found in {selection.questions_file}: {sorted(missing)}"
            )
    return selected


def build_suite_tasks(config: ExperimentSuiteConfig) -> list[SuiteTask]:
    validate_suite_config(config)
    raw_tasks: list[tuple[int, int, int, str, int, SuiteTask]] = []

    for corpus_index, selection in enumerate(config.corpora):
        questions = _selected_questions(selection, _load_questions(selection.questions_file, None))
        if not questions:
            raise ValueError(f"No questions selected for corpus {selection.corpus.value}")
        questions = sorted(questions, key=lambda q: (q.level, q.id))

        for model_index, model in enumerate(config.models):
            for question in questions:
                for system_index, system in enumerate(config.systems):
                    task = SuiteTask(
                        task_id=_task_id(model, selection.corpus.value, question.id, system.value),
                        index=0,
                        system=system,
                        model=model,
                        corpus=selection.corpus,
                        questions_file=selection.questions_file,
                        path_to_corpora=selection.path_to_corpora,
                        question_id=question.id,
                        question_text=question.question,
                        level=question.level,
                        command=_build_task_command(
                            system=system,
                            corpus=selection.corpus.value,
                            questions_file=selection.questions_file,
                            output_dir=config.output_dir,
                            model=model,
                            num_ctx=config.num_ctx,
                            path_to_corpora=selection.path_to_corpora,
                            question_id=question.id,
                            reasoning_enabled=config.reasoning_enabled,
                            no_trace=config.no_trace,
                        ),
                    )
                    raw_tasks.append((
                        model_index,
                        corpus_index,
                        question.level,
                        question.id,
                        system_index,
                        task,
                    ))

    sorted_tasks = [item[-1] for item in sorted(raw_tasks, key=lambda item: item[:-1])]
    for index, task in enumerate(sorted_tasks, 1):
        task.index = index
    return sorted_tasks


def build_suite_state(
    config: ExperimentSuiteConfig,
    *,
    config_path: str | None = None,
    log_path: str | None = None,
) -> ExperimentSuiteState:
    return ExperimentSuiteState(
        suite_id=config.suite_id,
        suite_name=config.name,
        config_path=config_path,
        log_path=log_path,
        tasks=build_suite_tasks(config),
    )


def reconcile_suite_state(config: ExperimentSuiteConfig, state: ExperimentSuiteState) -> ExperimentSuiteState:
    existing = {task.task_id: task for task in state.tasks}
    tasks: list[SuiteTask] = []
    for planned in build_suite_tasks(config):
        previous = existing.get(planned.task_id)
        if previous is not None:
            planned.status = previous.status
            planned.result_path = previous.result_path
            planned.error = previous.error
            planned.started_at = previous.started_at
            planned.finished_at = previous.finished_at
            planned.return_code = previous.return_code
        tasks.append(planned)
    state.tasks = tasks
    return state


def summarize_suite_state(state: ExperimentSuiteState) -> dict[str, int | str | bool]:
    counts = {status.value: 0 for status in SuiteTaskStatus}
    for task in state.tasks:
        counts[task.status.value] += 1
    completed = counts[SuiteTaskStatus.SUCCEEDED.value] + counts[SuiteTaskStatus.FAILED.value]
    completed += counts[SuiteTaskStatus.CANCELLED.value]
    return {
        "suite_id": state.suite_id,
        "suite_name": state.suite_name,
        "total": len(state.tasks),
        "completed": completed,
        "cancel_requested": state.cancel_requested,
        **counts,
    }


def _result_path_from_output(lines: Iterable[str]) -> str | None:
    for line in reversed(list(lines)):
        for prefix in _RESULT_RE_PREFIXES:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return None


def _task_result_exists(task: SuiteTask) -> bool:
    return bool(task.result_path and Path(task.result_path).is_file())


def run_suite(
    config: ExperimentSuiteConfig,
    state_path: str | Path,
    *,
    config_path: str | None = None,
) -> ExperimentSuiteState:
    path = Path(state_path)
    if path.exists():
        state = reconcile_suite_state(config, load_suite_state(path))
    else:
        log_path = str(path.with_suffix(".log"))
        state = build_suite_state(config, config_path=config_path, log_path=log_path)
    state.cancel_requested = False
    save_suite_state(path, state)

    log_path = Path(state.log_path or path.with_suffix(".log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("a", encoding="utf-8") as log:
        for task in state.tasks:
            if path.exists():
                state = load_suite_state(path)
                task = state.tasks[task.index - 1]
            if state.cancel_requested:
                break
            if task.status == SuiteTaskStatus.SUCCEEDED and _task_result_exists(task):
                continue

            task.status = SuiteTaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)
            task.finished_at = None
            task.error = None
            task.return_code = None
            state.active_pid = None
            save_suite_state(path, state)

            log.write(f"\n=== task {task.index}/{len(state.tasks)} {task.task_id} ===\n")
            log.write(" ".join(task.command) + "\n")
            log.flush()

            process = subprocess.Popen(
                task.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            state.active_pid = process.pid
            save_suite_state(path, state)

            lines: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip("\n")
                lines.append(line)
                log.write(line + "\n")
                log.flush()

                current = load_suite_state(path)
                if current.cancel_requested and process.poll() is None:
                    process.terminate()

            rc = process.wait()
            state = load_suite_state(path)
            task = state.tasks[task.index - 1]
            task.return_code = rc
            task.finished_at = datetime.now(timezone.utc)
            task.result_path = _result_path_from_output(lines) or task.result_path
            state.active_pid = None

            if state.cancel_requested:
                task.status = SuiteTaskStatus.CANCELLED
                task.error = "Suite cancellation requested."
            elif rc == 0:
                task.status = SuiteTaskStatus.SUCCEEDED
            else:
                task.status = SuiteTaskStatus.FAILED
                task.error = "\n".join(lines[-20:]) or f"Command exited with {rc}"
            save_suite_state(path, state)

    state = load_suite_state(path)
    state.active_pid = None
    save_suite_state(path, state)
    return state


def run_suite_plan(args: argparse.Namespace) -> None:
    config = load_suite_config(args.config)
    tasks = build_suite_tasks(config)
    if args.json:
        sys.stdout.write(json.dumps([task.model_dump(mode="json") for task in tasks], indent=2) + "\n")
        return
    for task in tasks:
        sys.stdout.write(f"[{task.index}/{len(tasks)}] {' '.join(task.command)}\n")


def run_suite_status(args: argparse.Namespace) -> None:
    state = load_suite_state(args.state)
    sys.stdout.write(json.dumps(summarize_suite_state(state), indent=2) + "\n")


def run_suite_cancel(args: argparse.Namespace) -> None:
    state = load_suite_state(args.state)
    state.cancel_requested = True
    save_suite_state(args.state, state)
    sys.stdout.write(f"cancel requested for {args.state}\n")


def run_suite_run(args: argparse.Namespace) -> None:
    config = load_suite_config(args.config)
    state_path = Path(args.state) if args.state else default_state_path(args.config, config)
    state = run_suite(config, state_path, config_path=args.config)
    sys.stdout.write(json.dumps(summarize_suite_state(state), indent=2) + "\n")
