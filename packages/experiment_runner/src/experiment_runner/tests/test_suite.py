from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from experiment_runner.commands import suite
from experiment_runner import main as runner_main
from experiment_runner.models.enums import Corpus, SystemName
from experiment_runner.models.suite import (
    ExperimentSuiteConfig,
    ExperimentSuiteState,
    SuiteCorpusSelection,
    SuiteTask,
    SuiteTaskStatus,
)


def _write_questions(path: Path, prefix: str = "q") -> None:
    rows = [
        {
            "id": f"{prefix}_L2_002",
            "corpus": "test",
            "level": 2,
            "question": "Level two?",
            "expected_facts": [],
        },
        {
            "id": f"{prefix}_L1_001",
            "corpus": "test",
            "level": 1,
            "question": "Level one?",
            "expected_facts": [],
        },
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def _config(tmp_path: Path) -> ExperimentSuiteConfig:
    questions = tmp_path / "questions.json"
    _write_questions(questions, "ss")
    return ExperimentSuiteConfig(
        name="smoke suite",
        systems=[SystemName.ACE, SystemName.CLAWCODE],
        models=["qwen3:4b", "qwen3:14b"],
        corpora=[
            SuiteCorpusSelection(
                corpus=Corpus.SOLAR_SYSTEM_WIKI,
                questions_file=str(questions),
                path_to_corpora="./corpora/scraped_data/solar_system_wiki",
            )
        ],
        output_dir=str(tmp_path / "results"),
        no_trace=True,
    )


def test_suite_tasks_expand_cross_product_and_sort_by_model_corpus_system_question(tmp_path) -> None:
    config = _config(tmp_path)

    tasks = suite.build_suite_tasks(config)

    assert [(t.model, t.level, t.question_id, t.system.value) for t in tasks] == [
        ("qwen3:4b", 1, "ss_L1_001", "ace"),
        ("qwen3:4b", 2, "ss_L2_002", "ace"),
        ("qwen3:4b", 1, "ss_L1_001", "clawcode"),
        ("qwen3:4b", 2, "ss_L2_002", "clawcode"),
        ("qwen3:14b", 1, "ss_L1_001", "ace"),
        ("qwen3:14b", 2, "ss_L2_002", "ace"),
        ("qwen3:14b", 1, "ss_L1_001", "clawcode"),
        ("qwen3:14b", 2, "ss_L2_002", "clawcode"),
    ]
    assert tasks[0].command == [
        sys.executable,
        "-m",
        "experiment_runner.main",
        "run",
        "--system",
        "ace",
        "--corpus",
        "solar_system_wiki",
        "--questions-file",
        config.corpora[0].questions_file,
        "--output-dir",
        config.output_dir,
        "--model",
        "qwen3:4b",
        "--num-ctx",
        "8192",
        "--path-to-corpora",
        "./corpora/scraped_data/solar_system_wiki",
        "--automation-level",
        "full",
        "--question-ids",
        "ss_L1_001",
        "--no-trace",
    ]


def test_suite_rejects_disabled_and_manual_systems(tmp_path) -> None:
    config = _config(tmp_path)
    config.systems = [SystemName.OPENCLAW]

    with pytest.raises(ValueError, match="disabled"):
        suite.build_suite_tasks(config)

    config.systems = [SystemName.PERPLEXITY]
    with pytest.raises(ValueError, match="not fully automated"):
        suite.build_suite_tasks(config)


def test_suite_state_round_trips_and_summary_counts(tmp_path) -> None:
    config = _config(tmp_path)
    state_path = tmp_path / "suite.state.json"
    state = suite.build_suite_state(config, config_path=str(tmp_path / "suite.json"))
    state.tasks[0].status = SuiteTaskStatus.SUCCEEDED
    state.tasks[1].status = SuiteTaskStatus.FAILED

    suite.save_suite_state(state_path, state)
    loaded = suite.load_suite_state(state_path)
    summary = suite.summarize_suite_state(loaded)

    assert isinstance(loaded, ExperimentSuiteState)
    assert summary["total"] == 8
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["completed"] == 2


def test_reconcile_suite_state_preserves_completed_task_status(tmp_path) -> None:
    config = _config(tmp_path)
    state = suite.build_suite_state(config)
    state.tasks[0].status = SuiteTaskStatus.SUCCEEDED
    state.tasks[0].result_path = str(tmp_path / "result.jsonl")

    reconciled = suite.reconcile_suite_state(config, state)

    assert reconciled.tasks[0].status == SuiteTaskStatus.SUCCEEDED
    assert reconciled.tasks[0].result_path == str(tmp_path / "result.jsonl")
    assert [task.index for task in reconciled.tasks] == list(range(1, 9))


def test_reconcile_suite_state_preserves_existing_task_order(tmp_path) -> None:
    config = _config(tmp_path)
    old_order = suite.build_suite_tasks(config)
    old_order = sorted(old_order, key=lambda t: (t.model, t.corpus.value, t.level, t.question_id, t.system.value))
    for index, task in enumerate(old_order, 1):
        task.index = index
    state = ExperimentSuiteState(suite_id=config.suite_id, suite_name=config.name, tasks=old_order)

    reconciled = suite.reconcile_suite_state(config, state)

    assert [task.task_id for task in reconciled.tasks] == [task.task_id for task in old_order]
    assert [(task.index, task.task_id) for task in reconciled.tasks] == [
        (index, task.task_id)
        for index, task in enumerate(old_order, 1)
    ]


def test_suite_cancel_marks_state_for_cooperative_stop(tmp_path, capsys) -> None:
    config = _config(tmp_path)
    state_path = tmp_path / "suite.state.json"
    suite.save_suite_state(state_path, suite.build_suite_state(config))

    suite.run_suite_cancel(argparse.Namespace(state=str(state_path)))

    assert suite.load_suite_state(state_path).cancel_requested is True
    assert "cancel requested" in capsys.readouterr().out


def test_result_path_from_output_parses_prefix_from_last_matching_line() -> None:
    lines = [
        "[1/2] q1: Question one?",
        f"{suite.RESULT_PATH_PREFIX}/data/results/ace__solar__123.jsonl",
    ]
    assert suite._result_path_from_output(lines) == "/data/results/ace__solar__123.jsonl"


def test_result_path_from_output_returns_none_when_prefix_absent() -> None:
    assert suite._result_path_from_output(["no match here"]) is None


def test_run_suite_marks_timed_out_task_failed(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    config.task_timeout_s = 1
    task = SuiteTask(
        task_id="slow-task",
        index=1,
        system=SystemName.ACE,
        model="qwen3:4b",
        corpus=Corpus.SOLAR_SYSTEM_WIKI,
        questions_file=config.corpora[0].questions_file,
        path_to_corpora=config.corpora[0].path_to_corpora,
        question_id="ss_L1_001",
        question_text="Slow?",
        level=1,
        command=[sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
    )
    monkeypatch.setattr(suite, "build_suite_tasks", lambda _config: [task])

    state = suite.run_suite(config, tmp_path / "suite.state.json")

    assert state.tasks[0].status == SuiteTaskStatus.FAILED
    assert state.tasks[0].error == "Task timed out after 1s"
    assert state.active_pid is None
    assert "task timed out after 1s" in (tmp_path / "suite.state.log").read_text(encoding="utf-8")


def test_run_suite_timeout_handles_partial_stdout_line(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    config.task_timeout_s = 1
    task = SuiteTask(
        task_id="partial-output-task",
        index=1,
        system=SystemName.ACE,
        model="qwen3:4b",
        corpus=Corpus.SOLAR_SYSTEM_WIKI,
        questions_file=config.corpora[0].questions_file,
        path_to_corpora=config.corpora[0].path_to_corpora,
        question_id="ss_L1_001",
        question_text="Partial?",
        level=1,
        command=[
            sys.executable,
            "-c",
            "import sys, time; sys.stdout.write('partial output'); sys.stdout.flush(); time.sleep(10)",
        ],
    )
    monkeypatch.setattr(suite, "build_suite_tasks", lambda _config: [task])

    state = suite.run_suite(config, tmp_path / "suite.state.json")

    assert state.tasks[0].status == SuiteTaskStatus.FAILED
    assert state.tasks[0].error == "Task timed out after 1s"
    assert "partial output" in (tmp_path / "suite.state.log").read_text(encoding="utf-8")


def test_run_suite_tracks_immediate_success_result_path(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    result_path = tmp_path / "instant.jsonl"
    task = SuiteTask(
        task_id="instant-task",
        index=1,
        system=SystemName.ACE,
        model="qwen3:4b",
        corpus=Corpus.SOLAR_SYSTEM_WIKI,
        questions_file=config.corpora[0].questions_file,
        path_to_corpora=config.corpora[0].path_to_corpora,
        question_id="ss_L1_001",
        question_text="Instant?",
        level=1,
        command=[
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                f"p=Path({str(result_path)!r}); "
                "p.write_text('{}\\n', encoding='utf-8'); "
                f"print('{suite.RESULT_PATH_PREFIX}{result_path}')"
            ),
        ],
    )
    monkeypatch.setattr(suite, "build_suite_tasks", lambda _config: [task])

    state = suite.run_suite(config, tmp_path / "suite.state.json")

    assert state.tasks[0].status == SuiteTaskStatus.SUCCEEDED
    assert state.tasks[0].result_path == str(result_path)


def test_run_suite_persists_heartbeat_for_long_running_task(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    config.task_timeout_s = 2
    task = SuiteTask(
        task_id="heartbeat-task",
        index=1,
        system=SystemName.ACE,
        model="qwen3:4b",
        corpus=Corpus.SOLAR_SYSTEM_WIKI,
        questions_file=config.corpora[0].questions_file,
        path_to_corpora=config.corpora[0].path_to_corpora,
        question_id="ss_L1_001",
        question_text="Heartbeat?",
        level=1,
        command=[sys.executable, "-c", "import time; print('started', flush=True); time.sleep(0.4)"],
    )
    monkeypatch.setattr(suite, "TASK_HEARTBEAT_INTERVAL_S", 0.1)
    monkeypatch.setattr(suite, "build_suite_tasks", lambda _config: [task])

    state = suite.run_suite(config, tmp_path / "suite.state.json")

    assert state.tasks[0].last_heartbeat_at is not None
    assert state.tasks[0].started_at is not None
    assert state.tasks[0].last_heartbeat_at > state.tasks[0].started_at
    assert "task still running" in (tmp_path / "suite.state.log").read_text(encoding="utf-8")


def test_run_suite_rejects_success_without_result_path(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    task = SuiteTask(
        task_id="missing-result-task",
        index=1,
        system=SystemName.ACE,
        model="qwen3:4b",
        corpus=Corpus.SOLAR_SYSTEM_WIKI,
        questions_file=config.corpora[0].questions_file,
        path_to_corpora=config.corpora[0].path_to_corpora,
        question_id="ss_L1_001",
        question_text="Missing result?",
        level=1,
        command=[sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(suite, "build_suite_tasks", lambda _config: [task])

    state = suite.run_suite(config, tmp_path / "suite.state.json")

    assert state.tasks[0].status == SuiteTaskStatus.FAILED
    assert state.tasks[0].error == "Command exited successfully but did not report a non-empty result file."


def test_experiment_runner_parses_suite_subcommands() -> None:
    args = runner_main.parse_args(["suite", "run", "--config", "suite.json", "--state", "state.json"])

    assert args.command == "suite"
    assert args.suite_command == "run"
    assert args.config == "suite.json"
    assert args.state == "state.json"
