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


def test_suite_tasks_expand_cross_product_and_sort_by_model_corpus_question_system(tmp_path) -> None:
    config = _config(tmp_path)

    tasks = suite.build_suite_tasks(config)

    assert [(t.model, t.level, t.question_id, t.system.value) for t in tasks] == [
        ("qwen3:4b", 1, "ss_L1_001", "ace"),
        ("qwen3:4b", 1, "ss_L1_001", "clawcode"),
        ("qwen3:4b", 2, "ss_L2_002", "ace"),
        ("qwen3:4b", 2, "ss_L2_002", "clawcode"),
        ("qwen3:14b", 1, "ss_L1_001", "ace"),
        ("qwen3:14b", 1, "ss_L1_001", "clawcode"),
        ("qwen3:14b", 2, "ss_L2_002", "ace"),
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


def test_suite_cancel_marks_state_for_cooperative_stop(tmp_path, capsys) -> None:
    config = _config(tmp_path)
    state_path = tmp_path / "suite.state.json"
    suite.save_suite_state(state_path, suite.build_suite_state(config))

    suite.run_suite_cancel(argparse.Namespace(state=str(state_path)))

    assert suite.load_suite_state(state_path).cancel_requested is True
    assert "cancel requested" in capsys.readouterr().out


def test_experiment_runner_parses_suite_subcommands() -> None:
    args = runner_main.parse_args(["suite", "run", "--config", "suite.json", "--state", "state.json"])

    assert args.command == "suite"
    assert args.suite_command == "run"
    assert args.config == "suite.json"
    assert args.state == "state.json"
