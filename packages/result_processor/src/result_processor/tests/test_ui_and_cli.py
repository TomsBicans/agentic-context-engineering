from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import pytest

from experiment_runner.models.enums import Corpus, SystemName
from experiment_runner.models.suite import ExperimentSuiteConfig, ExperimentSuiteState, SuiteCorpusSelection, SuiteTask
from result_processor import main as result_main
from result_processor.commands.analyze import run_analyze
from result_processor.commands.dashboard import run_dashboard
from result_processor.commands.visualize import run_visualize
from result_processor.tests.conftest import run_payload, write_jsonl
from result_processor.ui import streamlit_app as ui


def test_parse_ollama_list_extracts_model_names() -> None:
    output = """NAME                          ID              SIZE      MODIFIED
qwen3:14b                     bdbd181c33f2    9.3 GB    2 hours ago
qwen2.5-coder:14b-instruct    9ec8897f747e    9.0 GB    2 hours ago
"""

    assert ui._parse_ollama_list(output) == ["qwen3:14b", "qwen2.5-coder:14b-instruct"]


def test_model_options_include_default_when_ollama_has_no_default(monkeypatch) -> None:
    monkeypatch.setattr(ui, "_load_ollama_models", lambda: ["qwen3:14b"])

    assert ui._model_options() == ["qwen3:4b", "qwen3:14b"]


def test_query_ollama_models_returns_empty_on_command_failure(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("ollama missing")

    monkeypatch.setattr(ui.subprocess, "run", fail)

    assert ui._query_ollama_models() == []


def test_build_experiment_run_args_includes_selected_options(monkeypatch) -> None:
    monkeypatch.setattr(ui.sys, "executable", "/usr/bin/python")

    args = ui._build_experiment_run_args(
        system="ace",
        corpus="scipy",
        questions_file="./questions.json",
        output_dir="./out",
        model="qwen3:4b",
        num_ctx=16384,
        path_to_corpora="./corpora",
        automation_level="partial",
        selected_ids=["q1", "q2"],
        reasoning_enabled=True,
        no_trace=True,
        dry_run=True,
    )

    assert args == [
        "/usr/bin/python",
        "-m",
        "experiment_runner.main",
        "run",
        "--system",
        "ace",
        "--corpus",
        "scipy",
        "--questions-file",
        "./questions.json",
        "--output-dir",
        "./out",
        "--model",
        "qwen3:4b",
        "--num-ctx",
        "16384",
        "--path-to-corpora",
        "./corpora",
        "--automation-level",
        "partial",
        "--question-ids",
        "q1",
        "q2",
        "--reasoning-enabled",
        "--no-trace",
        "--dry-run",
    ]


def test_build_suite_run_and_cancel_args(monkeypatch) -> None:
    monkeypatch.setattr(ui.sys, "executable", "/usr/bin/python")

    assert ui._build_suite_run_args("suite.json", "suite.state.json") == [
        "/usr/bin/python",
        "-m",
        "experiment_runner.main",
        "suite",
        "run",
        "--config",
        "suite.json",
        "--state",
        "suite.state.json",
    ]
    assert ui._build_suite_cancel_args("suite.state.json") == [
        "/usr/bin/python",
        "-m",
        "experiment_runner.main",
        "suite",
        "cancel",
        "--state",
        "suite.state.json",
    ]


def test_suite_paths_use_slug_and_default_state_name(tmp_path) -> None:
    config = ExperimentSuiteConfig(
        name="Thesis Smoke Suite",
        systems=[SystemName.ACE],
        models=["qwen3:4b"],
        corpora=[
            SuiteCorpusSelection(
                corpus=Corpus.SOLAR_SYSTEM_WIKI,
                questions_file="questions.json",
                path_to_corpora="corpora",
            )
        ],
    )

    config_path, state_path, launcher_log_path = ui._suite_paths(tmp_path, config)

    assert config_path == tmp_path / "thesis-smoke-suite.json"
    assert state_path == tmp_path / "thesis-smoke-suite.state.json"
    assert launcher_log_path == tmp_path / "thesis-smoke-suite.state.launcher.log"


def test_suggest_suite_name_uses_selected_dimensions() -> None:
    name = ui._suggest_suite_name(
        systems=[SystemName.ACE, SystemName.CLAWCODE],
        models=["qwen3:8b"],
        corpus_selections=[
            SuiteCorpusSelection(
                corpus=Corpus.SOLAR_SYSTEM_WIKI,
                questions_file="questions.json",
                path_to_corpora="corpora",
                levels=[2],
            )
        ],
    )

    assert name == "solar-system-wiki-l2-qwen3-8b-ace-clawcode"


def test_result_files_from_suite_states_returns_existing_result_paths(tmp_path) -> None:
    existing = tmp_path / "result.jsonl"
    existing.write_text("{}\n", encoding="utf-8")
    missing = tmp_path / "missing.jsonl"
    state = ExperimentSuiteState(
        suite_id="suite-1",
        suite_name="suite",
        tasks=[
            SuiteTask(
                task_id="t1",
                index=1,
                system=SystemName.ACE,
                model="qwen3:4b",
                corpus=Corpus.SOLAR_SYSTEM_WIKI,
                questions_file="questions.json",
                path_to_corpora="corpora",
                question_id="ss_L1_001",
                question_text="Question?",
                level=1,
                command=["python"],
                result_path=str(existing),
            ),
            SuiteTask(
                task_id="t2",
                index=2,
                system=SystemName.CLAWCODE,
                model="qwen3:4b",
                corpus=Corpus.SOLAR_SYSTEM_WIKI,
                questions_file="questions.json",
                path_to_corpora="corpora",
                question_id="ss_L1_001",
                question_text="Question?",
                level=1,
                command=["python"],
                result_path=str(missing),
            ),
        ],
    )
    state_path = tmp_path / "suite.state.json"
    state_path.write_text(state.model_dump_json(), encoding="utf-8")

    assert ui._result_files_from_suite_states([state_path]) == [str(existing.resolve())]


def test_run_id_from_result_path_reads_first_valid_jsonl_row(tmp_path) -> None:
    path = tmp_path / "result.jsonl"
    path.write_text("\n{bad json}\n" + json.dumps(run_payload(run_id="run-123")) + "\n", encoding="utf-8")

    assert ui._run_id_from_result_path(str(path)) == "run-123"
    assert ui._run_id_from_result_path(str(tmp_path / "missing.jsonl")) is None


def test_dataframe_for_single_run_contains_detail_fields(tmp_path) -> None:
    path = tmp_path / "result.jsonl"
    path.write_text(json.dumps(run_payload(run_id="run-123")) + "\n", encoding="utf-8")
    run = ui._run_from_result_path(str(path))

    df = ui._dataframe_for_single_run(run)

    assert df.loc[0, "run_id"] == "run-123"
    assert df.loc[0, "system_name"] == "ace"
    assert df.loc[0, "execution_time_s"] == 12.5
    assert df.loc[0, "tool_call_count"] == 2


def test_shell_command_quotes_arguments() -> None:
    assert ui._shell_command(["python", "-m", "mod", "--output-dir", "path with space"]) == (
        "python -m mod --output-dir 'path with space'"
    )


def test_format_run_date_handles_missing_and_timezone_values() -> None:
    assert ui._format_run_date(pd.NA) == "—"
    assert ui._format_run_date(pd.Timestamp("2026-04-26T11:55:51Z")) == "2026-04-26 11:55:51 UTC"


def test_latest_runs_dataframe_sorts_newest_first_without_limiting_rows() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["old", "new", "middle"],
            "created_at": pd.to_datetime(
                [
                    "2026-04-26T10:00:00Z",
                    "2026-04-28T10:00:00Z",
                    "2026-04-27T10:00:00Z",
                ]
            ),
        }
    )

    latest = ui._latest_runs_dataframe(df)

    assert latest["run_id"].tolist() == ["new", "middle", "old"]


def test_locate_source_file_finds_run_id_and_ignores_bad_json(tmp_path) -> None:
    write_jsonl(tmp_path / "runs.jsonl", [run_payload(run_id="target")])
    (tmp_path / "bad.jsonl").write_text("{bad json}\n", encoding="utf-8")

    assert ui._locate_source_file(tmp_path, "target") == str(tmp_path / "runs.jsonl")
    assert ui._locate_source_file(tmp_path, "missing") is None


def test_parse_args_for_analyze_and_visualize() -> None:
    analyze = result_main.parse_args(
        ["analyze", "--path-to-corpora", "corpora", "--input-files", "one.jsonl", "two.jsonl", "--no-resume"]
    )
    visualize = result_main.parse_args(["visualize", "--formats", "html", "svg"])

    assert analyze.command == "analyze"
    assert analyze.input_files == ["one.jsonl", "two.jsonl"]
    assert analyze.resume is False
    assert visualize.command == "visualize"
    assert visualize.formats == ["html", "svg"]


def test_command_wrappers_delegate_to_pipelines(monkeypatch) -> None:
    calls = {}

    def fake_analyze_directory(**kwargs):
        calls["analyze"] = kwargs

    def fake_visualize_results(**kwargs):
        calls["visualize"] = kwargs

    monkeypatch.setattr("result_processor.analysis.pipeline.analyze_directory", fake_analyze_directory)
    monkeypatch.setattr("result_processor.visualization.pipeline.visualize_results", fake_visualize_results)

    run_analyze(
        argparse.Namespace(
            experiment_results_dir="exp",
            output_dir="analysis",
            path_to_corpora="corpora",
            examiner_model="qwen3:4b",
            num_ctx=8192,
            input_files=["runs.jsonl"],
            resume=False,
        )
    )
    run_visualize(
        argparse.Namespace(
            experiment_results_dir="exp",
            analysis_results_dir="analysis",
            output_dir="figures",
            formats=["html"],
        )
    )

    assert calls["analyze"]["resume"] is False
    assert calls["analyze"]["input_files"] == ["runs.jsonl"]
    assert calls["visualize"]["formats"] == ["html"]


def test_main_exits_with_error_for_value_error(monkeypatch, capsys) -> None:
    def fail(_args):
        raise ValueError("bad input")

    monkeypatch.setattr("result_processor.commands.visualize.run_visualize", fail)

    with pytest.raises(SystemExit) as exc:
        result_main.main(["visualize"])

    assert exc.value.code == 1
    assert "error: bad input" in capsys.readouterr().err


def test_dashboard_command_sets_env_and_streamlit_argv(monkeypatch, tmp_path) -> None:
    called = {}

    def fake_main():
        called["argv"] = list(sys.argv)
        return 0

    monkeypatch.setattr("streamlit.web.cli.main", fake_main)

    with pytest.raises(SystemExit) as exc:
        run_dashboard(
            argparse.Namespace(
                port=8765,
                experiment_results_dir=str(tmp_path / "exp"),
                analysis_results_dir=str(tmp_path / "analysis"),
            )
        )

    assert exc.value.code == 0
    assert os.environ["RP_EXPERIMENT_RESULTS_DIR"].endswith("exp")
    assert os.environ["RP_ANALYSIS_RESULTS_DIR"].endswith("analysis")
    assert called["argv"][:2] == ["streamlit", "run"]
    assert "--server.port" in called["argv"]
