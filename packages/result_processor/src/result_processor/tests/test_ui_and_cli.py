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
from result_processor.tests.conftest import analysis_result, run_payload, write_jsonl
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


def test_refresh_ollama_models_clears_cached_loader(monkeypatch) -> None:
    calls: list[bool] = []

    class Loader:
        @staticmethod
        def clear() -> None:
            calls.append(True)

    monkeypatch.setattr(ui, "_load_ollama_models", Loader)

    ui._refresh_ollama_models()

    assert calls == [True]


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


def test_build_analysis_job_run_and_cancel_args(monkeypatch) -> None:
    monkeypatch.setattr(ui.sys, "executable", "/usr/bin/python")

    assert ui._build_analysis_job_run_args("analysis.state.json") == [
        "/usr/bin/python",
        "-m",
        "result_processor.main",
        "analysis-job",
        "run",
        "--state",
        "analysis.state.json",
    ]
    assert ui._build_analysis_job_cancel_args("analysis.state.json") == [
        "/usr/bin/python",
        "-m",
        "result_processor.main",
        "analysis-job",
        "cancel",
        "--state",
        "analysis.state.json",
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


def test_analysis_job_paths_use_named_analysis_directory(tmp_path) -> None:
    output_dir, state_path, log_path = ui._analysis_job_paths(tmp_path, "Qwen3 Suite Analysis")

    assert output_dir == tmp_path / "qwen3-suite-analysis"
    assert state_path == tmp_path / "qwen3-suite-analysis.state.json"
    assert log_path == tmp_path / "qwen3-suite-analysis.log"


def test_analysis_result_dirs_only_include_dirs_with_jsonl(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "with-results").mkdir()
    (tmp_path / "with-results" / "run.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "top-level.jsonl").write_text("{}\n", encoding="utf-8")

    assert ui._analysis_result_dirs(tmp_path) == [tmp_path / "with-results"]
    assert ui._analysis_result_dirs(tmp_path / "missing") == []


def test_result_files_for_analysis_dir_matches_by_jsonl_name(tmp_path) -> None:
    analysis_dir = tmp_path / "analysis" / "suite-analysis"
    experiment_dir = tmp_path / "experiment"
    analysis_dir.mkdir(parents=True)
    experiment_dir.mkdir()
    (analysis_dir / "matched.jsonl").write_text("{}\n", encoding="utf-8")
    (analysis_dir / "missing.jsonl").write_text("{}\n", encoding="utf-8")
    (experiment_dir / "matched.jsonl").write_text("{}\n", encoding="utf-8")

    matched, missing = ui._result_files_for_analysis_dir(analysis_dir, experiment_dir)

    assert matched == [experiment_dir / "matched.jsonl"]
    assert missing == [experiment_dir / "missing.jsonl"]


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


def test_suite_widget_state_from_config_includes_per_corpus_selection_fields() -> None:
    config = ExperimentSuiteConfig(
        name="saved suite",
        systems=[SystemName.ACE, SystemName.CLAWCODE],
        models=["qwen3:4b"],
        corpora=[
            SuiteCorpusSelection(
                corpus=Corpus.SOLAR_SYSTEM_WIKI,
                questions_file="./corpora/questions/solar_system.json",
                path_to_corpora="./corpora/scraped_data/solar_system_wiki",
                levels=[2],
                question_ids=["ss_L2_005"],
            )
        ],
        output_dir="./data/experiment_results",
        num_ctx=16384,
        reasoning_enabled=True,
        no_trace=True,
    )

    state = ui._suite_widget_state_from_config(config, ["qwen3:4b"])

    assert state["suite_name_input"] == "saved suite"
    assert state["suite_systems"] == [SystemName.ACE, SystemName.CLAWCODE]
    assert state["suite_models"] == ["qwen3:4b"]
    assert state["suite_corpora"] == ["solar_system_wiki"]
    assert state["suite_qf_solar_system_wiki"] == "./corpora/questions/solar_system.json"
    assert state["suite_corpora_solar_system_wiki"] == "./corpora/scraped_data/solar_system_wiki"
    assert state["suite_levels_solar_system_wiki"] == [2]
    assert state["suite_ids_solar_system_wiki"] == ["ss_L2_005"]
    assert state["suite_num_ctx"] == 16384
    assert state["suite_reasoning_enabled"] is True
    assert state["suite_no_trace"] is True


def test_copy_suite_config_creates_independent_suite_identity() -> None:
    config = ExperimentSuiteConfig(
        suite_id="original-id",
        name="saved suite",
        systems=[SystemName.ACE, SystemName.CLAWCODE],
        models=["qwen3:4b"],
        corpora=[
            SuiteCorpusSelection(
                corpus=Corpus.SOLAR_SYSTEM_WIKI,
                questions_file="./corpora/questions/solar_system.json",
                path_to_corpora="./corpora/scraped_data/solar_system_wiki",
                levels=[2],
                question_ids=["ss_L2_005"],
            )
        ],
        output_dir="./data/experiment_results",
        num_ctx=16384,
        reasoning_enabled=True,
        no_trace=True,
    )

    copied = ui._copy_suite_config(config)

    assert copied.suite_id != config.suite_id
    assert copied.name == "saved-suite-copy"
    assert copied.systems == config.systems
    assert copied.models == config.models
    assert copied.corpora == config.corpora
    assert copied.output_dir == config.output_dir
    assert copied.num_ctx == config.num_ctx
    assert copied.reasoning_enabled == config.reasoning_enabled
    assert copied.no_trace == config.no_trace


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


def test_analysis_run_records_link_legacy_directory_by_suite_result_names(tmp_path) -> None:
    result_path = tmp_path / "experiment" / "run.jsonl"
    write_jsonl(result_path, [run_payload(run_id="r1")])
    analysis_dir = tmp_path / "analysis" / "legacy-analysis"
    write_jsonl(analysis_dir / "run.jsonl", [analysis_result(run_id="r1")])

    state = ExperimentSuiteState(
        suite_id="suite-1",
        suite_name="suite",
        config_path=str(tmp_path / "suite.json"),
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
                result_path=str(result_path),
            )
        ],
    )
    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    state_path = suite_dir / "suite.state.json"
    state_path.write_text(state.model_dump_json(), encoding="utf-8")
    suite_records = ui._suite_records(suite_dir)

    records = ui._analysis_run_records(tmp_path / "analysis", suite_records, tmp_path / "experiment")

    assert len(records) == 1
    assert records[0]["suite_key"] == str(state_path.resolve())
    assert records[0]["suite_name"] == "suite"
    assert records[0]["result_files"] == [result_path]


def test_create_analysis_export_zip_includes_suite_analysis_and_charts(tmp_path, monkeypatch) -> None:
    def fake_write_image(_fig, file, *args, **kwargs) -> None:
        file.write(b"%PDF-1.4\n")

    monkeypatch.setattr("plotly.graph_objects.Figure.write_image", fake_write_image)
    result_path = tmp_path / "experiment" / "run.jsonl"
    write_jsonl(result_path, [run_payload(run_id="r1")])
    analysis_dir = tmp_path / "analysis" / "analysis-a"
    write_jsonl(analysis_dir / "run.jsonl", [analysis_result(run_id="r1")])
    config_path = tmp_path / "suite.json"
    state_path = tmp_path / "suite.state.json"
    config_path.write_text("{}\n", encoding="utf-8")
    state_path.write_text("{}\n", encoding="utf-8")
    suite_record = {
        "suite_id": "suite-1",
        "suite_name": "suite",
        "state_path": state_path,
        "config_path": config_path,
    }
    analysis_record = {
        "analysis_name": "analysis-a",
        "analysis_dir": analysis_dir,
        "analysis_files": [analysis_dir / "run.jsonl"],
        "result_files": [result_path],
        "metadata": {},
    }
    analysis_df = ui._combined_analysis_dataframe([analysis_record])

    zip_bytes = ui._create_analysis_export_zip(suite_record, [analysis_record], analysis_df)

    import zipfile
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    assert "manifest.json" in names
    assert "suite/suite.json" in names
    assert "suite/suite.state.json" in names
    assert "tables/analysis_results.csv" in names
    assert "charts/charts_manifest.json" in names
    assert "charts/C05_time_vs_answer_chars.html" in names
    assert "charts/C05_time_vs_answer_chars.pdf" in names
    assert "charts/C11_error_rate_by_system.html" in names
    assert "charts/C11_error_rate_by_system.pdf" in names
    assert "charts/C20_answer_chars_by_model_system.html" in names
    assert "charts/C20_answer_chars_by_model_system.pdf" in names
    assert "charts/C26_time_vs_answer_chars_by_system.html" in names
    assert "charts/C26_time_vs_answer_chars_by_system.pdf" in names
    assert "experiment_data/analysis-a/run.jsonl" in names
    assert "analysis_results/analysis-a/run.jsonl" in names


def test_create_analysis_export_zip_reports_pdf_export_errors(tmp_path, monkeypatch) -> None:
    def fail_write_image(_fig, file, *args, **kwargs) -> None:
        raise RuntimeError("kaleido is missing chrome")

    monkeypatch.setattr("plotly.graph_objects.Figure.write_image", fail_write_image)
    result_path = tmp_path / "experiment" / "run.jsonl"
    write_jsonl(result_path, [run_payload(run_id="r1")])
    analysis_dir = tmp_path / "analysis" / "analysis-a"
    write_jsonl(analysis_dir / "run.jsonl", [analysis_result(run_id="r1")])
    config_path = tmp_path / "suite.json"
    state_path = tmp_path / "suite.state.json"
    config_path.write_text("{}\n", encoding="utf-8")
    state_path.write_text("{}\n", encoding="utf-8")
    suite_record = {
        "suite_id": "suite-1",
        "suite_name": "suite",
        "state_path": state_path,
        "config_path": config_path,
    }
    analysis_record = {
        "analysis_name": "analysis-a",
        "analysis_dir": analysis_dir,
        "analysis_files": [analysis_dir / "run.jsonl"],
        "result_files": [result_path],
        "metadata": {},
    }
    analysis_df = ui._combined_analysis_dataframe([analysis_record])
    export_errors: list[str] = []

    zip_bytes = ui._create_analysis_export_zip(suite_record, [analysis_record], analysis_df, export_errors)

    import zipfile
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        error_text = zf.read("charts/pdf_export_errors.txt").decode("utf-8")
    assert "charts/C01_support_by_system.html" in names
    assert "charts/C01_support_by_system.pdf" not in names
    assert "charts/pdf_export_errors.txt" in names
    assert export_errors
    assert "C01 support_by_system: PDF export failed: kaleido is missing chrome" in error_text


def test_create_analysis_export_zip_reports_progress(tmp_path, monkeypatch) -> None:
    def fake_write_image(_fig, file, *args, **kwargs) -> None:
        file.write(b"%PDF-1.4\n")

    monkeypatch.setattr("plotly.graph_objects.Figure.write_image", fake_write_image)
    result_path = tmp_path / "experiment" / "run.jsonl"
    write_jsonl(result_path, [run_payload(run_id="r1")])
    analysis_dir = tmp_path / "analysis" / "analysis-a"
    write_jsonl(analysis_dir / "run.jsonl", [analysis_result(run_id="r1")])
    config_path = tmp_path / "suite.json"
    state_path = tmp_path / "suite.state.json"
    config_path.write_text("{}\n", encoding="utf-8")
    state_path.write_text("{}\n", encoding="utf-8")
    suite_record = {
        "suite_id": "suite-1",
        "suite_name": "suite",
        "state_path": state_path,
        "config_path": config_path,
    }
    analysis_record = {
        "analysis_name": "analysis-a",
        "analysis_dir": analysis_dir,
        "analysis_files": [analysis_dir / "run.jsonl"],
        "result_files": [result_path],
        "metadata": {},
    }
    analysis_df = ui._combined_analysis_dataframe([analysis_record])
    progress_updates: list[tuple[int, int, str]] = []

    ui._create_analysis_export_zip(
        suite_record,
        [analysis_record],
        analysis_df,
        progress_callback=lambda current, total, label: progress_updates.append((current, total, label)),
    )

    assert progress_updates[0] == (0, 78, "Preparing chart export")
    assert progress_updates[-1] == (78, 78, "Exported PDF for C26 time_vs_answer_chars_by_system")
    assert any(update == (2, 78, "Exported HTML for C01 support_by_system") for update in progress_updates)


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


def test_dataframe_for_single_run_contains_optional_corpus_snapshot_fields(tmp_path) -> None:
    payload = run_payload(run_id="run-123")
    payload["corpus_snapshot"] = {
        "enabled": True,
        "source_corpus_path": "/source",
        "prepared_corpus_path": "/tmp/prepared",
        "file_count": 3,
        "total_bytes": 42,
    }
    path = tmp_path / "result.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    run = ui._run_from_result_path(str(path))

    df = ui._dataframe_for_single_run(run)

    assert bool(df.loc[0, "corpus_snapshot_enabled"]) is True
    assert df.loc[0, "corpus_snapshot_file_count"] == 3


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
    job = result_main.parse_args(["analysis-job", "run", "--state", "analysis.state.json"])

    assert analyze.command == "analyze"
    assert analyze.input_files == ["one.jsonl", "two.jsonl"]
    assert analyze.resume is False
    assert visualize.command == "visualize"
    assert visualize.formats == ["html", "svg"]
    assert job.command == "analysis-job"
    assert job.analysis_job_command == "run"
    assert job.state == "analysis.state.json"


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
