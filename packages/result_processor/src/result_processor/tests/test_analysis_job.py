from __future__ import annotations

import argparse
import json

from result_processor.commands import analysis_job
from result_processor.models.analysis_job import AnalysisTaskStatus
from result_processor.tests.conftest import analysis_result, run_payload, write_jsonl


def test_build_analysis_job_state_marks_cached_runs_as_skipped(tmp_path) -> None:
    runs_path = tmp_path / "experiment" / "runs.jsonl"
    output_dir = tmp_path / "analysis"
    write_jsonl(
        runs_path,
        [
            run_payload(run_id="r1"),
            run_payload(run_id="r2"),
        ],
    )
    write_jsonl(output_dir / "runs.jsonl", [analysis_result(run_id="r1")])

    state = analysis_job.build_analysis_job_state(
        job_name="suite analysis",
        experiment_results_dir=str(tmp_path / "experiment"),
        output_dir=str(output_dir),
        path_to_corpora="corpora",
        examiner_model="qwen3:4b",
        num_ctx=8192,
        input_files=[str(runs_path)],
        resume=True,
    )

    assert [task.run_id for task in state.tasks] == ["r1", "r2"]
    assert [task.status for task in state.tasks] == [
        AnalysisTaskStatus.SKIPPED,
        AnalysisTaskStatus.PENDING,
    ]


def test_analysis_job_state_writes_suite_metadata(tmp_path) -> None:
    runs_path = tmp_path / "experiment" / "runs.jsonl"
    output_dir = tmp_path / "analysis"
    write_jsonl(runs_path, [run_payload(run_id="r1")])

    state = analysis_job.build_analysis_job_state(
        job_name="suite analysis",
        experiment_results_dir=str(tmp_path / "experiment"),
        output_dir=str(output_dir),
        path_to_corpora="corpora",
        examiner_model="qwen3:4b",
        num_ctx=8192,
        input_files=[str(runs_path)],
        resume=True,
        suite_id="suite-1",
        suite_name="suite",
        suite_config_path="suite.json",
        suite_state_path="suite.state.json",
    )
    state_path = tmp_path / "analysis.state.json"

    analysis_job.save_analysis_job_state(state_path, state)

    metadata = json.loads((output_dir / "analysis.meta.json").read_text(encoding="utf-8"))
    assert metadata["suite_id"] == "suite-1"
    assert metadata["suite_name"] == "suite"
    assert metadata["analysis_name"] == "suite analysis"
    assert metadata["input_files"] == [str(runs_path.resolve())]


def test_analysis_job_cancel_and_summary(tmp_path, capsys) -> None:
    runs_path = tmp_path / "experiment" / "runs.jsonl"
    write_jsonl(runs_path, [run_payload(run_id="r1")])
    state = analysis_job.build_analysis_job_state(
        job_name="suite analysis",
        experiment_results_dir=str(tmp_path / "experiment"),
        output_dir=str(tmp_path / "analysis"),
        path_to_corpora="corpora",
        examiner_model="qwen3:4b",
        num_ctx=8192,
        input_files=[str(runs_path)],
        resume=True,
    )
    state_path = tmp_path / "analysis.state.json"
    analysis_job.save_analysis_job_state(state_path, state)

    analysis_job.run_analysis_job_cancel(argparse.Namespace(state=str(state_path)))
    loaded = analysis_job.load_analysis_job_state(state_path)
    summary = analysis_job.summarize_analysis_job_state(loaded)

    assert loaded.cancel_requested is True
    assert summary["pending"] == 1
    assert "cancel requested" in capsys.readouterr().out
