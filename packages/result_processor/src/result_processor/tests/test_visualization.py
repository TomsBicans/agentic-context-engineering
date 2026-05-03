from __future__ import annotations

import pandas as pd

from result_processor.tests.conftest import analysis_result, run_payload, write_jsonl
from result_processor.visualization.loader import build_dataframe, load_analyses, load_runs
from result_processor.visualization.pipeline import visualize_results
from result_processor.visualization.plots import ALL_PLOTS
from result_processor.visualization.tables import ALL_TABLES


def _write_result_files(tmp_path):
    experiment_dir = tmp_path / "experiment"
    analysis_dir = tmp_path / "analysis"
    write_jsonl(
        experiment_dir / "runs.jsonl",
        [
            run_payload(run_id="r1", question_id="ss_L1_001"),
            run_payload(run_id="r2", question_id="ss_L2_001", created_at="2026-04-27T10:00:00Z"),
        ],
    )
    write_jsonl(analysis_dir / "runs.jsonl", [analysis_result(run_id="r1")])
    return experiment_dir, analysis_dir


def test_loader_joins_runs_and_analyses_with_dates_and_levels(tmp_path) -> None:
    experiment_dir, analysis_dir = _write_result_files(tmp_path)

    runs = load_runs(experiment_dir)
    analyses = load_analyses(analysis_dir)
    df = build_dataframe(experiment_dir, analysis_dir)

    assert [run.run_id for run in runs] == ["r1", "r2"]
    assert list(analyses) == ["r1"]
    assert list(df["run_id"]) == ["r1", "r2"]
    assert list(df["level"]) == [1, 2]
    assert "run_date" in df.columns
    assert str(df.loc[df["run_id"] == "r1", "run_date"].iloc[0]).endswith("UTC")
    assert df.loc[df["run_id"] == "r1", "support_rate"].iloc[0] == 1.0
    assert pd.isna(df.loc[df["run_id"] == "r2", "support_rate"].iloc[0])


def test_plot_builders_return_figures_for_populated_and_empty_inputs(tmp_path) -> None:
    experiment_dir, analysis_dir = _write_result_files(tmp_path)
    df = build_dataframe(experiment_dir, analysis_dir)

    for name, builder in ALL_PLOTS.items():
        fig = builder(df)
        assert fig is not None, name
        assert hasattr(fig, "to_dict")


def test_table_builders_return_latex_for_analyzed_data_and_empty_for_missing_metrics(tmp_path) -> None:
    experiment_dir, analysis_dir = _write_result_files(tmp_path)
    df = build_dataframe(experiment_dir, analysis_dir)
    empty_metrics = df.assign(support_rate=pd.NA)

    for name, builder in ALL_TABLES.items():
        latex = builder(df)
        assert "\\begin{table}" in latex, name
        assert builder(empty_metrics) == ""


def test_visualize_results_writes_html_plots_and_latex_tables(tmp_path) -> None:
    experiment_dir, analysis_dir = _write_result_files(tmp_path)
    out_dir = tmp_path / "figures"

    visualize_results(str(experiment_dir), str(analysis_dir), str(out_dir), formats=["html"])

    assert (out_dir / "plots" / "support_by_system.html").is_file()
    assert (out_dir / "plots" / "verdict_breakdown.html").is_file()
    assert (out_dir / "tables" / "per_system_summary.tex").is_file()
