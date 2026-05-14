from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from result_processor.tests.conftest import analysis_result, run_payload, write_jsonl
from result_processor.visualization.loader import build_dataframe, build_dataframe_for_files, load_analyses, load_runs
from result_processor.visualization.pipeline import visualize_results
from result_processor.visualization.plots import ALL_PLOTS, CHARTS
from result_processor.visualization.tables import ALL_TABLES


def _write_result_files(tmp_path):
    experiment_dir = tmp_path / "experiment"
    analysis_dir = tmp_path / "analysis"
    write_jsonl(
        experiment_dir / "runs.jsonl",
        [
            run_payload(run_id="r1", question_id="ss_L1_001"),
            run_payload(
                run_id="r2",
                question_id="ss_L2_001",
                created_at="2026-04-27T10:00:00Z",
                answer_text="",
                answer_error="model timed out",
            ),
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
    assert df.loc[df["run_id"] == "r1", "exact_match"].iloc[0] == 1.0
    assert df.loc[df["run_id"] == "r1", "f1"].iloc[0] == 1.0
    assert df.loc[df["run_id"] == "r1", "precision"].iloc[0] == 1.0
    assert df.loc[df["run_id"] == "r1", "recall"].iloc[0] == 1.0
    assert pd.isna(df.loc[df["run_id"] == "r2", "support_rate"].iloc[0])
    assert df.loc[df["run_id"] == "r1", "analysis_time_s"].iloc[0] == 1.25
    assert pd.isna(df.loc[df["run_id"] == "r2", "analysis_time_s"].iloc[0])
    assert df.loc[df["run_id"] == "r1", "answer_char_count"].iloc[0] == len(
        "[Jupiter is a planet.] [file:planets.md, lines:0-1]"
    )
    assert df.loc[df["run_id"] == "r2", "answer_char_count"].iloc[0] == 0
    assert df.loc[df["run_id"] == "r2", "answer_error"].iloc[0] == "model timed out"
    assert bool(df.loc[df["run_id"] == "r2", "has_answer_error"].iloc[0]) is True


def test_loader_builds_dataframe_for_selected_result_files(tmp_path) -> None:
    experiment_dir, analysis_dir = _write_result_files(tmp_path)

    df = build_dataframe_for_files([experiment_dir / "runs.jsonl"], analysis_dir)

    assert df["run_id"].tolist() == ["r1", "r2"]
    assert df.loc[df["run_id"] == "r1", "support_rate"].iloc[0] == 1.0


def test_plot_builders_return_figures_for_populated_and_empty_inputs(tmp_path) -> None:
    experiment_dir, analysis_dir = _write_result_files(tmp_path)
    df = build_dataframe(experiment_dir, analysis_dir)

    for name, builder in ALL_PLOTS.items():
        fig = builder(df)
        assert fig is not None, name
        assert hasattr(fig, "to_dict")


def test_system_color_mapping_is_distinct_and_stable_across_charts() -> None:
    df = pd.DataFrame(
        [
            {
                "system_name": "chatgpt_codex",
                "model": "qwen3:4b",
                "corpus": "solar_system_wiki",
                "question_id": "q1",
                "answer_char_count": 100,
                "execution_time_s": 10,
                "support_rate": 0.1,
            },
            {
                "system_name": "ace",
                "model": "qwen3:4b",
                "corpus": "solar_system_wiki",
                "question_id": "q2",
                "answer_char_count": 200,
                "execution_time_s": 20,
                "support_rate": 0.2,
            },
            {
                "system_name": "anythingllm",
                "model": "qwen3:4b",
                "corpus": "solar_system_wiki",
                "question_id": "q3",
                "answer_char_count": 300,
                "execution_time_s": 30,
                "support_rate": 0.3,
            },
        ]
    )

    answer_fig = ALL_PLOTS["answer_chars_by_model_system"](df)
    time_fig = ALL_PLOTS["execution_time_by_model_system"](df)
    support_fig = ALL_PLOTS["support_by_system"](df)

    answer_colors = {trace.name: trace.marker.color for trace in answer_fig.data}
    time_colors = {trace.name: trace.marker.color for trace in time_fig.data}
    support_colors = {trace.name: trace.marker.color for trace in support_fig.data}
    assert len(set(answer_colors.values())) == 3
    assert answer_colors == time_colors
    assert answer_colors == support_colors
    assert support_fig.layout.showlegend is False


def test_table_builders_return_latex_for_analyzed_data_and_empty_for_missing_metrics(tmp_path) -> None:
    experiment_dir, analysis_dir = _write_result_files(tmp_path)
    df = build_dataframe(experiment_dir, analysis_dir)
    empty_metrics = df.assign(support_rate=pd.NA)

    for name, builder in ALL_TABLES.items():
        latex = builder(df)
        assert "\\begin{table}" in latex, name
        assert builder(empty_metrics) == ""


def test_visualize_results_writes_prefixed_html_pdf_manifest_and_latex_tables(tmp_path, monkeypatch) -> None:
    def fake_write_image(_fig, file, *args, **kwargs) -> None:
        if hasattr(file, "write"):
            file.write(b"%PDF-1.4\n")
            return
        Path(file).write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("plotly.graph_objects.Figure.write_image", fake_write_image)
    experiment_dir, analysis_dir = _write_result_files(tmp_path)
    out_dir = tmp_path / "figures"

    visualize_results(str(experiment_dir), str(analysis_dir), str(out_dir), formats=["html"])

    assert (out_dir / "plots" / "C01_support_by_system.html").is_file()
    assert (out_dir / "plots" / "C01_support_by_system.pdf").is_file()
    assert (out_dir / "plots" / "C05_time_vs_answer_chars.html").is_file()
    assert (out_dir / "plots" / "C11_error_rate_by_system.pdf").is_file()
    assert (out_dir / "plots" / "C19_execution_time_vs_analysis_time.html").is_file()
    assert (out_dir / "plots" / "C20_answer_chars_by_model_system.html").is_file()
    assert (out_dir / "plots" / "C26_time_vs_answer_chars_by_system.pdf").is_file()
    manifest = json.loads((out_dir / "charts_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == len(CHARTS) == 26
    assert manifest[0] == {
        "id": "C01",
        "slug": "support_by_system",
        "lv_title": "Vidējais atbalsta īpatsvars pa sistēmām",
        "html_file": "C01_support_by_system.html",
        "pdf_file": "C01_support_by_system.pdf",
        "latex_label": "fig:c01-support-by-system",
    }
    assert manifest[19] == {
        "id": "C20",
        "slug": "answer_chars_by_model_system",
        "lv_title": "Gala atbildes garuma sadalījums pa A1 modeli un sistēmu",
        "html_file": "C20_answer_chars_by_model_system.html",
        "pdf_file": "C20_answer_chars_by_model_system.pdf",
        "latex_label": "fig:c20-answer-chars-by-model-system",
    }
    assert manifest[25] == {
        "id": "C26",
        "slug": "time_vs_answer_chars_by_system",
        "lv_title": "Izpildes laiks pret gala atbildes garumu pa sistēmām",
        "html_file": "C26_time_vs_answer_chars_by_system.html",
        "pdf_file": "C26_time_vs_answer_chars_by_system.pdf",
        "latex_label": "fig:c26-time-vs-answer-chars-by-system",
    }
    assert (out_dir / "tables" / "per_system_summary.tex").is_file()
