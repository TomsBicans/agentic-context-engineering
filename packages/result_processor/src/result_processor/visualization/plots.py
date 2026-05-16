"""Plotly figure generators.

Each function takes a long-format DataFrame (one row per run) and returns a
plotly Figure. The orchestrator decides where to save it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


_COLOR_SEQUENCE = (
    px.colors.qualitative.Safe
    + px.colors.qualitative.Bold
    + px.colors.qualitative.Plotly
)


@dataclass(frozen=True)
class ChartSpec:
    id: str
    slug: str
    lv_title: str
    builder: Callable[[pd.DataFrame], go.Figure]

    @property
    def file_stem(self) -> str:
        # Stable chart IDs keep filenames deterministic for LaTeX references.
        return f"{self.id}_{self.slug}"

    @property
    def html_file(self) -> str:
        return f"{self.file_stem}.html"

    @property
    def pdf_file(self) -> str:
        return f"{self.file_stem}.pdf"

    @property
    def latex_label(self) -> str:
        return f"fig:{self.id.lower()}-{self.slug.replace('_', '-')}"

    def manifest_entry(self) -> dict[str, str]:
        return {
            "id": self.id,
            "slug": self.slug,
            "lv_title": self.lv_title,
            "html_file": self.html_file,
            "pdf_file": self.pdf_file,
            "latex_label": self.latex_label,
        }


def _category_values(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df:
        return []
    return sorted(str(value) for value in df[column].dropna().unique())


def _category_orders(df: pd.DataFrame, *columns: str) -> dict[str, list[str]]:
    return {column: values for column in columns if (values := _category_values(df, column))}


def _color_map(df: pd.DataFrame, column: str) -> dict[str, str]:
    # Stable per-chart color assignment: sorted legend values always map to the
    # same qualitative palette positions, avoiding grayscale/ambiguous exports.
    return {
        value: _COLOR_SEQUENCE[index % len(_COLOR_SEQUENCE)]
        for index, value in enumerate(_category_values(df, column))
    }


def _with_level_label(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(level_label=df["level"].astype(int).map(lambda level: f"L{level}"))


def _with_system_model_label(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(system_model=df["system_name"].astype(str) + " / " + df["model"].astype(str))


def support_rate_by_system(df: pd.DataFrame) -> go.Figure:
    grouped = (
        df.dropna(subset=["support_rate"])
        .groupby("system_name", as_index=False)["support_rate"]
        .mean()
        .sort_values("support_rate", ascending=False)
    )
    fig = px.bar(
        grouped,
        x="system_name",
        y="support_rate",
        color="system_name",
        color_discrete_map=_color_map(grouped, "system_name"),
        category_orders=_category_orders(grouped, "system_name"),
        title="Mean support rate per system",
        labels={"system_name": "System", "support_rate": "Support rate"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(showlegend=False)
    return fig


def support_rate_by_corpus_and_system(df: pd.DataFrame) -> go.Figure:
    grouped = (
        df.dropna(subset=["support_rate"])
        .groupby(["corpus", "system_name"], as_index=False)["support_rate"]
        .mean()
    )
    fig = px.bar(
        grouped,
        x="corpus",
        y="support_rate",
        color="system_name",
        color_discrete_map=_color_map(grouped, "system_name"),
        category_orders=_category_orders(grouped, "system_name"),
        barmode="group",
        title="Support rate per corpus, grouped by system",
        labels={"corpus": "Corpus", "support_rate": "Support rate", "system_name": "System"},
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def support_rate_by_level(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["support_rate", "level"])
    if subset.empty:
        return _empty("No level / support_rate data")
    grouped = (
        subset.groupby(["level", "system_name"], as_index=False)["support_rate"]
        .mean()
        .sort_values("level")
    )
    fig = px.line(
        grouped,
        x="level",
        y="support_rate",
        color="system_name",
        color_discrete_map=_color_map(grouped, "system_name"),
        category_orders=_category_orders(grouped, "system_name"),
        markers=True,
        title="Support rate by question difficulty level",
        labels={"level": "Difficulty level", "support_rate": "Support rate", "system_name": "System"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(dtick=1)
    return fig


def support_rate_by_level_model_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["support_rate", "level", "model", "system_name"])
    if subset.empty:
        return _empty("No support_rate / level / model / system_name data")
    grouped = (
        _with_system_model_label(subset)
        .groupby(["level", "system_model"], as_index=False)["support_rate"]
        .mean()
        .sort_values(["level", "system_model"])
    )
    fig = px.line(
        grouped,
        x="level",
        y="support_rate",
        color="system_model",
        color_discrete_map=_color_map(grouped, "system_model"),
        category_orders=_category_orders(grouped, "system_model"),
        markers=True,
        title="Support rate by difficulty level, A1 model, and system",
        labels={
            "level": "Difficulty level",
            "support_rate": "Support rate",
            "system_model": "System / A1 model",
        },
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(dtick=1)
    return fig


def support_rate_by_model_system_level(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["support_rate", "level", "model", "system_name"])
    if subset.empty:
        return _empty("No support_rate / level / model / system_name data")
    grouped = (
        _with_level_label(subset)
        .groupby(["system_name", "model", "level_label"], as_index=False)["support_rate"]
        .mean()
        .sort_values(["system_name", "model", "level_label"])
    )
    fig = px.bar(
        grouped,
        x="model",
        y="support_rate",
        color="level_label",
        color_discrete_map=_color_map(grouped, "level_label"),
        category_orders=_category_orders(grouped, "system_name", "model", "level_label"),
        barmode="group",
        facet_col="system_name",
        title="Support rate by A1 model, system, and difficulty level",
        labels={
            "model": "A1 model",
            "support_rate": "Support rate",
            "level_label": "Difficulty level",
            "system_name": "System",
        },
    )
    fig.update_yaxes(range=[0, 1])
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    return fig


def execution_time_by_level_model_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["execution_time_s", "level", "model", "system_name"])
    if subset.empty:
        return _empty("No execution_time_s / level / model / system_name data")
    subset = _with_level_label(subset)
    fig = px.box(
        subset,
        x="level_label",
        y="execution_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "level_label", "system_name", "model"),
        points="all",
        facet_col="model",
        hover_data=["question_id", "corpus", "has_answer_error"],
        title="Execution time by difficulty level, A1 model, and system",
        labels={
            "level_label": "Difficulty level",
            "execution_time_s": "Execution time (s)",
            "system_name": "System",
            "model": "A1 model",
        },
    )
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    return fig


def answer_chars_by_level_model_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["answer_char_count", "level", "model", "system_name"])
    if subset.empty:
        return _empty("No answer_char_count / level / model / system_name data")
    subset = _with_level_label(subset)
    fig = px.box(
        subset,
        x="level_label",
        y="answer_char_count",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "level_label", "system_name", "model"),
        points="all",
        facet_col="model",
        hover_data=["question_id", "corpus", "execution_time_s", "has_answer_error"],
        title="Final answer length by difficulty level, A1 model, and system",
        labels={
            "level_label": "Difficulty level",
            "answer_char_count": "Final answer length (characters)",
            "system_name": "System",
            "model": "A1 model",
        },
    )
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    return fig


def execution_time_vs_support(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["execution_time_s", "support_rate"])
    if subset.empty:
        return _empty("No execution_time / support_rate data")
    fig = px.scatter(
        subset,
        x="execution_time_s",
        y="support_rate",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        symbol="corpus",
        hover_data=["question_id", "model"],
        title="Execution time vs support rate",
        labels={"execution_time_s": "Execution time (s)", "support_rate": "Support rate"},
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def execution_time_vs_answer_chars(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["execution_time_s", "answer_char_count"])
    if subset.empty:
        return _empty("No execution_time / answer_char_count data")
    fig = px.scatter(
        subset,
        x="answer_char_count",
        y="execution_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        symbol="corpus",
        hover_data=["question_id", "model", "has_answer_error"],
        title="Execution time vs final answer length",
        labels={
            "answer_char_count": "Final answer length (characters)",
            "execution_time_s": "Execution time (s)",
            "system_name": "System",
            "corpus": "Corpus",
        },
    )
    return fig


def answer_chars_by_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["answer_char_count"])
    if subset.empty:
        return _empty("No answer_char_count data")
    fig = px.box(
        subset,
        x="system_name",
        y="answer_char_count",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        points="all",
        hover_data=["question_id", "model"],
        title="Final answer length distribution by system",
        labels={"system_name": "System", "answer_char_count": "Final answer length (characters)"},
    )
    fig.update_layout(showlegend=False)
    return fig


def execution_time_by_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["execution_time_s"])
    if subset.empty:
        return _empty("No execution_time_s data")
    fig = px.box(
        subset,
        x="system_name",
        y="execution_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        points="all",
        hover_data=["question_id", "model", "answer_char_count"],
        title="Execution time distribution by system",
        labels={"system_name": "System", "execution_time_s": "Execution time (s)"},
    )
    fig.update_layout(showlegend=False)
    return fig


def tool_calls_vs_execution_time(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["tool_call_count", "execution_time_s"])
    if subset.empty:
        return _empty("No tool_call_count / execution_time_s data")
    fig = px.scatter(
        subset,
        x="tool_call_count",
        y="execution_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        symbol="corpus",
        hover_data=["question_id", "model", "answer_char_count"],
        title="Tool calls vs execution time",
        labels={
            "tool_call_count": "Tool calls",
            "execution_time_s": "Execution time (s)",
            "system_name": "System",
            "corpus": "Corpus",
        },
    )
    return fig


def uncited_claims_by_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["claims_without_citation_count"])
    if subset.empty:
        return _empty("No claims_without_citation_count data")
    fig = px.box(
        subset,
        x="system_name",
        y="claims_without_citation_count",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        points="all",
        hover_data=["question_id", "model", "claims_total"],
        title="Uncited factual claims by system",
        labels={
            "system_name": "System",
            "claims_without_citation_count": "Uncited factual claims",
        },
    )
    fig.update_layout(showlegend=False)
    return fig


def claims_total_vs_support(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["claims_total", "support_rate"])
    if subset.empty:
        return _empty("No claims_total / support_rate data")
    fig = px.scatter(
        subset,
        x="claims_total",
        y="support_rate",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        symbol="corpus",
        hover_data=["question_id", "model", "claims_without_citation_count"],
        title="Total claims vs support rate",
        labels={
            "claims_total": "Total factual claims",
            "support_rate": "Support rate",
            "system_name": "System",
            "corpus": "Corpus",
        },
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def error_rate_by_system(df: pd.DataFrame) -> go.Figure:
    if "has_answer_error" not in df.columns or df.empty:
        return _empty("No answer error data")
    grouped = (
        df.groupby("system_name", as_index=False)["has_answer_error"]
        .mean()
        .rename(columns={"has_answer_error": "answer_error_rate"})
        .sort_values("answer_error_rate", ascending=False)
    )
    fig = px.bar(
        grouped,
        x="system_name",
        y="answer_error_rate",
        color="system_name",
        color_discrete_map=_color_map(grouped, "system_name"),
        category_orders=_category_orders(grouped, "system_name"),
        title="Answer-error rate by system",
        labels={"system_name": "System", "answer_error_rate": "Answer-error rate"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(showlegend=False)
    return fig


def tool_call_distribution(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["tool_call_count"])
    if subset.empty:
        return _empty("No tool_call_count data")
    fig = px.box(
        subset,
        x="system_name",
        y="tool_call_count",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        points="all",
        title="Tool-call count distribution by system",
        labels={"system_name": "System", "tool_call_count": "Tool calls"},
    )
    fig.update_layout(showlegend=False)
    return fig


def verdict_breakdown(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["verdict"])
    if subset.empty:
        return _empty("No verdict data")
    grouped = (
        subset.groupby(["system_name", "verdict"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig = px.bar(
        grouped,
        x="system_name",
        y="count",
        color="verdict",
        color_discrete_map=_color_map(grouped, "verdict"),
        category_orders=_category_orders(grouped, "verdict"),
        title="PASS / FAIL verdicts per system",
        labels={"system_name": "System", "count": "Run count", "verdict": "Verdict"},
    )
    return fig


def helpfulness_distribution(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["helpfulness_rating"])
    if subset.empty:
        return _empty("No helpfulness_rating data")
    fig = px.box(
        subset,
        x="system_name",
        y="helpfulness_rating",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        points="all",
        title="Examiner helpfulness rating (Likert 1–5) per system",
        labels={"system_name": "System", "helpfulness_rating": "Helpfulness"},
    )
    fig.update_yaxes(range=[0.5, 5.5], dtick=1)
    fig.update_layout(showlegend=False)
    return fig


def analysis_time_by_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["analysis_time_s"])
    if subset.empty:
        return _empty("No analysis_time_s data")
    fig = px.box(
        subset,
        x="system_name",
        y="analysis_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        points="all",
        hover_data=["question_id", "model", "examiner_model", "claims_total"],
        title="Analysis time distribution by system",
        labels={"system_name": "System", "analysis_time_s": "Analysis time (s)"},
    )
    fig.update_layout(showlegend=False)
    return fig


def analysis_time_by_examiner(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["analysis_time_s", "examiner_model"])
    if subset.empty:
        return _empty("No analysis_time_s / examiner_model data")
    fig = px.box(
        subset,
        x="examiner_model",
        y="analysis_time_s",
        color="examiner_model",
        color_discrete_map=_color_map(subset, "examiner_model"),
        category_orders=_category_orders(subset, "examiner_model"),
        points="all",
        hover_data=["question_id", "system_name", "model", "claims_total"],
        title="Analysis time distribution by examiner model",
        labels={"examiner_model": "Examiner model", "analysis_time_s": "Analysis time (s)"},
    )
    fig.update_layout(showlegend=False)
    return fig


def analysis_time_vs_claims(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["analysis_time_s", "claims_total"])
    if subset.empty:
        return _empty("No analysis_time_s / claims_total data")
    fig = px.scatter(
        subset,
        x="claims_total",
        y="analysis_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        symbol="corpus",
        hover_data=["question_id", "model", "examiner_model", "support_rate"],
        title="Analysis time vs total factual claims",
        labels={
            "claims_total": "Total factual claims",
            "analysis_time_s": "Analysis time (s)",
            "system_name": "System",
            "corpus": "Corpus",
        },
    )
    return fig


def analysis_time_vs_answer_chars(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["analysis_time_s", "answer_char_count"])
    if subset.empty:
        return _empty("No analysis_time_s / answer_char_count data")
    fig = px.scatter(
        subset,
        x="answer_char_count",
        y="analysis_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        symbol="corpus",
        hover_data=["question_id", "model", "examiner_model", "claims_total"],
        title="Analysis time vs final answer length",
        labels={
            "answer_char_count": "Final answer length (characters)",
            "analysis_time_s": "Analysis time (s)",
            "system_name": "System",
            "corpus": "Corpus",
        },
    )
    return fig


def execution_time_vs_analysis_time(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["execution_time_s", "analysis_time_s"])
    if subset.empty:
        return _empty("No execution_time_s / analysis_time_s data")
    fig = px.scatter(
        subset,
        x="execution_time_s",
        y="analysis_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        symbol="corpus",
        hover_data=["question_id", "model", "examiner_model", "claims_total"],
        title="Experiment execution time vs analysis time",
        labels={
            "execution_time_s": "Execution time (s)",
            "analysis_time_s": "Analysis time (s)",
            "system_name": "System",
            "corpus": "Corpus",
        },
    )
    return fig


def answer_chars_by_model_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["answer_char_count", "model", "system_name"])
    if subset.empty:
        return _empty("No answer_char_count / model / system_name data")
    fig = px.box(
        subset,
        x="model",
        y="answer_char_count",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "model", "system_name"),
        points="all",
        hover_data=["question_id", "corpus"],
        title="Final answer length distribution by A1 model and system",
        labels={
            "model": "A1 model",
            "answer_char_count": "Final answer length (characters)",
            "system_name": "System",
        },
    )
    return fig


def execution_time_by_model_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["execution_time_s", "model", "system_name"])
    if subset.empty:
        return _empty("No execution_time_s / model / system_name data")
    fig = px.box(
        subset,
        x="model",
        y="execution_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "model", "system_name"),
        points="all",
        hover_data=["question_id", "corpus", "answer_char_count"],
        title="Execution time distribution by A1 model and system",
        labels={
            "model": "A1 model",
            "execution_time_s": "Execution time (s)",
            "system_name": "System",
        },
    )
    return fig


def helpfulness_by_model_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["helpfulness_rating", "model", "system_name"])
    if subset.empty:
        return _empty("No helpfulness_rating / model / system_name data")
    fig = px.box(
        subset,
        x="model",
        y="helpfulness_rating",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "model", "system_name"),
        points="all",
        hover_data=["question_id", "corpus", "examiner_model"],
        title="Examiner helpfulness rating by A1 model and system",
        labels={
            "model": "A1 model",
            "helpfulness_rating": "Helpfulness",
            "system_name": "System",
        },
    )
    fig.update_yaxes(range=[0.5, 5.5], dtick=1)
    return fig


def unsupported_claim_ratio_by_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["unsupported_claim_ratio"])
    if subset.empty:
        return _empty("No unsupported_claim_ratio data")
    grouped = (
        subset.groupby("system_name", as_index=False)["unsupported_claim_ratio"]
        .mean()
        .sort_values("unsupported_claim_ratio", ascending=False)
    )
    fig = px.bar(
        grouped,
        x="system_name",
        y="unsupported_claim_ratio",
        color="system_name",
        color_discrete_map=_color_map(grouped, "system_name"),
        category_orders=_category_orders(grouped, "system_name"),
        title="Mean unsupported claim ratio per system",
        labels={"system_name": "System", "unsupported_claim_ratio": "Unsupported claim ratio"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(showlegend=False)
    return fig


def unsupported_claim_ratio_by_model_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["unsupported_claim_ratio", "model", "system_name"])
    if subset.empty:
        return _empty("No unsupported_claim_ratio / model / system_name data")
    grouped = (
        subset.groupby(["model", "system_name"], as_index=False)["unsupported_claim_ratio"]
        .mean()
        .sort_values("unsupported_claim_ratio", ascending=False)
    )
    fig = px.bar(
        grouped,
        x="model",
        y="unsupported_claim_ratio",
        color="system_name",
        color_discrete_map=_color_map(grouped, "system_name"),
        category_orders=_category_orders(grouped, "model", "system_name"),
        barmode="group",
        title="Mean unsupported claim ratio by A1 model and system",
        labels={
            "model": "A1 model",
            "unsupported_claim_ratio": "Unsupported claim ratio",
            "system_name": "System",
        },
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def verdict_by_model_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["verdict", "model", "system_name"])
    if subset.empty:
        return _empty("No verdict / model / system_name data")
    grouped = (
        subset.groupby(["model", "system_name", "verdict"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig = px.bar(
        grouped,
        x="model",
        y="count",
        color="verdict",
        color_discrete_map=_color_map(grouped, "verdict"),
        category_orders=_category_orders(grouped, "model", "system_name", "verdict"),
        facet_col="system_name",
        title="PASS / FAIL verdicts by A1 model and system",
        labels={"model": "A1 model", "count": "Run count", "verdict": "Verdict", "system_name": "System"},
    )
    return fig


def time_vs_answer_chars_by_system(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["execution_time_s", "answer_char_count", "system_name"])
    if subset.empty:
        return _empty("No execution_time / answer_char_count / system_name data")
    fig = px.scatter(
        subset,
        x="answer_char_count",
        y="execution_time_s",
        color="system_name",
        color_discrete_map=_color_map(subset, "system_name"),
        category_orders=_category_orders(subset, "system_name"),
        hover_data=["question_id", "model", "corpus", "has_answer_error"],
        title="Execution time vs final answer length by system",
        labels={
            "answer_char_count": "Final answer length (characters)",
            "execution_time_s": "Execution time (s)",
            "system_name": "System",
        },
    )
    return fig


# Registry used by exports. Adding a chart should only require adding one
# ChartSpec entry here; filenames and manifest rows are derived from it.
CHARTS = (
    ChartSpec("C01", "support_by_system", "Vidējais atbalsta īpatsvars pa sistēmām", support_rate_by_system),
    ChartSpec(
        "C02",
        "support_by_corpus_system",
        "Atbalsta īpatsvars pa korpusiem un sistēmām",
        support_rate_by_corpus_and_system,
    ),
    ChartSpec("C03", "support_by_level", "Atbalsta īpatsvars pa jautājumu sarežģītības līmeņiem", support_rate_by_level),
    ChartSpec("C04", "time_vs_support", "Izpildes laiks pret atbalsta īpatsvaru", execution_time_vs_support),
    ChartSpec(
        "C05",
        "time_vs_answer_chars",
        "Izpildes laiks pret gala atbildes garumu",
        execution_time_vs_answer_chars,
    ),
    ChartSpec(
        "C06",
        "answer_chars_by_system",
        "Gala atbildes garuma sadalījums pa sistēmām",
        answer_chars_by_system,
    ),
    ChartSpec(
        "C07",
        "execution_time_by_system",
        "Izpildes laika sadalījums pa sistēmām",
        execution_time_by_system,
    ),
    ChartSpec("C08", "tool_calls_vs_time", "Rīku izsaukumi pret izpildes laiku", tool_calls_vs_execution_time),
    ChartSpec(
        "C09",
        "uncited_claims_by_system",
        "Necitēto faktisko apgalvojumu skaits pa sistēmām",
        uncited_claims_by_system,
    ),
    ChartSpec("C10", "claims_vs_support", "Faktisko apgalvojumu skaits pret atbalsta īpatsvaru", claims_total_vs_support),
    ChartSpec("C11", "error_rate_by_system", "Atbilžu kļūdu īpatsvars pa sistēmām", error_rate_by_system),
    ChartSpec("C12", "tool_calls_box", "Rīku izsaukumu skaita sadalījums pa sistēmām", tool_call_distribution),
    ChartSpec("C13", "verdict_breakdown", "PASS un FAIL vērtējumi pa sistēmām", verdict_breakdown),
    ChartSpec("C14", "helpfulness_box", "Eksaminatora noderīguma vērtējums pa sistēmām", helpfulness_distribution),
    ChartSpec(
        "C15",
        "analysis_time_by_system",
        "Analīzes laika sadalījums pa sistēmām",
        analysis_time_by_system,
    ),
    ChartSpec(
        "C16",
        "analysis_time_by_examiner",
        "Analīzes laika sadalījums pa eksaminatora modeļiem",
        analysis_time_by_examiner,
    ),
    ChartSpec("C17", "analysis_time_vs_claims", "Analīzes laiks pret faktisko apgalvojumu skaitu", analysis_time_vs_claims),
    ChartSpec(
        "C18",
        "analysis_time_vs_answer_chars",
        "Analīzes laiks pret gala atbildes garumu",
        analysis_time_vs_answer_chars,
    ),
    ChartSpec(
        "C19",
        "execution_time_vs_analysis_time",
        "Eksperimenta izpildes laiks pret analīzes laiku",
        execution_time_vs_analysis_time,
    ),
    ChartSpec(
        "C20",
        "answer_chars_by_model_system",
        "Gala atbildes garuma sadalījums pa A1 modeli un sistēmu",
        answer_chars_by_model_system,
    ),
    ChartSpec(
        "C21",
        "execution_time_by_model_system",
        "Izpildes laika sadalījums pa A1 modeli un sistēmu",
        execution_time_by_model_system,
    ),
    ChartSpec(
        "C22",
        "helpfulness_by_model_system",
        "A2 noderīguma vērtējuma sadalījums pa A1 modeli un sistēmu",
        helpfulness_by_model_system,
    ),
    ChartSpec(
        "C23",
        "unsupported_claim_ratio_by_system",
        "Nepamatoto apgalvojumu īpatsvars pa sistēmām",
        unsupported_claim_ratio_by_system,
    ),
    ChartSpec(
        "C24",
        "unsupported_claim_ratio_by_model_system",
        "Nepamatoto apgalvojumu īpatsvars pa A1 modeli un sistēmu",
        unsupported_claim_ratio_by_model_system,
    ),
    ChartSpec(
        "C25",
        "verdict_by_model_system",
        "PASS un FAIL vērtējumi pa A1 modeli un sistēmu",
        verdict_by_model_system,
    ),
    ChartSpec(
        "C26",
        "time_vs_answer_chars_by_system",
        "Izpildes laiks pret gala atbildes garumu pa sistēmām",
        time_vs_answer_chars_by_system,
    ),
    ChartSpec(
        "C27",
        "support_by_level_model_system",
        "Atbalsta īpatsvars pa sarežģītības līmeni, A1 modeli un sistēmu",
        support_rate_by_level_model_system,
    ),
    ChartSpec(
        "C28",
        "support_by_model_system_level",
        "Atbalsta īpatsvars pa A1 modeli, sistēmu un sarežģītības līmeni",
        support_rate_by_model_system_level,
    ),
    ChartSpec(
        "C29",
        "execution_time_by_level_model_system",
        "Izpildes laiks pa sarežģītības līmeni, A1 modeli un sistēmu",
        execution_time_by_level_model_system,
    ),
    ChartSpec(
        "C30",
        "answer_chars_by_level_model_system",
        "Gala atbildes garums pa sarežģītības līmeni, A1 modeli un sistēmu",
        answer_chars_by_level_model_system,
    ),
)


# Backward-compatible registry for UI chart selection and existing tests.
ALL_PLOTS = {chart.slug: chart.builder for chart in CHARTS}


def charts_manifest() -> list[dict[str, str]]:
    return [chart.manifest_entry() for chart in CHARTS]


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    return fig
