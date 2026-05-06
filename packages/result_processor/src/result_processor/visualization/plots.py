"""Plotly figure generators.

Each function takes a long-format DataFrame (one row per run) and returns a
plotly Figure. The orchestrator decides where to save it.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
        title="Mean support rate per system",
        labels={"system_name": "System", "support_rate": "Support rate"},
    )
    fig.update_yaxes(range=[0, 1])
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
        markers=True,
        title="Support rate by question difficulty level",
        labels={"level": "Difficulty level", "support_rate": "Support rate", "system_name": "System"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(dtick=1)
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
        title="Answer-error rate by system",
        labels={"system_name": "System", "answer_error_rate": "Answer-error rate"},
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def tool_call_distribution(df: pd.DataFrame) -> go.Figure:
    subset = df.dropna(subset=["tool_call_count"])
    if subset.empty:
        return _empty("No tool_call_count data")
    fig = px.box(
        subset,
        x="system_name",
        y="tool_call_count",
        points="all",
        title="Tool-call count distribution by system",
        labels={"system_name": "System", "tool_call_count": "Tool calls"},
    )
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
        points="all",
        title="Examiner helpfulness rating (Likert 1–5) per system",
        labels={"system_name": "System", "helpfulness_rating": "Helpfulness"},
    )
    fig.update_yaxes(range=[0.5, 5.5], dtick=1)
    return fig


# Registry — name → builder function. Used by the pipeline.
ALL_PLOTS = {
    "support_by_system": support_rate_by_system,
    "support_by_corpus_system": support_rate_by_corpus_and_system,
    "support_by_level": support_rate_by_level,
    "time_vs_support": execution_time_vs_support,
    "time_vs_answer_chars": execution_time_vs_answer_chars,
    "answer_chars_by_system": answer_chars_by_system,
    "execution_time_by_system": execution_time_by_system,
    "tool_calls_vs_time": tool_calls_vs_execution_time,
    "uncited_claims_by_system": uncited_claims_by_system,
    "claims_vs_support": claims_total_vs_support,
    "error_rate_by_system": error_rate_by_system,
    "tool_calls_box": tool_call_distribution,
    "verdict_breakdown": verdict_breakdown,
    "helpfulness_box": helpfulness_distribution,
}


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    return fig
