"""LaTeX table generators for the thesis."""
from __future__ import annotations

import pandas as pd
from tabulate import tabulate


def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    body = tabulate(df, headers="keys", tablefmt="latex_booktabs", showindex=False, floatfmt=".3f")
    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}\n"
        "\\end{table}\n"
    )


def per_system_summary(df: pd.DataFrame) -> str:
    subset = df.dropna(subset=["support_rate"])
    if subset.empty:
        return ""
    grouped = (
        subset.groupby("system_name")
        .agg(
            n_runs=("run_id", "count"),
            support_rate=("support_rate", "mean"),
            error_rate=("error_rate", "mean"),
            mean_time_s=("execution_time_s", "mean"),
            mean_tool_calls=("tool_call_count", "mean"),
            mean_helpfulness=("helpfulness_rating", "mean"),
        )
        .round(3)
        .reset_index()
        .rename(columns={"system_name": "System"})
    )
    return _to_latex(
        grouped,
        caption="Per-system aggregate metrics across all corpora and difficulty levels.",
        label="tab:per-system-summary",
    )


def per_corpus_summary(df: pd.DataFrame) -> str:
    subset = df.dropna(subset=["support_rate"])
    if subset.empty:
        return ""
    grouped = (
        subset.groupby(["system_name", "corpus"])
        .agg(
            n_runs=("run_id", "count"),
            support_rate=("support_rate", "mean"),
            error_rate=("error_rate", "mean"),
            mean_time_s=("execution_time_s", "mean"),
        )
        .round(3)
        .reset_index()
        .rename(columns={"system_name": "System", "corpus": "Corpus"})
    )
    return _to_latex(
        grouped,
        caption="Aggregate metrics per system and corpus.",
        label="tab:per-corpus-summary",
    )


def per_level_summary(df: pd.DataFrame) -> str:
    subset = df.dropna(subset=["support_rate", "level"])
    if subset.empty:
        return ""
    grouped = (
        subset.groupby(["level", "system_name"])
        .agg(
            n_runs=("run_id", "count"),
            support_rate=("support_rate", "mean"),
            mean_helpfulness=("helpfulness_rating", "mean"),
        )
        .round(3)
        .reset_index()
        .rename(columns={"level": "Level", "system_name": "System"})
    )
    return _to_latex(
        grouped,
        caption="Per-difficulty-level metrics.",
        label="tab:per-level-summary",
    )


ALL_TABLES = {
    "per_system_summary": per_system_summary,
    "per_corpus_summary": per_corpus_summary,
    "per_level_summary": per_level_summary,
}
