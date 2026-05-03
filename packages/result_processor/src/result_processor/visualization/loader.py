"""Load and join experiment + analysis JSONL files into a single DataFrame."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_runner.models.result import RunResult

from result_processor.models.analysis import AnalysisResult


_LEVEL_RE = re.compile(r"_L(\d+)_")


def _parse_level(question_id: str) -> int | None:
    match = _LEVEL_RE.search(question_id or "")
    return int(match.group(1)) if match else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_runs(experiment_results_dir: Path) -> list[RunResult]:
    runs: list[RunResult] = []
    for jsonl in sorted(experiment_results_dir.glob("*.jsonl")):
        for row in _read_jsonl(jsonl):
            runs.append(RunResult.model_validate(row))
    return runs


def load_analyses(analysis_results_dir: Path) -> dict[str, AnalysisResult]:
    """Return a dict keyed by run_id. Last write wins on duplicates."""
    by_run_id: dict[str, AnalysisResult] = {}
    if not analysis_results_dir.exists():
        return by_run_id
    for jsonl in sorted(analysis_results_dir.glob("*.jsonl")):
        for row in _read_jsonl(jsonl):
            analysis = AnalysisResult.model_validate(row)
            by_run_id[analysis.run_id] = analysis
    return by_run_id


def build_dataframe(
    experiment_results_dir: Path,
    analysis_results_dir: Path,
) -> pd.DataFrame:
    """Join RunResult + AnalysisResult into one row per run.

    Rows from runs without a matching analysis are still included (analysis
    columns are NaN) so visualisations can show coverage gaps.
    """
    runs = load_runs(experiment_results_dir)
    analyses = load_analyses(analysis_results_dir)

    records: list[dict[str, Any]] = []
    for run in runs:
        analysis = analyses.get(run.run_id)
        tokens_total = (
            run.metrics.tokens.total
            if run.metrics and run.metrics.tokens
            else None
        )
        record: dict[str, Any] = {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "run_date": run.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "system_name": run.system_name.value,
            "corpus": run.corpus.value,
            "model": run.model,
            "reasoning_enabled": run.reasoning_enabled,
            "question_id": run.question_id,
            "question_text": run.question_text,
            "level": _parse_level(run.question_id),
            "automation_level": run.automation_level.value,
            "answer_text": run.answer_text,
            "execution_time_s": run.metrics.execution_time_s if run.metrics else None,
            "step_count": run.metrics.step_count if run.metrics else None,
            "tool_call_count": run.metrics.tool_call_count if run.metrics else None,
            "tokens_total": tokens_total,
            "corpus_used": run.metrics.corpus_used if run.metrics else None,
            # analysis fields — None when no analysis yet
            "support_rate": analysis.support_rate if analysis else None,
            "error_rate": analysis.error_rate if analysis else None,
            "overclaim_rate": analysis.overclaim_rate if analysis else None,
            "unsupported_claim_ratio": analysis.unsupported_claim_ratio if analysis else None,
            "claims_total": analysis.claims_total if analysis else None,
            "claims_supported": analysis.claims_supported if analysis else None,
            "claims_without_citation_count": analysis.claims_without_citation_count if analysis else None,
            "verdict": analysis.verdict.value if analysis else None,
            "helpfulness_rating": analysis.helpfulness_rating if analysis else None,
            "examiner_model": analysis.examiner_model if analysis else None,
        }
        records.append(record)

    return pd.DataFrame.from_records(records)
