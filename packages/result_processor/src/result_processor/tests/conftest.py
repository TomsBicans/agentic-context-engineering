from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from result_processor.models.analysis import AnalysisResult, ClaimAnalysis, ClaimStatus, Verdict


def run_payload(
    *,
    run_id: str = "run-1",
    question_id: str = "ss_L2_001",
    answer_text: str = "[Jupiter is a planet.] [file:planets.md, lines:0-1]",
    created_at: str = "2026-04-26T11:55:51Z",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": created_at,
        "system_name": "ace",
        "automation_level": "full",
        "corpus": "solar_system_wiki",
        "question_id": question_id,
        "question_text": "What is Jupiter?",
        "model": "qwen3:4b",
        "reasoning_enabled": False,
        "answer_text": answer_text,
        "metrics": {
            "execution_time_s": 12.5,
            "tool_call_count": 2,
            "tokens": {"input": 10, "output": 15},
            "corpus_used": True,
        },
    }


def analysis_result(
    *,
    run_id: str = "run-1",
    status: ClaimStatus = ClaimStatus.SUPPORTED,
) -> AnalysisResult:
    return AnalysisResult(
        run_id=run_id,
        examiner_model="qwen3:4b",
        analyzed_at=datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc),
        claims=[
            ClaimAnalysis(
                statement="Jupiter is a planet.",
                cited_file="planets.md",
                cited_line_start=0,
                cited_line_end=1,
                excerpt="Jupiter is a planet.",
                status=status,
                justification="Supported by the excerpt.",
            )
        ],
        claims_total=1,
        claims_supported=1 if status == ClaimStatus.SUPPORTED else 0,
        claims_bad_reference=1 if status == ClaimStatus.BAD_REFERENCE else 0,
        support_rate=1.0 if status == ClaimStatus.SUPPORTED else 0.0,
        error_rate=0.0 if status == ClaimStatus.SUPPORTED else 1.0,
        verdict=Verdict.PASS if status == ClaimStatus.SUPPORTED else Verdict.FAIL,
        helpfulness_rating=4,
    )


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump_json"):
                handle.write(row.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(row) + "\n")
