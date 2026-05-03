"""JSONL I/O for RunResult inputs and AnalysisResult outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from experiment_runner.models.result import RunResult

from result_processor.models.analysis import AnalysisResult


def iter_run_results(jsonl_path: Path) -> Iterator[RunResult]:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no}: invalid JSON ({exc})") from exc
            yield RunResult.model_validate(payload)


def load_existing_run_ids(analysis_path: Path) -> set[str]:
    """Return run_ids already present in an analysis output file (idempotency key)."""
    if not analysis_path.exists():
        return set()
    seen: set[str] = set()
    with analysis_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = payload.get("run_id")
            if isinstance(run_id, str):
                seen.add(run_id)
    return seen


def append_analysis(analysis_path: Path, analysis: AnalysisResult) -> None:
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with analysis_path.open("a", encoding="utf-8") as handle:
        handle.write(analysis.model_dump_json() + "\n")
        handle.flush()
