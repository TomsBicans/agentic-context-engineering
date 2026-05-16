from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AnalysisTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    ANALYZED = "analyzed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisJobTask(BaseModel):
    run_id: str
    source_file: str
    output_file: str
    status: AnalysisTaskStatus = AnalysisTaskStatus.PENDING
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class AnalysisJobState(BaseModel):
    job_name: str
    experiment_results_dir: str
    output_dir: str
    path_to_corpora: str
    examiner_model: str
    suite_id: Optional[str] = None
    suite_name: Optional[str] = None
    suite_config_path: Optional[str] = None
    suite_state_path: Optional[str] = None
    augmented_from_state_path: Optional[str] = None
    num_ctx: int = 8192
    input_files: list[str] = Field(default_factory=list)
    resume: bool = True
    cancel_requested: bool = False
    active_pid: Optional[int] = None
    log_path: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tasks: list[AnalysisJobTask] = Field(default_factory=list)


def analysis_job_state_path(analysis_root: str | Path, job_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in job_name.lower()).strip("-")
    return Path(analysis_root) / f"{safe or 'analysis-job'}.state.json"
