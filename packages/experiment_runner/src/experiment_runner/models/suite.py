from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from experiment_runner.models.enums import Corpus, SystemName


class SuiteTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SuiteCorpusSelection(BaseModel):
    corpus: Corpus
    questions_file: str
    path_to_corpora: str
    question_ids: list[str] = Field(default_factory=list)
    levels: list[int] = Field(default_factory=list)


class ExperimentSuiteConfig(BaseModel):
    suite_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    systems: list[SystemName]
    models: list[str]
    corpora: list[SuiteCorpusSelection]
    output_dir: str = "./data/experiment_results"
    num_ctx: int = 8192
    reasoning_enabled: bool = False
    no_trace: bool = False


class SuiteTask(BaseModel):
    task_id: str
    index: int
    system: SystemName
    model: str
    corpus: Corpus
    questions_file: str
    path_to_corpora: str
    question_id: str
    question_text: str
    level: int
    command: list[str]
    status: SuiteTaskStatus = SuiteTaskStatus.PENDING
    result_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    return_code: Optional[int] = None


class ExperimentSuiteState(BaseModel):
    suite_id: str
    suite_name: str
    config_path: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cancel_requested: bool = False
    active_pid: Optional[int] = None
    log_path: Optional[str] = None
    tasks: list[SuiteTask] = Field(default_factory=list)


def suite_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return slug or "experiment-suite"


def default_state_path(config_path: str | Path, config: ExperimentSuiteConfig) -> Path:
    path = Path(config_path)
    return path.with_name(f"{path.stem}.state.json")
