import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field

from .enums import AutomationLevel, Corpus, SystemName
from .metrics import RunMetrics
from .trace import SessionTrace


class CorpusSnapshot(BaseModel):
    source_corpus_path: str
    prepared_corpus_path: str
    temp_root_path: Optional[str] = None
    temp_root_filesystem: Optional[str] = None
    copied_paths: list[str] = Field(default_factory=list)
    file_count: int = 0
    total_bytes: int = 0
    config_json: Optional[dict[str, Any]] = None
    manifest_entry_count: Optional[int] = None
    pre_run_tree: Optional[str] = None
    post_run_tree: Optional[str] = None
    error: Optional[str] = None


class RunResult(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Identity ---
    system_name: SystemName
    automation_level: AutomationLevel
    corpus: Corpus
    question_id: str
    question_text: str

    # --- Run configuration (nominal scale) ---
    model: str
    quantization: Optional[str] = None
    # Free-form dict for engine-specific knobs (e.g. num_ctx, temperature, gpu_layers).
    inference_config: Optional[dict[str, Any]] = None
    reasoning_enabled: bool = False
    prompt_language: str = "en"
    corpus_language: str = "en"

    # --- Output ---
    answer_text: Optional[str] = None
    # Populated when the run itself fails (timeout, crash, API error).
    answer_error: Optional[str] = None

    # --- Metrics (aggregated + post-processing) ---
    metrics: RunMetrics = Field(default_factory=RunMetrics)

    # --- Corpus preparation metadata (optional; absent in older results) ---
    corpus_snapshot: Optional[CorpusSnapshot] = None

    # --- Full session trace (optional, may be large) ---
    trace: Optional[SessionTrace] = None
