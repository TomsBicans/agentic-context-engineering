import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field

from .enums import AutomationLevel, Corpus, SystemName
from .metrics import RunMetrics
from .trace import SessionTrace


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

    # --- Full session trace (optional, may be large) ---
    trace: Optional[SessionTrace] = None
