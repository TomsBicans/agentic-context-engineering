from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel

from .enums import AutomationLevel, Corpus, SystemName


class RunConfig(BaseModel):
    system: SystemName
    corpus: Corpus
    model: str
    automation_level: AutomationLevel = AutomationLevel.FULL
    reasoning_enabled: bool = False
    store_trace: bool = True
    # Absolute path to the root corpora directory. Required for local systems.
    path_to_corpora: Optional[Path] = None
    quantization: Optional[str] = None
    # Engine-specific knobs forwarded verbatim to inference_config in RunResult.
    inference_config: Optional[dict[str, Any]] = None
