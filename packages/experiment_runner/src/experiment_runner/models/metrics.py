from typing import Optional
from pydantic import BaseModel, Field

from .enums import CitationQuality


class TokenCounts(BaseModel):
    input: Optional[int] = None
    output: Optional[int] = None
    cache_read: Optional[int] = None
    cache_creation: Optional[int] = None

    @property
    def total(self) -> Optional[int]:
        if self.input is None and self.output is None:
            return None
        return (self.input or 0) + (self.output or 0)


class RunMetrics(BaseModel):
    # --- Collected during the run ---

    # Ratio scale
    execution_time_s: Optional[float] = None
    step_count: Optional[int] = None
    tool_call_count: Optional[int] = None
    tokens: Optional[TokenCounts] = None

    # Nominal scale
    corpus_used: Optional[bool] = None
    tool_call_sequence: list[str] = Field(default_factory=list)

    # --- Collected post-run by the examiner agent ---

    # Ordinal scale (Likert 1-5)
    helpfulness_rating: Optional[int] = Field(default=None, ge=1, le=5)
    citation_quality: Optional[CitationQuality] = None

    # --- Populated during post-processing analysis ---

    # Ratio scale
    exact_match: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    f1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    precision: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    unsupported_claim_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
