from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    """Per-claim verdict produced by the A2 examiner agent.

    Mirrors the rubric defined in the thesis EXAMINER_SYSTEM_MESSAGE
    (packages/agent/src/agent/core.py).
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    BAD_REFERENCE = "bad_reference"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class ClaimAnalysis(BaseModel):
    """One claim from the answer, verified against the cited corpus excerpt."""

    statement: str
    cited_file: Optional[str] = None
    cited_line_start: Optional[int] = None
    cited_line_end: Optional[int] = None
    excerpt: Optional[str] = None
    status: ClaimStatus
    justification: str = ""


class AnalysisResult(BaseModel):
    """A2 examiner output for a single RunResult.

    Stored separately from RunResult — `run_id` is the foreign key. This keeps
    raw experimental data immutable: re-running analysis with a different
    examiner model never touches experiment_results/.
    """

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # The model used by the A2 examiner (may differ from the A1 model under test).
    examiner_model: str

    # Per-claim verifications.
    claims: list[ClaimAnalysis] = Field(default_factory=list)

    # Counts derived from `claims` for quick filtering / aggregation.
    claims_total: int = 0
    claims_supported: int = 0
    claims_partially_supported: int = 0
    claims_not_supported: int = 0
    claims_bad_reference: int = 0

    # Claims found in the answer text without any citation at all
    # (relevant for baseline systems that do not produce structured citations).
    claims_without_citation_count: int = 0

    # Aggregate rates in [0, 1].
    support_rate: float = 0.0
    error_rate: float = 0.0
    overclaim_rate: float = 0.0
    unsupported_claim_ratio: float = 0.0

    verdict: Verdict = Verdict.FAIL

    # Optional helpfulness rating on a Likert 1-5 scale.
    helpfulness_rating: Optional[int] = Field(default=None, ge=1, le=5)

    # Free-form notes from the examiner (overall summary, caveats, etc.).
    examiner_notes: str = ""
