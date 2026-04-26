"""A2 examiner — LLM-based per-claim verification.

The thesis describes A2 as a separate agent that uses `resolve_reference` to
verify each cited claim. Here we split the responsibility:

  * Citation parsing and excerpt resolution are deterministic (no LLM).
  * The LLM is asked one focused question per claim: "given this excerpt,
    classify the claim", returning a strict JSON object that we validate.

This keeps the LLM's role narrow (classification) where it is reliable, and
moves anything mechanical (regex, file IO) out of the agent loop.
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

from result_processor.models.analysis import ClaimStatus


_CLASSIFY_PROMPT = """\
You are a strict verifier. Decide whether the CLAIM is supported by the EXCERPT.

CLAIM:
{claim}

EXCERPT (file: {file_path}, lines: {a}-{b}):
{excerpt}

Classification rules:
- supported: the excerpt clearly entails the full claim.
- partially_supported: the excerpt supports part, but the claim adds extra detail or stronger wording.
- not_supported: the excerpt does not support the claim, contradicts it, or is unrelated.
- bad_reference: the excerpt is empty, unreadable, or cannot be matched to the claim at all.

Respond with a single valid JSON object on one line, no prose, no markdown:
{{"status": "supported|partially_supported|not_supported|bad_reference", "justification": "1-2 sentences"}}
"""

_SUMMARIZE_PROMPT = """\
You are scoring an answer that was produced for a question, given the per-claim verdicts of its citations.

QUESTION:
{question}

ANSWER:
{answer}

PER-CLAIM VERDICTS:
{claim_summary}

Provide a Likert 1-5 helpfulness rating for the answer (5 = very helpful and well-grounded, 1 = unhelpful or unsupported)
and a one-paragraph note about the answer's overall quality.

Respond with a single valid JSON object on one line, no prose, no markdown:
{{"helpfulness_rating": <1-5>, "notes": "<one short paragraph>"}}
"""


class _ClaimVerdict(BaseModel):
    status: ClaimStatus
    justification: str = ""


class _OverallVerdict(BaseModel):
    helpfulness_rating: int = Field(ge=1, le=5)
    notes: str = ""


class ExaminerLLM:
    """Stateless wrapper around ChatOllama with JSON-mode validation."""

    def __init__(self, model: str, num_ctx: int = 8192, temperature: float = 0.0) -> None:
        self.model = model
        self._client = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=temperature,
            num_ctx=num_ctx,
            format="json",
        )

    def classify_claim(
        self,
        claim: str,
        excerpt: Optional[str],
        file_path: str,
        a: int,
        b: int,
    ) -> _ClaimVerdict:
        if excerpt is None:
            return _ClaimVerdict(
                status=ClaimStatus.BAD_REFERENCE,
                justification="Excerpt could not be resolved (missing file or empty range).",
            )

        prompt = _CLASSIFY_PROMPT.format(
            claim=claim,
            file_path=file_path,
            a=a,
            b=b,
            excerpt=excerpt,
        )
        return self._invoke_json(prompt, _ClaimVerdict, fallback=_ClaimVerdict(
            status=ClaimStatus.BAD_REFERENCE,
            justification="Examiner failed to produce a parseable verdict.",
        ))

    def summarize(
        self,
        question: str,
        answer: str,
        claim_summary: str,
    ) -> _OverallVerdict:
        prompt = _SUMMARIZE_PROMPT.format(
            question=question,
            answer=answer or "(empty answer)",
            claim_summary=claim_summary or "(no claims)",
        )
        return self._invoke_json(prompt, _OverallVerdict, fallback=_OverallVerdict(
            helpfulness_rating=1,
            notes="Examiner failed to produce a parseable summary.",
        ))

    def _invoke_json(self, prompt: str, schema: type[BaseModel], *, fallback: BaseModel) -> BaseModel:
        try:
            response = self._client.invoke(prompt)
        except Exception:
            return fallback

        content = getattr(response, "content", "") or ""
        if not content:
            return fallback

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return fallback

        try:
            return schema.model_validate(data)
        except ValidationError:
            return fallback
