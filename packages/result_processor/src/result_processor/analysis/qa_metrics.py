"""Standard QA answer metrics used during post-processing analysis."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import string


@dataclass(frozen=True)
class QAMetrics:
    exact_match: float | None
    f1: float | None
    precision: float | None
    recall: float | None


_PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    return text.lower().translate(_PUNCTUATION_TRANSLATION).strip()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def _normalized_token_text(text: str) -> str:
    return " ".join(_tokens(text))


def compute_qa_metrics(answer_text: str | None, expected_facts: list[str] | None) -> QAMetrics:
    """Compute EM/F1/precision/recall against joined expected facts.

    The gold answer is the whitespace-joined ``expected_facts`` list from the
    question definition. Text is lowercased, punctuation is removed, surrounding
    whitespace is stripped, and token metrics use simple whitespace tokens.
    """
    gold_text = " ".join(expected_facts or [])
    gold_tokens = _tokens(gold_text)
    if not gold_tokens:
        return QAMetrics(
            exact_match=None,
            f1=None,
            precision=None,
            recall=None,
        )

    prediction_text = answer_text or ""
    prediction_tokens = _tokens(prediction_text)
    exact_match = 1.0 if _normalized_token_text(prediction_text) == _normalized_token_text(gold_text) else 0.0

    if not prediction_tokens:
        return QAMetrics(
            exact_match=exact_match,
            f1=0.0,
            precision=0.0,
            recall=0.0,
        )

    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return QAMetrics(
        exact_match=exact_match,
        f1=f1,
        precision=precision,
        recall=recall,
    )
