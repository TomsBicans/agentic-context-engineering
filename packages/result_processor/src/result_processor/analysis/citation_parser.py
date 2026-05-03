"""Parse citations from agent answer text.

The expected format (per thesis system prompt) is:
    [statement] [file:<relative_path>, lines:<a>-<b>]

with line ranges treated as 0-based, half-open [a, b).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


CITATION_RE = re.compile(
    r"\[(?P<statement>[^\[\]]+?)\]"
    r"\s*"
    r"\[\s*file\s*:\s*(?P<file>[^,\]]+?)"
    r"\s*,\s*lines\s*:\s*"
    r"(?P<a>\d+)\s*-\s*(?P<b>\d+)\s*\]",
    re.IGNORECASE | re.DOTALL,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove `<think>...</think>` blocks (and any leading content before the
    final `</think>`) so downstream parsing only sees the model's final answer.

    Robust to the common pattern where reasoning models produce their
    chain-of-thought before the user-facing answer.
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # If `<think>` was not closed, fall back to taking content after the last
    # `</think>` if any, otherwise leave the text as-is.
    last_close = cleaned.rfind("</think>")
    if last_close != -1:
        cleaned = cleaned[last_close + len("</think>"):]
    return cleaned.strip()


@dataclass(frozen=True)
class ParsedCitation:
    statement: str
    file_path: str
    line_start: int
    line_end: int


def extract_citations(answer_text: str) -> list[ParsedCitation]:
    """Return all `[statement] [file:..., lines:a-b]` citations found in the text.

    Reasoning blocks (`<think>...</think>`) are stripped before parsing so that
    citations mentioned inside the model's chain-of-thought are not double-counted.
    Whitespace inside the statement is normalised (newlines collapsed).
    """
    if not answer_text:
        return []

    cleaned = strip_reasoning(answer_text)
    citations: list[ParsedCitation] = []
    for match in CITATION_RE.finditer(cleaned):
        statement = " ".join(match.group("statement").split())
        file_path = match.group("file").strip()
        a = int(match.group("a"))
        b = int(match.group("b"))
        if not statement or not file_path or b <= a:
            continue
        citations.append(
            ParsedCitation(
                statement=statement,
                file_path=file_path,
                line_start=a,
                line_end=b,
            )
        )
    return citations


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter for fallback claim extraction.

    Used when an answer contains no parseable citations — we still want to
    count each declarative sentence as an (unsupported) claim so baseline
    systems get a meaningful score.
    """
    if not text:
        return []
    cleaned = strip_reasoning(text)
    cleaned = CITATION_RE.sub("", cleaned).strip()
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [p.strip() for p in parts if p.strip()]
