"""Resolve cited file paths and line ranges into raw text excerpts.

This is the deterministic counterpart of the A2 agent's `resolve_reference`
tool — invoked directly without going through the LLM.
"""
from __future__ import annotations

from pathlib import Path

from experiment_runner.models.enums import Corpus


# Maps a Corpus enum to the on-disk subdirectory under the corpora root.
# (Corpus.SCIPY -> "scipy_repo" because the scraped data is named after the
# repo, not the package.)
CORPUS_DIR_NAMES: dict[Corpus, str] = {
    Corpus.OBLIVION_WIKI: "oblivion_wiki",
    Corpus.SOLAR_SYSTEM_WIKI: "solar_system_wiki",
    Corpus.SCIPY: "scipy_repo",
}


_MAX_EXCERPT_CHARS = 12_000


class ExcerptResolver:
    """Reads corpus excerpts for a fixed corpora root."""

    def __init__(self, corpora_root: Path) -> None:
        self.corpora_root = corpora_root

    def corpus_dir(self, corpus: Corpus) -> Path:
        return self.corpora_root / CORPUS_DIR_NAMES[corpus]

    def resolve(self, corpus: Corpus, relative_path: str, a: int, b: int) -> str | None:
        """Return excerpt text or None if the reference is unresolvable.

        None signals the caller that the citation deserves a BAD_REFERENCE
        status without any LLM call.
        """
        if a < 0 or b <= a:
            return None

        requested_path = Path(relative_path)
        if requested_path.is_absolute():
            return None

        corpus_root = self.corpus_dir(corpus).resolve()
        path = (corpus_root / requested_path).resolve()
        if not path.is_relative_to(corpus_root):
            return None

        if not path.exists() or not path.is_file():
            return None

        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None

        lines = text.splitlines()
        n = len(lines)
        if n == 0:
            return None

        a_clamped = max(0, min(a, n))
        b_clamped = max(0, min(b, n))
        if b_clamped <= a_clamped:
            return None

        chunk = "\n".join(lines[a_clamped:b_clamped])
        if len(chunk) > _MAX_EXCERPT_CHARS:
            chunk = chunk[:_MAX_EXCERPT_CHARS] + "\n...[truncated]"
        return chunk
