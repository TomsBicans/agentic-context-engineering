import re
import time
from pathlib import Path
from typing import List
from langchain_core.tools import tool, BaseTool
from dataclasses import dataclass


@dataclass(frozen=True)
class PerformerTools:
    list_paths: BaseTool
    read_lines: BaseTool
    search: BaseTool
    file_meta: BaseTool
    time_elapsed: BaseTool
    time_left: BaseTool

    def as_list(self) -> List[BaseTool]:
        return [
            self.list_paths,
            self.read_lines,
            self.search,
            self.file_meta,
            self.time_elapsed,
            self.time_left,
        ]


@dataclass(frozen=True)
class ValidatorTools:
    resolve_reference: BaseTool

    def as_list(self) -> List[BaseTool]:
        return [
            self.resolve_reference,
        ]


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def create_performer_tools(start_time_stamp: int, time_limit_s: int, path_to_corpora: Path) -> PerformerTools:
    @tool
    def list_paths(pattern: str) -> List[str]:
        """List paths under corpora matching a glob pattern (e.g. '**/*.txt', '*.html')."""
        return [
            path.relative_to(path_to_corpora).as_posix()
            for path in path_to_corpora.rglob(pattern)
        ]

    @tool
    def read_lines(relative_path: str, a: int, b: int) -> str:
        """Read lines [a:b] (0-based, end-exclusive) from a text file under corpora root.
        Output is bounded to avoid blowing up the LLM context."""
        path = path_to_corpora.joinpath(relative_path)
        if not path.exists():
            return f"Error: File {relative_path} does not exist in the corpora."

        # basic validation
        if a < 0 or b < 0:
            return "Error: line indices must be non-negative."
        if b <= a:
            return "Error: invalid range (b must be > a)."

        MAX_LINES_PER_READ = 80
        MAX_CHARS_PER_READ = 12_000

        # Enforce window size: b <= a + MAX_LINES_PER_READ
        b = min(b, a + MAX_LINES_PER_READ)

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")

        lines = text.splitlines()
        n = len(lines)
        if n == 0:
            return f"[file: {relative_path}] (empty file)"

        # Clamp to file bounds (b can be == n)
        a = clamp(a, 0, n)
        b = clamp(b, 0, n)

        if b <= a:
            return f"Error: requested range is empty after clamping (file has {n} lines)."

        chunk = "\n".join(lines[a:b])

        # clamp output chars (protect against gigantic text lines)
        if len(chunk) > MAX_CHARS_PER_READ:
            chunk = chunk[:MAX_CHARS_PER_READ] + "\n...[truncated]"

        # include a small header so the model 'knows' what it’s seeing
        return f"[file: {relative_path}, lines:{a}-{b}]\n{chunk}"

    @tool
    def search(relative_path: str, pattern: str) -> str:
        """Search for a pattern in matching files and return hits with file + line ranges."""
        # Returns response in format '[statement] [file: path, lines:a-b]'
        # statement - text
        # file: path - relative path to file
        # lines:a-b - lines in file

        # TODO: Add an actual researched optimal implementation later
        if not pattern:
            return "Error: search pattern is empty."

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex pattern: {exc}"

        matches = []
        for path in path_to_corpora.rglob(relative_path):
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue

            rel_path = path.relative_to(path_to_corpora).as_posix()
            for idx, line in enumerate(text.splitlines()):
                if regex.search(line):
                    statement = line.strip()
                    matches.append(f"{statement} [file: {rel_path}, lines:{idx}-{idx + 1}]")

        if not matches:
            return "No matches found."
        return "\n".join(matches)

    @tool
    def file_meta(relative_path: str) -> str:
        """Return file size in MB for a file under corpora root."""
        # du -b path
        B = path_to_corpora.joinpath(relative_path).stat().st_size
        MB = B / 1024 / 1024
        return f"{str(MB)} MB"

    @tool
    def time_elapsed() -> int:
        """Return seconds elapsed since agent start."""
        return int(time.time()) - start_time_stamp

    @tool
    def time_left() -> int:
        """Return seconds remaining until time limit is reached."""
        return time_limit_s - (int(time.time()) - start_time_stamp)

    return PerformerTools(
        list_paths=list_paths,
        read_lines=read_lines,
        search=search,
        file_meta=file_meta,
        time_elapsed=time_elapsed,
        time_left=time_left,
    )


def create_validator_tools(path_to_corpora: Path) -> ValidatorTools:
    @tool
    def resolve_reference(relative_path: str, a: int, b: int):
        """Return lines [a:b] (0-based, end-exclusive) from a text file under corpora root."""
        return path_to_corpora.joinpath(relative_path).read_text().splitlines()[a:b]

    return ValidatorTools(
        resolve_reference=resolve_reference
    )
