import argparse
import time
from pathlib import Path
from typing import List
from langchain_core.tools import tool, BaseTool


def create_performer_tools(start_time_stamp: int, time_limit_s: int, path_to_corpora: Path) -> List[BaseTool]:
    @tool
    def list_paths(relative_path: str) -> List[str]:
        """List paths under corpora matching a glob pattern (e.g. '**/*.txt')."""
        return [str(path) for path in path_to_corpora.rglob(relative_path)]

    @tool
    def read_file(relative_path: str) -> str:
        """Read a text file (relative to corpora root) and return its contents."""
        return path_to_corpora.joinpath(relative_path).read_text()

    @tool
    def search(relative_path: str, pattern: str) -> str:
        """Search for a pattern in matching files and return hits with file + line ranges."""
        # Returns response in format '[statement] [file: path, lines:a-b]'
        # statement - text
        # file: path - relative path to file
        # lines:a-b - lines in file

        # TODO: implement
        return ""

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
        return time_limit_s - int(time.time()) - start_time_stamp

    return [list_paths, read_file, search, file_meta, time_elapsed, time_left]


def create_validator_tools(path_to_corpora: Path) -> List[BaseTool]:
    @tool
    def resolve_reference(relative_path: str, a: int, b: int):
        """Return lines [a:b] (0-based, end-exclusive) from a text file under corpora root."""
        return path_to_corpora.joinpath(relative_path).read_text().splitlines()[a:b]

    return [resolve_reference]


