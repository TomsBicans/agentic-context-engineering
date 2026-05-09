from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from experiment_runner.models.result import CorpusSnapshot

_METADATA_FILES = ("config.json", "manifest.jsonl")
_TMP_DIR_ENV = "EXPERIMENT_RUNNER_CORPUS_TMP_DIR"
_AUTO_TEMP_DIR_NAME = "ace_corpora"
_AUTO_TEMP_ROOTS = (Path("/dev/shm"), Path("/tmp"))
_MIN_FREE_BYTES = 100 * 1024 * 1024
_MIN_FREE_MULTIPLIER = 2


def capture_tree(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["tree", "-a", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return _fallback_tree(path)
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout
    return _fallback_tree(path)


@contextmanager
def isolated_corpus(source_corpus_path: Path) -> Iterator[tuple[Path, CorpusSnapshot]]:
    source = source_corpus_path.resolve()
    required_bytes = _estimate_required_bytes(source)
    temp_root = _select_temp_root(required_bytes)
    with TemporaryDirectory(prefix="experiment_corpus_", dir=str(temp_root) if temp_root else None) as tmp:
        prepared = Path(tmp) / source.name
        snapshot = _prepare_corpus(source, prepared)
        snapshot.temp_root_path = str(temp_root) if temp_root else None
        snapshot.temp_root_filesystem = _filesystem_type(Path(tmp))
        snapshot.pre_run_tree = capture_tree(prepared)
        try:
            yield prepared, snapshot
        finally:
            try:
                snapshot.post_run_tree = capture_tree(prepared)
            except Exception as exc:
                snapshot.error = _append_error(snapshot.error, f"post-run snapshot failed: {exc}")


def _prepare_corpus(source: Path, prepared: Path) -> CorpusSnapshot:
    snapshot = CorpusSnapshot(
        source_corpus_path=str(source),
        prepared_corpus_path=str(prepared),
    )
    prepared.mkdir(parents=True, exist_ok=True)

    text_source = source / "text"
    text_target = prepared / "text"
    try:
        if not text_source.is_dir():
            snapshot.error = f"source corpus text directory not found: {text_source}"
            return snapshot

        shutil.copytree(text_source, text_target)
        snapshot.copied_paths.append("text")

        for filename in _METADATA_FILES:
            source_file = source / filename
            if source_file.is_file():
                shutil.copy2(source_file, prepared / filename)
                snapshot.copied_paths.append(filename)

        snapshot.config_json = _load_config_json(prepared / "config.json")
        snapshot.manifest_entry_count = _count_manifest_entries(prepared / "manifest.jsonl")
        snapshot.file_count, snapshot.total_bytes = _count_files(prepared)
    except Exception as exc:
        snapshot.error = _append_error(snapshot.error, f"corpus preparation failed: {exc}")
    return snapshot


def _select_temp_root(required_bytes: int) -> Path | None:
    override = os.environ.get(_TMP_DIR_ENV)
    if override:
        path = Path(override).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    for root in _AUTO_TEMP_ROOTS:
        candidate = root / _AUTO_TEMP_DIR_NAME
        if _is_usable_temp_root(candidate, required_bytes):
            return candidate

    return Path(tempfile.gettempdir())


def _is_usable_temp_root(path: Path, required_bytes: int) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            return False
        test_dir = path / f".experiment_runner_write_test_{os.getpid()}"
        test_dir.mkdir()
        test_dir.rmdir()
        free_bytes = shutil.disk_usage(path).free
    except OSError:
        return False
    required_free = max(_MIN_FREE_BYTES, required_bytes * _MIN_FREE_MULTIPLIER)
    return free_bytes >= required_free


def _estimate_required_bytes(source: Path) -> int:
    total = 0
    for item in [source / "text", *(source / filename for filename in _METADATA_FILES)]:
        if not item.exists():
            continue
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
            continue
        for child in item.rglob("*"):
            if not child.is_file():
                continue
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _load_config_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _count_manifest_entries(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _count_files(path: Path) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        count += 1
        try:
            total_bytes += item.stat().st_size
        except OSError:
            pass
    return count, total_bytes


def _fallback_tree(path: Path) -> str:
    lines = [path.name]
    for item in sorted(path.rglob("*")):
        try:
            relative = item.relative_to(path)
        except ValueError:
            continue
        indent = "  " * (len(relative.parts) - 1)
        suffix = "/" if item.is_dir() else ""
        lines.append(f"{indent}{relative.name}{suffix}")
    return "\n".join(lines) + "\n"


def _filesystem_type(path: Path) -> str | None:
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
        resolved = path.resolve()
    except OSError:
        return None

    best_mount = Path("/")
    best_fs: str | None = None
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = Path(parts[1].replace("\\040", " "))
        try:
            if resolved == mount_point or resolved.is_relative_to(mount_point):
                if len(mount_point.parts) >= len(best_mount.parts):
                    best_mount = mount_point
                    best_fs = parts[2]
        except OSError:
            continue
    return best_fs


def _append_error(existing: str | None, message: str) -> str:
    return f"{existing}; {message}" if existing else message
