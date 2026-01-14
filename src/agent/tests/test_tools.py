import time
from pathlib import Path
from tempfile import TemporaryDirectory

from src.agent.tools import create_performer_tools, create_validator_tools


def test_performer_tools_read_lines():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        text = "line0\nline1\nline2\nline3\n"
        root.joinpath("file1.txt").write_text(text)
        root.joinpath("file2.txt").write_text(text)

        tools = create_performer_tools(
            start_time_stamp=0,
            time_limit_s=60,
            path_to_corpora=root,
        )

        out = tools.read_lines.invoke({"relative_path": "file1.txt", "a": 1, "b": 3})
        assert out.startswith("[file: file1.txt, lines:1-3]")
        assert "line1" in out
        assert "line2" in out

        out = tools.read_lines.invoke({"relative_path": "file3.txt", "a": 0, "b": 1})
        assert "error" in str(out).lower()


def test_performer_tools_list_paths_relative():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        root.joinpath("notes.txt").write_text("a")
        root.joinpath("subdir").mkdir()
        root.joinpath("subdir", "more.txt").write_text("b")
        root.joinpath("subdir", "image.png").write_text("c")

        tools = create_performer_tools(
            start_time_stamp=0,
            time_limit_s=60,
            path_to_corpora=root,
        )

        out = tools.list_paths.invoke({"relative_path": "**/*.txt"})
        assert set(out) == {"notes.txt", "subdir/more.txt"}
        assert all(not Path(path).is_absolute() for path in out)


def test_performer_tools_search():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        root.joinpath("planets.txt").write_text(
            "Earth is here\nMars is there\nEarth again\n"
        )

        tools = create_performer_tools(
            start_time_stamp=0,
            time_limit_s=60,
            path_to_corpora=root,
        )

        out = tools.search.invoke({"relative_path": "**/*.txt", "pattern": "Earth"})
        lines = out.splitlines()
        assert "Earth is here [file: planets.txt, lines:0-1]" in lines
        assert "Earth again [file: planets.txt, lines:2-3]" in lines


def test_performer_tools_file_meta():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        payload = "hello"
        root.joinpath("file1.txt").write_text(payload)

        tools = create_performer_tools(
            start_time_stamp=0,
            time_limit_s=60,
            path_to_corpora=root,
        )

        out = tools.file_meta.invoke({"relative_path": "file1.txt"})
        value, unit = out.split()
        assert unit == "MB"
        assert abs(float(value) - (len(payload) / 1024 / 1024)) < 1e-6


def test_performer_tools_time_elapsed_and_left():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        start_time_stamp = int(time.time())
        time_limit_s = 5

        tools = create_performer_tools(
            start_time_stamp=start_time_stamp,
            time_limit_s=time_limit_s,
            path_to_corpora=root,
        )

        elapsed = tools.time_elapsed.invoke({})
        left = tools.time_left.invoke({})

        assert 0 <= elapsed <= 2
        assert time_limit_s - 2 <= left <= time_limit_s


def test_validator_tools_resolve_reference():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        root.joinpath("doc.txt").write_text("line0\nline1\nline2\nline3\n")

        tools = create_validator_tools(path_to_corpora=root)
        out = tools.resolve_reference.invoke({"relative_path": "doc.txt", "a": 1, "b": 3})
        assert out == ["line1", "line2"]
