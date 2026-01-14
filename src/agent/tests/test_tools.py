from pathlib import Path

from src.agent.tools import create_performer_tools
from src.agent.tools import create_validator_tools
from tempfile import TemporaryDirectory

def test_performer_tools_read_file():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        text = "Hello, world!"
        root.joinpath("file1.txt").write_text(text)
        root.joinpath("file2.txt").write_text(text)

        tools = create_performer_tools(
            start_time_stamp=0,
            time_limit_s=60,
            path_to_corpora=root,
        )

        # Test read_file
        out = tools.read_file.invoke({"relative_path": "file1.txt"})
        assert out == text

        out = tools.read_file.invoke({"relative_path": "file3.txt"})
        assert "error" in str(out).lower()


