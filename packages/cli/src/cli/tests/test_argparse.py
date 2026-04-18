from unittest.mock import patch
import pytest

from cli.main import main


def test_query_defaults(capsys):
    """ace query <prompt> routes to run_query with correct defaults."""
    with patch("cli.commands.query.run_query") as mock_run:
        main(["query", "What is gravity?"])
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args.prompt == "What is gravity?"
    assert args.model == "qwen3:4b"
    assert args.num_ctx == 8192
    assert args.k == 10
    assert args.format == "table"
    assert args.time_limit == 60
    assert args.stream is True
    assert args.require_tools is True
    assert args.reasoning_enabled is False
    assert args.path_to_corpora is None


def test_query_custom_flags():
    """ace query with custom flags sets correct values."""
    with patch("cli.commands.query.run_query") as mock_run:
        main([
            "query", "List planets",
            "--format", "json",
            "--k", "5",
            "--model", "llama3",
            "--no-stream",
            "--no-require-tools",
            "--reasoning-enabled",
            "--num-ctx", "4096",
            "--time-limit", "30",
            "--path-to-corpora", "/data/corpus",
        ])
    args = mock_run.call_args[0][0]
    assert args.format == "json"
    assert args.k == 5
    assert args.model == "llama3"
    assert args.stream is False
    assert args.require_tools is False
    assert args.reasoning_enabled is True
    assert args.num_ctx == 4096
    assert args.time_limit == 30
    assert args.path_to_corpora == "/data/corpus"


def test_no_subcommand_launches_repl():
    """ace with no subcommand routes to run_repl."""
    with patch("cli.repl.session.run_repl") as mock_repl:
        main([])
    mock_repl.assert_called_once()


def test_missing_prompt_exits_with_code_2():
    """ace query with no prompt exits with code 2."""
    with pytest.raises(SystemExit) as exc_info:
        main(["query"])
    assert exc_info.value.code == 2
