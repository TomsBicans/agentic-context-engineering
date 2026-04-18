from unittest.mock import MagicMock, patch


def _run_repl_with_input(input_text: str, extra_args=None):
    """Drive the REPL with mocked prompt input and capture console output."""
    from rich.console import Console
    import cli.ui.rich_render as rr
    import cli.repl.session as sess

    fake_console = Console(record=True, width=120)

    with patch.object(rr, "console", fake_console), \
            patch.object(sess, "console", fake_console):
        args = MagicMock()
        args.k = 10
        args.model = "qwen3:4b"
        args.num_ctx = 8192
        args.time_limit = 60
        args.require_tools = True
        args.reasoning_enabled = False
        args.path_to_corpora = None
        if extra_args:
            for k, v in extra_args.items():
                setattr(args, k, v)

        with patch("cli.repl.session.PromptSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            lines = [line for line in input_text.split("\n") if line]
            lines.append(EOFError())

            side_effects = []
            for item in lines:
                if isinstance(item, type) and issubclass(item, BaseException):
                    side_effects.append(item())
                elif isinstance(item, BaseException):
                    side_effects.append(item)
                else:
                    side_effects.append(item)

            mock_session.prompt.side_effect = side_effects

            from cli.repl.session import run_repl
            run_repl(args)

    return fake_console.export_text()


def test_help_command():
    """/help prints help text."""
    output = _run_repl_with_input("/help\n")
    assert "Slash commands" in output


def test_exit_command():
    """/exit terminates the REPL loop."""
    output = _run_repl_with_input("/exit\n")
    assert "Bye" in output


def test_quit_command():
    """/quit terminates the REPL loop."""
    output = _run_repl_with_input("/quit\n")
    assert "Bye" in output


def test_set_k():
    """/set k 5 updates k display confirmation."""
    output = _run_repl_with_input("/set k 5\n/exit\n")
    assert "k set to 5" in output


def test_set_model():
    """/set model llama3 updates model display confirmation."""
    output = _run_repl_with_input("/set model llama3\n/exit\n")
    assert "model set to" in output
    assert "llama3" in output


def test_set_model_resets_agent(tmp_path):
    """/set model after a query resets the agent so it reinitializes with the new model."""
    import cli.repl.session as sess
    from rich.console import Console
    import cli.ui.rich_render as rr

    fake_console = Console(record=True, width=120)
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.structured_response = None
    mock_response.message_content = "answer"

    with patch("cli.repl.session.PromptSession") as mock_session_cls, \
            patch("agent.core.initialize_agent", return_value=mock_agent), \
            patch("cli.ui.rich_render.render_stream_live", return_value=mock_response), \
            patch("cli.ui.rich_render.render_statements"), \
            patch.object(rr, "console", fake_console), \
            patch.object(sess, "console", fake_console):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.prompt.side_effect = [
            "What is gravity?",   # triggers agent init
            "/set model llama3",  # should reset agent
            EOFError(),
        ]

        args = MagicMock()
        args.k = 10
        args.model = "qwen3:4b"
        args.num_ctx = 8192
        args.time_limit = 60
        args.require_tools = True
        args.reasoning_enabled = False
        args.path_to_corpora = str(tmp_path)

        from cli.repl.session import run_repl
        run_repl(args)

    output = fake_console.export_text()
    assert "reinitialize" in output


def test_query_triggers_agent_init(tmp_path):
    """A real query triggers agent initialization (mocked)."""
    import cli.repl.session as sess
    from rich.console import Console
    import cli.ui.rich_render as rr

    fake_console = Console(record=True, width=120)
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.structured_response = None
    mock_response.message_content = "answer"

    with patch("cli.repl.session.PromptSession") as mock_session_cls, \
            patch("agent.core.initialize_agent", return_value=mock_agent) as mock_init, \
            patch("cli.ui.rich_render.render_stream_live", return_value=mock_response), \
            patch("cli.ui.rich_render.render_statements"), \
            patch.object(rr, "console", fake_console), \
            patch.object(sess, "console", fake_console):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.prompt.side_effect = ["What is gravity?", EOFError()]

        args = MagicMock()
        args.k = 10
        args.model = "qwen3:4b"
        args.num_ctx = 8192
        args.time_limit = 60
        args.require_tools = True
        args.reasoning_enabled = False
        args.path_to_corpora = str(tmp_path)

        from cli.repl.session import run_repl
        run_repl(args)

    mock_init.assert_called_once()
