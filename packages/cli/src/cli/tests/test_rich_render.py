from unittest.mock import MagicMock

from rich.console import Console

from agent.interface.response import AgentResponse
from cli.ui.rich_render import render_error, render_statements


def _make_console() -> Console:
    return Console(record=True, width=120)


def _mock_response(statements=None, message_content="") -> AgentResponse:
    structured = None
    if statements is not None:
        structured = MagicMock()
        stmt_objects = []
        for s in statements:
            stmt = MagicMock()
            stmt.statement = s["statement"]
            stmt.file_path = s.get("file_path", "")
            stmt.lines = s.get("lines", (0, 1))
            stmt_objects.append(stmt)
        structured.statements = stmt_objects

    return AgentResponse(
        message_content=message_content,
        structured_response=structured,
        human_messages=0,
        ai_messages=0,
        tool_messages=0,
        steps=[],
    )


def test_render_statements_table(monkeypatch):
    """render_statements table format includes statement text."""
    import cli.ui.rich_render as rr
    test_console = _make_console()
    monkeypatch.setattr(rr, "console", test_console)

    response = _mock_response(
        statements=[
            {"statement": "The sky is blue.", "file_path": "docs/sky.txt", "lines": (10, 15)},
            {"statement": "Water is wet.", "file_path": "docs/water.txt", "lines": (0, 3)},
        ]
    )
    render_statements(response, k=10, fmt="table")

    output = test_console.export_text()
    assert "The sky is blue." in output
    assert "Water is wet." in output
    assert "docs/sky.txt" in output


def test_render_statements_respects_k(monkeypatch):
    """render_statements only shows up to k rows."""
    import cli.ui.rich_render as rr
    test_console = _make_console()
    monkeypatch.setattr(rr, "console", test_console)

    statements = [{"statement": f"Statement {i}", "file_path": "f.txt", "lines": (i, i + 1)} for i in range(10)]
    response = _mock_response(statements=statements)
    render_statements(response, k=3, fmt="table")

    output = test_console.export_text()
    assert "Statement 0" in output
    assert "Statement 2" in output
    assert "Statement 3" not in output


def test_render_statements_json(monkeypatch):
    """render_statements json format outputs valid JSON."""
    import json
    import cli.ui.rich_render as rr
    test_console = _make_console()
    monkeypatch.setattr(rr, "console", test_console)

    response = _mock_response(
        statements=[{"statement": "Fact A.", "file_path": "a.txt", "lines": (0, 1)}]
    )
    render_statements(response, k=10, fmt="json")

    output = test_console.export_text().strip()
    # export_text strips markup so JSON should be parseable
    data = json.loads(output)
    assert isinstance(data, list)
    assert data[0]["statement"] == "Fact A."


def test_render_statements_fallback_to_message_content(monkeypatch):
    """render_statements falls back to paragraph splitting when no structured response."""
    import cli.ui.rich_render as rr
    test_console = _make_console()
    monkeypatch.setattr(rr, "console", test_console)

    # Double-newline separates paragraphs → two rows
    response = _mock_response(message_content="First paragraph.\n\nSecond paragraph.")
    render_statements(response, k=10, fmt="table")

    output = test_console.export_text()
    assert "First paragraph." in output
    assert "Second paragraph." in output


def test_ace_segment_reasoning_flag():
    """_AceSegment.reasoning is True only while actively inside an open <think> block."""
    from cli.ui.rich_render import _AceSegment
    t = _AceSegment()
    assert t.reasoning is False  # no think block yet
    t.feed("<think>")
    assert t.reasoning is True  # inside open think block
    t.feed("some reasoning text")
    assert t.reasoning is True
    t.feed("</think>")
    assert t.reasoning is False  # think block closed


def test_ace_segment_hides_pre_think_content():
    """_AceSegment.display is empty while inside an open <think> block."""
    from cli.ui.rich_render import _AceSegment
    t = _AceSegment()
    t.feed("<think>Lots of reasoning text that should be hidden.")
    assert t.display == ""


def test_ace_segment_shows_post_think_content():
    """_AceSegment.display contains only the text after the last </think>."""
    from cli.ui.rich_render import _AceSegment
    t = _AceSegment()
    t.feed("reasoning</think>The real answer.")
    assert t.display == "The real answer."


def test_ace_segment_uses_last_think_close():
    """_AceSegment resets to content after the LAST </think>."""
    from cli.ui.rich_render import _AceSegment
    t = _AceSegment()
    t.feed("first reasoning</think>intermediate</think>final answer")
    assert t.display == "final answer"


def test_ace_segment_think_chars_closed_block():
    """_AceSegment.think_chars is > 0 (position past </think>) when a think block completed."""
    from cli.ui.rich_render import _AceSegment
    t = _AceSegment()
    t.feed("<think>hello world</think>answer")
    # think_chars = _last_close_end = index just past the </think> tag
    assert t.think_chars == len("<think>hello world</think>")
    assert t.display == "answer"


def test_ace_segment_think_chars_open_block():
    """_AceSegment.think_chars is 0 while inside an open (unclosed) think block."""
    from cli.ui.rich_render import _AceSegment
    t = _AceSegment()
    t.feed("<think>still thinking")
    assert t.think_chars == 0  # no </think> seen yet
    assert t.reasoning is True


def test_ace_segment_think_chars_no_blocks():
    """_AceSegment.think_chars is zero when there are no think blocks."""
    from cli.ui.rich_render import _AceSegment
    t = _AceSegment()
    t.feed("plain answer text")
    assert t.think_chars == 0


def test_ace_segment_advances_past_every_close():
    """_AceSegment always shows content after the very last </think>, including inline ones."""
    from cli.ui.rich_render import _AceSegment
    t = _AceSegment()
    # Inline <think>…</think> in post-think segment: last </think> is the inline close.
    # display = everything after that final </think>.
    t.feed("preamble</think>answer text <think>aside</think> final part")
    assert "final part" in t.display
    assert "aside" not in t.display  # was inside an inline think block
    # "answer text" is before the inline </think> so it's not in the display segment either
    assert "preamble" not in t.display


def test_render_statements_parses_inline_citations(monkeypatch):
    """render_statements extracts 'statement [file: path, lines:a-b]' inline citations."""
    import cli.ui.rich_render as rr
    test_console = _make_console()
    monkeypatch.setattr(rr, "console", test_console)

    content = (
        "The Sun is a star [file: solar_system.txt, lines:10-12]\n"
        "It has 8 planets [file: planets.txt, lines:5-6]"
    )
    response = _mock_response(message_content=content)
    render_statements(response, k=10, fmt="table")

    output = test_console.export_text()
    assert "The Sun is a star" in output
    assert "solar_system.txt" in output
    assert "10-12" in output
    assert "It has 8 planets" in output
    assert "planets.txt" in output


def test_render_statements_inline_citations_with_bullets(monkeypatch):
    """render_statements strips bullet markers from inline-cited statements."""
    import cli.ui.rich_render as rr
    test_console = _make_console()
    monkeypatch.setattr(rr, "console", test_console)

    content = "- Mercury is closest to the Sun [file: planets.txt, lines:1-2]"
    response = _mock_response(message_content=content)
    render_statements(response, k=10, fmt="table")

    output = test_console.export_text()
    assert "Mercury is closest to the Sun" in output
    assert "planets.txt" in output


def test_render_statements_parses_camelcase_embedded(monkeypatch):
    """render_statements parses <ExamineeResponse>[...]</ExamineeResponse> from message_content."""
    import cli.ui.rich_render as rr
    test_console = _make_console()
    monkeypatch.setattr(rr, "console", test_console)

    content = (
        'Some preamble.\n'
        '<ExamineeResponse>\n'
        '[{"statement": "The sky is blue.", "file_path": "sky.txt", "lines": [10, 15]}]\n'
        '</ExamineeResponse>'
    )
    response = _mock_response(message_content=content)
    render_statements(response, k=10, fmt="table")

    output = test_console.export_text()
    assert "The sky is blue." in output
    assert "sky.txt" in output


def test_render_statements_parses_lowercase_embedded(monkeypatch):
    """render_statements parses <examinee-response>[...]</examinee-response> from message_content."""
    import cli.ui.rich_render as rr
    test_console = _make_console()
    monkeypatch.setattr(rr, "console", test_console)

    content = (
        '<examinee-response>\n'
        '[{"statement": "Water is wet.", "file_path": "water.txt", "lines": [0, 2]}]\n'
        '</examinee-response>'
    )
    response = _mock_response(message_content=content)
    render_statements(response, k=10, fmt="table")

    output = test_console.export_text()
    assert "Water is wet." in output
    assert "water.txt" in output


def test_render_error(monkeypatch):
    """render_error outputs 'Error:' prefix."""
    import cli.ui.rich_render as rr
    test_console = _make_console()
    monkeypatch.setattr(rr, "console", test_console)

    render_error("Something went wrong")

    output = test_console.export_text()
    assert "Error:" in output
    assert "Something went wrong" in output
