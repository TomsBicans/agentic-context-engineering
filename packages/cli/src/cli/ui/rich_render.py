import json
import re
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.interface.response import AgentResponse

console = Console()

_THINK_CLOSE = "</think>"

# Inline citation: "Some statement text [file: path/to/file.txt, lines:10-20]"
# The system prompt mandates this exact format from the Examinee.
_CITATION_RE = re.compile(
    r"(.+?)\s*\[file:\s*([^\],]+),\s*lines?:\s*(\d+)-(\d+)\]",
    re.MULTILINE,
)

# Match any variant the model might emit for the structured response block
_EMBEDDED_RE = re.compile(
    r"<(?:ExamineeResponse|examinee-response|examinee_response)>"
    r"\s*(\[.*?\])\s*"
    r"</(?:ExamineeResponse|examinee-response|examinee_response)>",
    re.DOTALL | re.IGNORECASE,
)



class _AceSegment:
    """Tracks tokens for one ace 'turn' (between tool calls).

    States:
      - reasoning=True  : an open <think> tag is present but no matching </think> yet
      - reasoning=False, think_chars>0 : reasoning completed (</think> was seen)
      - reasoning=False, think_chars==0: model generated no think blocks at all

    display returns visible answer text in all states — the cases are:
      - inside open <think>   → ""  (nothing to show yet)
      - post-</think> content → content after the last </think> (think-blocks stripped)
      - no <think> at all     → full _all content (model answered directly)
    """

    def __init__(self):
        self._all = ""
        self._last_close_end: int = -1  # index just after the last </think>

    def feed(self, token: str) -> None:
        overlap = len(_THINK_CLOSE) - 1
        search_from = max(0, len(self._all) - overlap)
        self._all += token
        pos = self._all.find(_THINK_CLOSE, search_from)
        while pos != -1:
            self._last_close_end = pos + len(_THINK_CLOSE)
            pos = self._all.find(_THINK_CLOSE, self._last_close_end)

    @property
    def reasoning(self) -> bool:
        """True only while actively inside an open (unclosed) <think> block."""
        if self._last_close_end >= 0:
            return False  # All think blocks completed
        return "<think>" in self._all

    @property
    def think_chars(self) -> int:
        """Chars generated in the reasoning phase.

        Uses _last_close_end as the metric so it works even when <think> is
        absent from the token stream (e.g. stripped by the Ollama layer) but
        </think> did arrive to delimit the end of reasoning.
        Returns 0 if no </think> was ever seen.
        """
        return max(0, self._last_close_end)

    @property
    def display(self) -> str:
        """Visible answer text (think-block content stripped)."""
        if self._last_close_end >= 0:
            # Completed reasoning — show post-close content
            return _strip_think_blocks(self._all[self._last_close_end:].lstrip())
        if "<think>" in self._all:
            return ""  # Still inside an open think block
        # No think blocks at all — the whole response is the answer
        return _strip_think_blocks(self._all).strip()

    @property
    def final_content(self) -> str:
        """Cleaned content for post-processing / statement extraction."""
        if self._last_close_end >= 0:
            return _strip_think_blocks(self._all[self._last_close_end:].lstrip())
        return _strip_think_blocks(self._all).strip()


def _strip_think_blocks(text: str) -> str:
    """Remove any remaining <think>…</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()



def render_stream_live(agent, prompt: str) -> AgentResponse:
    """Stream agent execution as a chronological timeline of panels.

    The `segments` list grows left-to-right as streaming progresses:

        [ace]  →  [tool 1]  →  [ace]  →  [tool 2]  →  [ace (final)]

    Each [ace] panel shows the visible tokens for that turn (think-blocks
    stripped).  While a segment is in the reasoning phase it shows a dim
    "Reasoning…" placeholder.  Completed ace segments with no visible content
    are omitted from the rendered output.

    Returns the final AgentResponse when streaming completes.
    """
    from agent.interface.events import iter_stream_events

    # segments: alternating _AceSegment and tool-dicts {"call", "result"}
    segments: list = [_AceSegment()]
    final_response: Optional[AgentResponse] = None
    tool_count = [0]  # mutable counter visible to inner functions

    def _current_ace() -> _AceSegment | None:
        return segments[-1] if isinstance(segments[-1], _AceSegment) else None

    def _build_renderable() -> Group:
        parts: list = []
        last_seg = segments[-1]
        t = 0  # tool counter for labels

        for seg in segments:
            if isinstance(seg, _AceSegment):
                tc = seg.think_chars
                display = seg.display

                if seg.reasoning:
                    # Still inside a think block — animated indicator with growing count
                    label = f"Reasoning…  ({tc:,} chars)" if tc else "Reasoning…"
                    parts.append(
                        Panel(
                            Text(label, style="dim italic"),
                            title="[bold yellow]think[/bold yellow]",
                            border_style="yellow",
                            padding=(0, 1),
                        )
                    )
                else:
                    # Think block(s) completed — show collapsed summary if there was any
                    if tc:
                        parts.append(
                            Panel(
                                Text(f"{tc:,} chars", style="dim"),
                                title="[dim yellow]think[/dim yellow]",
                                border_style="dim yellow",
                                padding=(0, 1),
                            )
                        )
                    # Then show the answer text (if any)
                    if display:
                        parts.append(
                            Panel(
                                Text(display),
                                title="[bold cyan]ace[/bold cyan]",
                                border_style="cyan",
                                padding=(0, 1),
                            )
                        )
                    elif seg is last_seg:
                        # Seen </think> but no answer yet — transient gap
                        parts.append(
                            Panel(
                                Text("Waiting for response…", style="dim"),
                                title="[bold cyan]ace[/bold cyan]",
                                border_style="cyan",
                                padding=(0, 1),
                            )
                        )
            else:
                t += 1
                body = Text()
                body.append("call   ", style="bold yellow")
                body.append(seg["call"])
                if seg["result"] is not None:
                    body.append("\nresult ", style="bold green")
                    body.append(f"{seg['result']['name']}  ")
                    body.append(seg["result"]["snippet"])
                else:
                    body.append("\n")
                    body.append("running…", style="dim")
                parts.append(
                    Panel(
                        body,
                        title=f"[yellow]tool {t}[/yellow]",
                        border_style="dim yellow",
                        padding=(0, 1),
                    )
                )

        return Group(*parts)

    with Live(_build_renderable(), refresh_per_second=25, console=console) as live:
        for event_type, payload in iter_stream_events(agent, prompt):
            if event_type == "token":
                ace = _current_ace()
                if ace is None:
                    # First token after a tool call: start a new ace segment
                    ace = _AceSegment()
                    segments.append(ace)
                prev_display = ace.display
                prev_reasoning = ace.reasoning
                ace.feed(payload)
                if ace.display != prev_display or ace.reasoning != prev_reasoning:
                    live.update(_build_renderable())

            elif event_type == "tool_call":
                segments.append({"call": payload, "result": None})
                tool_count[0] += 1
                live.update(_build_renderable())

            elif event_type == "tool_result":
                # payload is {"name": str, "snippet": str}
                for seg in reversed(segments):
                    if isinstance(seg, dict) and seg["result"] is None:
                        seg["result"] = payload
                        break
                live.update(_build_renderable())

            elif event_type == "done":
                final_response = payload
                live.update(_build_renderable())

    # Use the last ace segment's cleaned content for statement extraction
    last_ace = next((s for s in reversed(segments) if isinstance(s, _AceSegment)), None)
    if final_response and last_ace:
        final_response.message_content = last_ace.final_content

    return final_response



def render_statements(response: AgentResponse, k: int = 10, fmt: str = "table") -> None:
    statements = _extract_statements(response)

    if fmt == "json":
        data = [
            {"statement": s["statement"], "source": s["source"], "lines": s["lines"]}
            for s in statements[:k]
        ]
        console.print_json(json.dumps(data))
        return

    if fmt == "md":
        lines = []
        for i, s in enumerate(statements[:k], start=1):
            lines.append(f"{i}. {s['statement']}")
            if s["source"]:
                lines.append(f"   *Source: {s['source']}, lines {s['lines']}*")
        console.print(Markdown("\n".join(lines)))
        return

    # Default: table
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Statement")
    table.add_column("Source", style="cyan", no_wrap=False)
    table.add_column("Lines", style="green", width=12)

    for i, s in enumerate(statements[:k], start=1):
        table.add_row(str(i), s["statement"], s["source"], s["lines"])

    if table.row_count == 0:
        console.print("[yellow]No statements found in response.[/yellow]")
    else:
        console.print(table)


def render_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")



def _extract_statements(response: AgentResponse) -> list[dict]:
    """Extract [{statement, source, lines}] from an AgentResponse.

    Priority:
      1. structured_response  — Pydantic ExamineeResponse with .statements
      2. Embedded JSON block  — <examinee-response>[…]</examinee-response> in message_content
      3. Inline citations     — "claim [file: path, lines:a-b]" from system prompt format
      4. Paragraph fallback   — double-newline split of cleaned message_content
    """
    structured = response.structured_response

    # 1. Proper structured output from LangGraph ToolStrategy
    if structured is not None and hasattr(structured, "statements"):
        result = []
        for stmt in structured.statements:
            lines_str = f"{stmt.lines[0]}-{stmt.lines[1]}" if stmt.lines else ""
            result.append({
                "statement": stmt.statement,
                "source": stmt.file_path or "",
                "lines": lines_str,
            })
        return result

    text = response.message_content or ""

    # 2. Model embedded the structured block as raw text
    parsed = _try_parse_embedded_response(text)
    if parsed is not None:
        return parsed

    # 3. Inline citation format mandated by the system prompt:
    #    "statement text [file: relative/path.txt, lines:10-20]"
    cited = _try_parse_inline_citations(text)
    if cited is not None:
        return cited

    # 4. Last resort: paragraph-level split so the table still shows something.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [{"statement": p, "source": "", "lines": ""} for p in paragraphs]


def _try_parse_inline_citations(text: str) -> list[dict] | None:
    """Parse 'statement [file: path, lines:a-b]' inline citations.

    Matches the citation format the Examinee system prompt mandates:
        The Sun's mass is 1.99e30 kg [file: solar_system.txt, lines:42-43]
    Returns None if no citations are found so the caller can try the next tier.
    """
    results = []
    for m in _CITATION_RE.finditer(text):
        statement = m.group(1).strip().lstrip("-•* ")
        if not statement:
            continue
        results.append({
            "statement": statement,
            "source": m.group(2).strip(),
            "lines": f"{m.group(3)}-{m.group(4)}",
        })
    return results if results else None


def _try_parse_embedded_response(text: str) -> list[dict] | None:
    """Parse a JSON array from an embedded structured-response tag."""
    match = _EMBEDDED_RE.search(text)
    if not match:
        return None
    try:
        items = json.loads(match.group(1))
        result = []
        for item in items:
            raw_lines = item.get("lines", [0, 0])
            if isinstance(raw_lines, (list, tuple)) and len(raw_lines) == 2:
                lines_str = f"{raw_lines[0]}-{raw_lines[1]}"
            else:
                lines_str = str(raw_lines)
            result.append({
                "statement": item.get("statement", ""),
                "source": item.get("file_path", ""),
                "lines": lines_str,
            })
        return result
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None
