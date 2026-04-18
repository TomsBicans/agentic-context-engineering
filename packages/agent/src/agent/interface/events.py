from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from agent.interface.invoke import invoke_agent
from agent.interface.response import AgentResponse

# EventTuple: ("token", str) | ("tool_call", str) | ("tool_result", {"name", "snippet"}) | ("done", AgentResponse)
EventTuple = tuple[str, Any]


def _format_tool_content(content: Any, max_len: int = 120) -> str:
    """Return a single-line summary of a tool's return value."""
    if content is None:
        return "(no output)"
    if isinstance(content, list):
        # List of content blocks — join text parts
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or str(block))
        text = " | ".join(parts)
    else:
        text = str(content)
    # Collapse whitespace and truncate
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[:max_len - 1] + "…"
    return text or "(empty)"


def _format_tool_call(tool_call: dict) -> str:
    name = tool_call.get("name", "unknown_tool")
    args = tool_call.get("args")
    if args is None:
        return f"{name}()"
    if isinstance(args, dict):
        args_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"{name}({args_text})"
    return f"{name}({args})"


def iter_stream_events(agent: CompiledStateGraph, prompt: str) -> Iterator[EventTuple]:
    """Yield (event_type, payload) tuples from a streaming agent run.

    Event types:
      ("token", str)          - AI response token chunk
      ("tool_call", str)      - tool invocation starting (formatted as "name(args)")
      ("tool_result", dict)   - tool result received ({"name": str, "snippet": str})
      ("done", AgentResponse) - final response with all data
    """
    if not hasattr(agent, "stream"):
        response = invoke_agent(agent, prompt)
        yield ("done", response)
        return

    final_content = ""
    tool_calls_seen = 0
    last_state: dict = {}

    for mode, chunk in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode=["messages", "values"],
    ):
        if mode == "values":
            last_state = chunk
            continue

        if isinstance(chunk, tuple):
            message = chunk[0]
        else:
            message = chunk

        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls_seen += 1
                yield ("tool_call", _format_tool_call(tool_call))
            continue

        if isinstance(message, ToolMessage):
            tool_name = message.name or "tool"
            yield ("tool_result", {"name": tool_name, "snippet": _format_tool_content(message.content)})
            continue

        if isinstance(message, (AIMessage, AIMessageChunk)) and message.content:
            yield ("token", message.content)
            final_content += message.content

    structured_response = last_state.get("structured_response") if isinstance(last_state, dict) else None

    yield (
        "done",
        AgentResponse(
            message_content=final_content,
            tool_messages=tool_calls_seen,
            structured_response=structured_response,
            human_messages=0,
            ai_messages=0,
            steps=[],
        ),
    )
