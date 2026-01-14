import sys
import time
from typing import Tuple
from langchain_core.messages import AIMessage, ToolMessage, AIMessageChunk
from langgraph.graph.state import CompiledStateGraph


def log_status(start_time: float, message: str) -> None:
    elapsed_s = int(time.time() - start_time)
    print(f"[{elapsed_s:>4}s] {message}", file=sys.stderr, flush=True)


def format_tool_call(tool_call) -> str:
    name = tool_call.get("name", "unknown_tool")
    args = tool_call.get("args")
    if args is None:
        return f"{name}()"
    if isinstance(args, dict):
        args_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"{name}({args_text})"
    return f"{name}({args})"


def stream_agent(agent: CompiledStateGraph, prompt: str) -> Tuple[str, int]:
    start_time = time.time()
    if not hasattr(agent, "stream"):
        log_status(start_time, "stream not available; falling back to invoke")
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, print_mode="values")
        content = result["messages"][-1].content
        print(content)
        return content, 0

    log_status(start_time, "starting agent stream")

    final_content = ""
    tool_calls_seen = 0
    for event in agent.stream({"messages": [{"role": "user", "content": prompt}]}, stream_mode="messages"):
        if isinstance(event, tuple):
            message = event[0]
        else:
            message = event

        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls_seen += 1
                log_status(start_time, f"tool call: {format_tool_call(tool_call)}")
            continue

        if isinstance(message, ToolMessage):
            tool_name = message.name or "tool"
            log_status(start_time, f"tool result: {tool_name}")
            continue

        if isinstance(message, (AIMessage, AIMessageChunk)) and message.content:
            print(message.content, end="", flush=True)
            final_content += message.content

    if final_content:
        print("", flush=True)
    log_status(start_time, "done")
    return final_content, tool_calls_seen
