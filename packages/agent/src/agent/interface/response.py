from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class StepType(str, Enum):
    human_message = "HumanMessage"
    ai_message = "AIMessage"
    tool_message = "ToolMessage"

@dataclass(frozen=True)
class HumanStep:
    type: StepType
    content: Optional[str]


@dataclass(frozen=True)
class AIStep:
    type: StepType
    content: Optional[str]
    reasoning: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]


@dataclass(frozen=True)
class ToolStep:
    type: StepType
    content: Optional[str]
    tool_name: Optional[str]
    tool_call_id: Optional[str]


MessageStep = Union[HumanStep, AIStep, ToolStep]


@dataclass
class AgentResponse:
    message_content: str
    structured_response: Optional[str]
    human_messages: int
    ai_messages: int
    tool_messages: int
    steps: List[MessageStep]


def format_agent_response(response: AgentResponse) -> str:
    lines: List[str] = []
    tool_call_index: Dict[str, Dict[str, Any]] = {}
    for step in response.steps:
        if isinstance(step, AIStep) and step.tool_calls:
            for call in step.tool_calls:
                if isinstance(call, dict):
                    call_id = call.get("id")
                    if call_id:
                        tool_call_index[call_id] = call
    lines.append("AGENT RESPONSE")
    lines.append("=" * 70)
    lines.append(f"human_messages: {response.human_messages}")
    lines.append(f"ai_messages: {response.ai_messages}")
    lines.append(f"tool_messages: {response.tool_messages}")
    if response.structured_response is not None:
        lines.append("structured_response:")
        lines.append(str(response.structured_response))
    lines.append("=" * 70)
    lines.append("STEPS (chronological)")
    lines.append("-" * 70)

    for idx, step in enumerate(response.steps, start=1):
        lines.append(f"[{idx:03d}] {step.type.value}")
        if isinstance(step, ToolStep):
            lines.append(f"tool_name: {step.tool_name}")
            lines.append(f"tool_call_id: {step.tool_call_id}")
            if step.tool_call_id and step.tool_call_id in tool_call_index:
                call = tool_call_index[step.tool_call_id]
                args = call.get("args")
                lines.append(f"tool_args: {args}")
            if step.content:
                lines.append("content:")
                lines.append(step.content)
        elif isinstance(step, AIStep):
            if step.reasoning:
                lines.append("reasoning:")
                lines.append(step.reasoning)
            if step.content:
                lines.append("content:")
                lines.append(step.content)
            if step.tool_calls:
                lines.append("tool_calls:")
                for call in step.tool_calls:
                    if isinstance(call, dict):
                        name = call.get("name", "unknown_tool")
                        args = call.get("args")
                        lines.append(f"- {name} args={args}")
                    else:
                        lines.append(f"- {call}")
        elif isinstance(step, HumanStep):
            if step.content:
                lines.append("content:")
                lines.append(step.content)
        lines.append("-" * 70)

    return "\n".join(lines)
