from typing import List

from langgraph.graph.state import CompiledStateGraph

from src.agent.interface.response import AgentResponse, AIStep, HumanStep, ToolStep, MessageStep, StepType


def invoke_agent(agent: CompiledStateGraph, prompt: str) -> AgentResponse:
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, print_mode="values")
    messages = result["messages"]
    content = messages[-1].content
    structured_response = result.get("structured_response")

    # source: https://docs.langchain.com/oss/python/langchain/tools
    human_msgs = sum(1 for m in messages if m.__class__.__name__ == StepType.human_message.value)
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == StepType.ai_message.value)
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == StepType.tool_message.value)

    steps: List[MessageStep] = []
    for message in messages:
        msg_type = message.__class__.__name__
        content = getattr(message, "content", None)
        if msg_type == StepType.human_message.value:
            steps.append(HumanStep(type=StepType.human_message, content=content))
            continue
        if msg_type == StepType.tool_message.value:
            steps.append(
                ToolStep(
                    type=StepType.tool_message,
                    content=content,
                    tool_name=getattr(message, "name", None),
                    tool_call_id=getattr(message, "tool_call_id", None),
                )
            )
            continue
        if msg_type == StepType.ai_message.value:
            additional = getattr(message, "additional_kwargs", {}) or {}
            reasoning = additional.get("reasoning_content")
            tool_calls = getattr(message, "tool_calls", None)
            steps.append(
                AIStep(
                    type=StepType.ai_message,
                    content=content,
                    reasoning=reasoning,
                    tool_calls=tool_calls,
                )
            )
            continue

    return AgentResponse(
        message_content=content,
        structured_response=structured_response,
        tool_messages=tool_msgs,
        human_messages=human_msgs,
        ai_messages=ai_msgs,
        steps=steps,
    )
