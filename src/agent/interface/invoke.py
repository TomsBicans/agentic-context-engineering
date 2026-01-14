from langgraph.graph.state import CompiledStateGraph

from src.agent.interface.response import AgentResponse


def invoke_agent(agent: CompiledStateGraph, prompt: str) -> AgentResponse:
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, print_mode="values")
    messages = result["messages"]
    content = messages[-1].content
    structured_response = result.get("structured_response")

    # source: https://docs.langchain.com/oss/python/langchain/tools
    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return AgentResponse(
        message_content=content,
        structured_response=structured_response,
        tool_messages=tool_msgs,
        human_messages=human_msgs,
        ai_messages=ai_msgs
    )
