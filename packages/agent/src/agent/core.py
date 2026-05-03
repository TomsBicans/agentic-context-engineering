import argparse
import os
import time
from enum import Enum
from pathlib import Path
from typing import Literal, Tuple, List

from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agent.interface.invoke import invoke_agent
from agent.interface.response import format_agent_response
from agent.interface.streaming import stream_agent
from agent.prompts import EXAMINEE_SYSTEM_MESSAGE, EXAMINER_SYSTEM_MESSAGE, TOOL_USE_ENFORCEMENT
from agent.tools import create_validator_tools, create_performer_tools


class AgentRole(Enum):
    EXAMINEE = "examinee"
    EXAMINER = "examiner"


class Statement(BaseModel):
    statement: str
    file_path: str
    lines: Tuple[int, int]


class ExamineeResponse(BaseModel):
    statements: List[Statement]


def initialize_agent(
        llm_model: str,
        role: Literal[AgentRole.EXAMINEE, AgentRole.EXAMINER],
        path_to_corpora: Path,
        temperature: float,
        num_ctx: int,
        time_limit: int,
        enforce_tools: bool,
        reasoning_enabled: bool,
) -> CompiledStateGraph:
    if role not in AgentRole:
        raise ValueError(f"Invalid role: {role}")

    if role == AgentRole.EXAMINEE:
        system_message = EXAMINEE_SYSTEM_MESSAGE
        if enforce_tools:
            system_message = f"{EXAMINEE_SYSTEM_MESSAGE}\n\n{TOOL_USE_ENFORCEMENT}"
        tools = create_performer_tools(
            start_time_stamp=int(time.time()),
            time_limit_s=time_limit,
            path_to_corpora=path_to_corpora
        )
        response_format = ToolStrategy(ExamineeResponse)
    elif role == AgentRole.EXAMINER:
        system_message = EXAMINER_SYSTEM_MESSAGE
        tools = create_validator_tools(
            path_to_corpora=path_to_corpora
        )
        response_format = None  # TODO: implement
    else:
        raise ValueError(f"Invalid role: {role}")

    llm_model = ChatOllama(
        model=llm_model,
        reasoning=reasoning_enabled,
        base_url="http://localhost:11434",
        temperature=temperature,
        num_ctx=num_ctx,
    )

    return create_agent(
        model=llm_model,
        tools=tools.as_list(),
        system_prompt=system_message,
        response_format=response_format,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model", type=str, default="qwen3:4b", required=True)
    parser.add_argument("--num_ctx", type=int, default=8192, required=True)
    parser.add_argument("--role", type=str, choices=[AgentRole.EXAMINEE.value, AgentRole.EXAMINER.value], required=True)
    parser.add_argument("--path-to-corpora", type=str, help="Absolute path to a directory on the operating system")
    parser.add_argument("--require-tools", dest="require_tools", action="store_true")
    parser.add_argument("--no-require-tools", dest="require_tools", action="store_false")
    parser.set_defaults(require_tools=True)
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
    parser.add_argument("--reasoning-enabled", dest="reasoning_enabled", action="store_true")
    parser.add_argument("--no-reasoning-enabled", dest="reasoning_enabled", action="store_false")
    parser.set_defaults(reasoning_enabled=False)
    return parser.parse_args()


def main():
    args = parse_args()
    path_to_corpora = args.path_to_corpora
    if not os.path.exists(path_to_corpora) and not Path(path_to_corpora).is_dir():
        raise ValueError(f"Path to corpora does not exist: {path_to_corpora}")

    agent = initialize_agent(
        llm_model=args.model,
        role=AgentRole.EXAMINEE,
        path_to_corpora=Path(path_to_corpora).absolute(),
        temperature=0.03,
        num_ctx=args.num_ctx,
        time_limit=60,
        enforce_tools=args.require_tools,
        reasoning_enabled=args.reasoning_enabled,
    )
    if args.stream:
        response = stream_agent(agent, args.prompt)
        formatted_response = format_agent_response(response)
        print(formatted_response)
        if args.require_tools and response.tool_messages == 0:
            raise RuntimeError("Tool use is required but no tool calls were made.")
        return response

    t_start = time.perf_counter()
    response = invoke_agent(agent, args.prompt)
    elapsed = time.perf_counter() - t_start
    print()
    formatted_response = format_agent_response(response)
    print(formatted_response)
    print(f"Execution time: {elapsed:.3f}s")
    return response


if __name__ == "__main__":
    main()
