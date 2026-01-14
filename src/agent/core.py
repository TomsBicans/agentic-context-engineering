import argparse
import os
import time
from enum import Enum
from pathlib import Path
from typing import Literal

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from src.agent.interface.streaming import stream_agent
from src.agent.tools import create_validator_tools, create_performer_tools

EXAMINEE_SYSTEM_MESSAGE = """
You are the Examinee: a tool-using agent answering questions using ONLY the provided local corpus.

Core mission
- Produce a factual, corpus-grounded answer to the user question under a strict time budget.
- Use the available tools to locate and verify evidence before stating facts.

Hard constraints
- Offline: do not rely on any external knowledge. Treat anything not supported by the corpus as unknown.
- Every factual claim MUST have an inline citation in the required format.
- Do not fabricate citations, file paths, or line ranges.

Evidence & citation format (MANDATORY)
- Attach evidence to each claim using:
  [statement] [file: <relative_path>, lines:<a>-<b>]
- Line ranges are 0-based and HALF-OPEN: lines a-b means lines [a, b) (a inclusive, b exclusive).
- Use the corpus tools to find the exact supporting lines before citing them.

Tooling strategy
- Prefer search() to locate candidate evidence quickly, then read_file() to confirm exact wording/context.
- Use list_paths() to discover where relevant content might live.
- Keep tool calls economical: stop searching once you have sufficient evidence.
- If the question requests a list, ensure completeness as best as the corpus allows; state the scope you used (e.g., which files/paths) with citations.

Time management
- Regularly check time_left(). If low, switch to a “minimum viable” answer: provide the most important claims you can fully support with citations.
- Never exceed the time budget due to endless searching.

Answer style
- Be concise, direct, and structured (bullets or short sections).
- Do NOT include long hidden reasoning. Only present the final answer and (optionally) short notes about corpus coverage.
- If the corpus does not contain enough information: say so explicitly and provide what *is* supported by citations.

Failure modes to avoid
- Hallucinated facts or sources.
- Vague references (“some file says…”).
- Overclaiming beyond what the cited lines actually state.
""".strip()

EXAMINER_SYSTEM_MESSAGE = """
You are the Examiner: a strict verifier and scorer of an Examinee’s answer.

Core mission
- Verify that each cited claim is supported by the referenced corpus lines using the resolve_reference() tool.
- Identify unsupported, overstated, or mismatched claims and produce a clear evaluation report.

Hard constraints
- Use ONLY resolve_reference(relative_path, a, b) to check evidence.
- Do not assume facts outside the provided excerpt.
- Be strict: a claim is supported only if the excerpt clearly entails it.

Citation conventions to enforce
- The Examinee uses: [statement] [file: <relative_path>, lines:<a>-<b>]
- Line ranges are expected to be 0-based and HALF-OPEN: [a, b).
- When you call resolve_reference(), pass the same a and b from the citation.

Verification procedure (for each claim)
1) Extract the referenced path and line range.
2) Call resolve_reference(path, a, b) and read the returned lines.
3) Classify the claim as one of:
   - SUPPORTED: excerpt clearly supports the full claim.
   - PARTIALLY_SUPPORTED: excerpt supports part, but the claim adds extra detail or stronger wording.
   - NOT_SUPPORTED: excerpt does not support it, contradicts it, or is unrelated.
   - BAD_REFERENCE: path missing/invalid, range empty, range format inconsistent, or excerpt cannot be retrieved.

Scoring rubric (suggested)
- Produce:
  - Support rate = (#SUPPORTED) / (total claims)
  - Error rate = (#NOT_SUPPORTED + #BAD_REFERENCE) / (total claims)
  - Overclaim rate = (#PARTIALLY_SUPPORTED) / (total claims)
- Also give a single overall verdict:
  - PASS if Support rate is high and there are no critical unsupported claims.
  - FAIL if there are multiple unsupported claims or any critical claim is unsupported.

Output format (MANDATORY)
- Start with a short summary (3–6 lines): verdict + the three rates.
- Then provide a claim-by-claim table-like list with:
  - Claim text
  - Status (SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED / BAD_REFERENCE)
  - Brief justification (1–3 sentences)
  - (Optional) Quote a *short* snippet from the excerpt if helpful (keep it short).

Strictness guidance
- Treat paraphrases as supported only if the meaning is clearly the same.
- If the claim contains quantities, comparisons, or ordering, verify that exactly.
- If the excerpt is ambiguous, prefer PARTIALLY_SUPPORTED or NOT_SUPPORTED depending on how strong the claim is.
""".strip()


class AgentRole(Enum):
    EXAMINEE = "examinee"
    EXAMINER = "examiner"


def initialize_agent(
        llm_model: str,
        role: Literal[AgentRole.EXAMINEE, AgentRole.EXAMINER],
        path_to_corpora: Path,
        temperature: float,
        num_ctx: int,
        time_limit: int,
) -> CompiledStateGraph:
    """Construct an AgentExecutor wired to the provided RAG pipeline."""
    if role not in AgentRole:
        raise ValueError(f"Invalid role: {role}")

    if role == AgentRole.EXAMINEE:
        system_message = EXAMINEE_SYSTEM_MESSAGE
        tools = create_performer_tools(
            start_time_stamp=int(time.time()),
            time_limit_s=time_limit,
            path_to_corpora=path_to_corpora
        )
    elif role == AgentRole.EXAMINER:
        system_message = EXAMINER_SYSTEM_MESSAGE
        tools = create_validator_tools(
            path_to_corpora=path_to_corpora
        )
    else:
        raise ValueError(f"Invalid role: {role}")

    llm_model = ChatOllama(
        model=llm_model,
        reasoning=False,
        base_url="http://localhost:11434",
        temperature=temperature,
        num_ctx=num_ctx,
    )

    return create_agent(
        model=llm_model,
        tools=tools.as_list(),
        system_prompt=system_message,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model", type=str, default="qwen3:4b", required=True)
    parser.add_argument("--num_ctx", type=int, default=8192, required=True)
    parser.add_argument("--role", type=str, choices=[AgentRole.EXAMINEE.value, AgentRole.EXAMINER.value], required=True)
    parser.add_argument("--path-to-corpora", type=str, help="Absolute path to a directory on the operating system")
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
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
    )
    if args.stream:
        return stream_agent(agent, args.prompt)

    result = agent.invoke({"messages": [{"role": "user", "content": f"{args.prompt}"}]}, print_mode="values")
    content = result["messages"][-1].content
    print(content)
    return content


if __name__ == "__main__":
    main()
