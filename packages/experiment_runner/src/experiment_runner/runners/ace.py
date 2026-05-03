import time
from pathlib import Path

from experiment_runner.models.metrics import RunMetrics, TokenCounts
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult
from experiment_runner.runners.base import BaseRunner


class AceRunner(BaseRunner):
    """Runner for the ACE agent (packages/agent).

    Initialises the LangGraph agent once on first call and reuses it across
    questions to avoid paying the model-load cost on every run.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._agent = None

    def _get_agent(self):
        if self._agent is not None:
            return self._agent

        from agent.core import AgentRole, initialize_agent

        if self.config.path_to_corpora is None:
            raise ValueError("AceRunner requires path_to_corpora in RunConfig")

        self._agent = initialize_agent(
            llm_model=self.config.model,
            role=AgentRole.EXAMINEE,
            path_to_corpora=Path(self.config.path_to_corpora),
            temperature=0.03,
            num_ctx=(self.config.inference_config or {}).get("num_ctx", 8192),
            time_limit=60,
            enforce_tools=True,
            reasoning_enabled=self.config.reasoning_enabled,
        )
        return self._agent

    def run(self, question: Question) -> RunResult:
        from agent.interface.events import iter_stream_events

        result = self._base_result(question)
        agent = self._get_agent()

        tool_sequence: list[str] = []
        answer_parts: list[str] = []
        t_start = time.perf_counter()

        try:
            for event_type, payload in iter_stream_events(agent, question.question):
                if event_type == "token":
                    answer_parts.append(payload)
                elif event_type == "tool_call":
                    tool_sequence.append(payload)
                elif event_type == "done":
                    pass
        except Exception as exc:
            result.answer_error = str(exc)
            result.metrics = RunMetrics(
                execution_time_s=time.perf_counter() - t_start,
                tool_call_sequence=tool_sequence,
            )
            return result

        execution_time = time.perf_counter() - t_start
        result.answer_text = "".join(answer_parts).strip() or None
        result.metrics = RunMetrics(
            execution_time_s=execution_time,
            tool_call_count=len(tool_sequence),
            tool_call_sequence=tool_sequence,
            corpus_used=len(tool_sequence) > 0,
        )
        return result
