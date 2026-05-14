import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from agent.prompts import EXAMINEE_SYSTEM_MESSAGE
from experiment_runner.models.metrics import RunMetrics, TokenCounts
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult
from experiment_runner.models.trace import SessionTrace, TraceStep
from experiment_runner.runners.base import BaseRunner

# Per-question wall-clock budget for the Codex subprocess.
_TIMEOUT_SECONDS = 180

# Ollama local provider identifier passed to the Codex CLI.
_LOCAL_PROVIDER = "ollama"


def build_codex_command(*, model: str, prompt: str) -> list[str]:
    return [
        "codex", "exec",
        "--oss",
        "--local-provider", _LOCAL_PROVIDER,
        "-m", model,
        "--sandbox", "read-only",
        "--skip-git-repo-check",
        "--json",
        prompt,
    ]


class GptCodexLocalRunner(BaseRunner):
    """Runner for the OpenAI Codex CLI baseline against a local Ollama model.

    Each question is executed inside a fresh temporary workspace so the real
    corpus directory is never written to and concurrent runs cannot interfere
    with each other. The corpus is exposed via a single symlink at ``./corpus``
    inside the temp workspace, keeping file paths stable in the model's context.

    The Codex CLI is invoked in read-only sandbox mode with a local Ollama
    provider, so the agent can read corpus files but cannot mutate the
    workspace:

        codex exec --oss --local-provider ollama -m <model>
                   --sandbox read-only --skip-git-repo-check --json <prompt>

    The prompt is passed as the final positional argument (no ``-p`` flag).

    Codex emits JSON Lines on stdout — one object per event. The runner
    scans the stream for ``item.completed`` events whose ``item.type`` is
    ``"agent_message"`` and uses the last such item's ``text`` field as the
    answer. Token counts are extracted from the ``turn.completed`` event.
    Stderr lines (e.g. Codex model-download progress or non-fatal error logs)
    are tolerated and not treated as failures; only a non-zero exit code or a
    missing agent_message triggers an error.

    Any subprocess, parsing, or I/O failure is converted into a populated
    ``answer_error`` instead of a raised exception.
    """

    def run(self, question: Question) -> RunResult:
        result = self._base_result(question)

        if self.config.path_to_corpora is None:
            result.answer_error = "GptCodexLocalRunner requires path_to_corpora in RunConfig"
            result.metrics = RunMetrics()
            return result

        corpus_root = Path(self.config.path_to_corpora).resolve()
        if not corpus_root.exists():
            result.answer_error = f"Corpus path does not exist: {corpus_root}"
            result.metrics = RunMetrics()
            return result

        t_start = time.perf_counter()

        try:
            with tempfile.TemporaryDirectory(prefix="codex_run_") as tmp:
                workspace = Path(tmp)
                corpus_link = workspace / "corpus"
                corpus_link.symlink_to(corpus_root, target_is_directory=True)

                completed = self._invoke_codex(
                    self._build_prompt(question.question),
                    workspace,
                )

                if completed.returncode != 0:
                    detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
                    result.answer_error = (
                        f"codex exited with code {completed.returncode}: {detail[:500]}"
                    )
                    result.metrics = RunMetrics(
                        execution_time_s=time.perf_counter() - t_start,
                    )
                    return result

                events = self._parse_jsonlines(completed.stdout)

        except subprocess.TimeoutExpired:
            result.answer_error = f"codex timed out after {_TIMEOUT_SECONDS}s"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except FileNotFoundError:
            result.answer_error = "codex executable not found on PATH"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except Exception as exc:
            result.answer_error = f"GptCodexLocal runner failed: {exc}"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result

        execution_time = time.perf_counter() - t_start

        answer_text = self._extract_answer(events)
        tokens = self._extract_tokens(events)
        tool_sequence = self._extract_tool_sequence(events)

        result.answer_text = answer_text
        if self.config.store_trace:
            result.trace = SessionTrace(steps=self._extract_steps(events))
        result.metrics = RunMetrics(
            execution_time_s=execution_time,
            tokens=tokens,
            tool_call_count=len(tool_sequence),
            tool_call_sequence=tool_sequence,
            corpus_used=len(tool_sequence) > 0,
        )
        return result

    @staticmethod
    def _build_prompt(question: str) -> str:
        return f"{EXAMINEE_SYSTEM_MESSAGE}\n\nQuestion:\n{question}"

    def _invoke_codex(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        cmd = build_codex_command(model=self.config.model, prompt=prompt)

        env = os.environ.copy()
        # Strip vars set by an outer Claude Code session so the inner process
        # does not detect nesting and alter its output behaviour.
        for _var in ("CLAUDECODE", "AI_AGENT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXECPATH"):
            env.pop(_var, None)

        return subprocess.run(
            cmd,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )

    @staticmethod
    def _parse_jsonlines(stdout: str) -> list[dict[str, Any]]:
        """Parse Codex JSON Lines output, silently skipping non-JSON lines."""
        events: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return events

    @staticmethod
    def _extract_steps(events: list[dict[str, Any]]) -> list[TraceStep]:
        """Convert item.completed events into ordered TraceStep records."""
        steps: list[TraceStep] = []
        for event in events:
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            text = item.get("text")
            if item_type in ("reasoning", "agent_message") and isinstance(text, str):
                steps.append(TraceStep(type=item_type, content=text.strip() or None))
            elif GptCodexLocalRunner._is_tool_call_item_type(item_type):
                steps.append(
                    TraceStep(
                        type="tool_call",
                        name=GptCodexLocalRunner._tool_name(item),
                        input=json.dumps(item, ensure_ascii=False),
                    )
                )
            elif GptCodexLocalRunner._is_tool_result_item_type(item_type):
                steps.append(
                    TraceStep(
                        type="tool_result",
                        name=GptCodexLocalRunner._tool_name(item),
                        output=json.dumps(item, ensure_ascii=False),
                    )
                )
        return steps

    @staticmethod
    def _extract_answer(events: list[dict[str, Any]]) -> Optional[str]:
        """Return the last agent_message text from item.completed events."""
        answer: Optional[str] = None
        for event in events:
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            if item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    answer = text.strip() or None
        return answer

    @staticmethod
    def _extract_tool_sequence(events: list[dict[str, Any]]) -> list[str]:
        """Return command/tool-like item names from the Codex JSONL event stream."""
        sequence: list[str] = []
        for event in events:
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if not GptCodexLocalRunner._is_tool_call_item_type(item_type):
                continue
            sequence.append(GptCodexLocalRunner._tool_name(item))
        return sequence

    @staticmethod
    def _is_tool_call_item_type(item_type: Any) -> bool:
        if not isinstance(item_type, str):
            return False
        normalized = item_type.lower()
        if normalized in {"reasoning", "agent_message", "message"}:
            return False
        if any(marker in normalized for marker in ("output", "result")):
            return False
        return any(marker in normalized for marker in ("tool", "call", "command", "exec", "shell"))

    @staticmethod
    def _is_tool_result_item_type(item_type: Any) -> bool:
        if not isinstance(item_type, str):
            return False
        normalized = item_type.lower()
        return any(marker in normalized for marker in ("tool_result", "tool_output", "call_output", "command_output"))

    @staticmethod
    def _tool_name(item: dict[str, Any]) -> str:
        for key in ("name", "tool", "command", "cmd", "type"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list) and value:
                return " ".join(str(part) for part in value[:3])
        return "codex_tool"

    @staticmethod
    def _extract_tokens(events: list[dict[str, Any]]) -> Optional[TokenCounts]:
        """Extract token counts from the turn.completed event."""
        for event in reversed(events):
            if event.get("type") != "turn.completed":
                continue
            usage = event.get("usage")
            if not isinstance(usage, dict):
                continue

            def _coerce(*keys: str) -> Optional[int]:
                for key in keys:
                    value = usage.get(key)
                    if isinstance(value, int):
                        return value
                    if isinstance(value, float):
                        return int(value)
                return None

            return TokenCounts(
                input=_coerce("input_tokens"),
                output=_coerce("output_tokens"),
                cache_read=_coerce("cached_input_tokens"),
                cache_creation=None,
            )
        return None
