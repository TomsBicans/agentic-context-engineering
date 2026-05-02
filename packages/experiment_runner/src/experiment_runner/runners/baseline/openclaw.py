import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from agent.prompts import EXAMINEE_SYSTEM_MESSAGE
from experiment_runner.models.metrics import RunMetrics, TokenCounts
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult
from experiment_runner.runners.base import BaseRunner


# Per-question wall-clock budget for the OpenClaw CLI subprocess.
_TIMEOUT_SECONDS = 180

# Per-corpus agent name prefix. A dedicated isolated OpenClaw agent is created
# for each corpus so that workspace context and session state cannot cross
# between corpora (e.g. "thesis-oblivion_wiki").
_AGENT_PREFIX = "thesis"

# Overrides the ACE-specific tooling strategy section of EXAMINEE_SYSTEM_MESSAGE.
# OpenClaw exposes a `read` tool and an `exec` tool; there is no time_left(),
# list_paths(), or search() — those are ACE-specific.
_OPENCLAW_PROMPT_SUFFIX = """
OpenClaw runtime notes
- You are a research agent. Answer using ONLY the files in your workspace (the corpus).
- Use the `read` tool to open individual files by path.
- Use `exec` with a shell command (e.g. `ls`, `grep -r`) to discover file paths when needed.
- Do NOT use web_search or any tool that fetches external information.
- Every factual claim must be cited with a file path and approximate line range.
""".strip()


class OpenClawRunner(BaseRunner):
    """Runner for the OpenClaw baseline via the official `openclaw` CLI.

    Each corpus gets a dedicated isolated OpenClaw agent whose workspace is
    pointed at the corpus directory. This design avoids the BOOTSTRAP.md
    onboarding flow (corpus directories do not contain that file) and gives
    the model direct `read`-tool access to corpus files without additional
    configuration.

    Each question is sent on a freshly-generated session ID (``--session-id
    <uuid>``) so that conversation history from earlier questions cannot leak
    into the current answer.

    The runner shells out to:
        openclaw agent --agent <name> --session-id <uuid> --message <prompt> --json

    and maps the JSON envelope onto the framework's RunResult / RunMetrics
    types. Any subprocess, parsing, or I/O failure is converted into a
    populated ``answer_error`` instead of a raised exception.

    Agent creation (``openclaw agents add``) is lazy and idempotent — it runs
    once per corpus on first use and is skipped on subsequent calls.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._agent_name: Optional[str] = None

    def _ensure_agent(self) -> str:
        """Return the per-corpus agent name, creating the agent if needed."""
        if self._agent_name is not None:
            return self._agent_name

        if self.config.path_to_corpora is None:
            raise ValueError("OpenClawRunner requires path_to_corpora in RunConfig")

        corpus_dir = Path(self.config.path_to_corpora).resolve() / self.config.corpus.value
        if not corpus_dir.exists():
            raise ValueError(f"Corpus directory does not exist: {corpus_dir}")

        agent_name = f"{_AGENT_PREFIX}-{self.config.corpus.value}"

        if not self._agent_exists(agent_name):
            subprocess.run(
                [
                    "openclaw", "agents", "add", agent_name,
                    "--workspace", str(corpus_dir),
                    "--model", self.config.model,
                    "--non-interactive",
                ],
                capture_output=True, text=True, timeout=60, check=False,
            )

        self._agent_name = agent_name
        return agent_name

    @staticmethod
    def _agent_exists(name: str) -> bool:
        result = subprocess.run(
            ["openclaw", "agents", "list", "--json"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False
        try:
            agents = json.loads(result.stdout.strip())
            if isinstance(agents, list):
                return any(
                    isinstance(a, dict) and a.get("id") == name
                    for a in agents
                )
        except json.JSONDecodeError:
            pass
        return False

    def run(self, question: Question) -> RunResult:
        result = self._base_result(question)

        try:
            agent_name = self._ensure_agent()
        except ValueError as exc:
            result.answer_error = str(exc)
            result.metrics = RunMetrics()
            return result

        session_id = str(uuid.uuid4())
        t_start = time.perf_counter()

        try:
            completed = self._invoke_openclaw(
                self._build_prompt(question.question),
                agent_name,
                session_id,
            )
        except subprocess.TimeoutExpired:
            result.answer_error = f"openclaw timed out after {_TIMEOUT_SECONDS}s"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except FileNotFoundError:
            result.answer_error = "openclaw executable not found on PATH"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except Exception as exc:
            result.answer_error = f"OpenClaw runner failed: {exc}"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result

        execution_time = time.perf_counter() - t_start

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()[:500]
            result.answer_error = (
                f"openclaw exited with code {completed.returncode}: {stderr}"
            )
            result.metrics = RunMetrics(execution_time_s=execution_time)
            return result

        data = self._parse_json_output(completed.stdout)
        if data is None:
            result.answer_error = "Failed to parse JSON output from openclaw"
            result.metrics = RunMetrics(execution_time_s=execution_time)
            return result

        answer_text = self._extract_answer(data)
        aborted = data.get("result", {}).get("meta", {}).get("aborted", False)
        tokens = self._extract_tokens(data)

        result.answer_text = answer_text
        result.metrics = RunMetrics(
            execution_time_s=execution_time,
            tokens=tokens,
            corpus_used=answer_text is not None and not aborted,
        )
        return result

    @staticmethod
    def _build_prompt(question: str) -> str:
        return f"{EXAMINEE_SYSTEM_MESSAGE}\n\n{_OPENCLAW_PROMPT_SUFFIX}\n\nQuestion:\n{question}"

    def _invoke_openclaw(
        self, prompt: str, agent_name: str, session_id: str
    ) -> subprocess.CompletedProcess:
        cmd = [
            "openclaw", "agent",
            "--agent", agent_name,
            "--session-id", session_id,
            "--message", prompt,
            "--json",
        ]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )

    @staticmethod
    def _parse_json_output(stdout: str) -> Optional[dict[str, Any]]:
        if not stdout:
            return None
        stripped = stdout.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        # Fallback: extract the outermost {...} block in case of stray banner output.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(stripped[start: end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_answer(data: dict[str, Any]) -> Optional[str]:
        try:
            payloads = data["result"]["payloads"]
            if isinstance(payloads, list) and payloads:
                text = payloads[0].get("text")
                if isinstance(text, str):
                    return text.strip() or None
        except (KeyError, TypeError, IndexError):
            pass
        return None

    @staticmethod
    def _extract_tokens(data: dict[str, Any]) -> Optional[TokenCounts]:
        # Prefer lastCallUsage (per-turn) over usage (session aggregate).
        agent_meta = data.get("result", {}).get("meta", {}).get("agentMeta", {})
        usage = agent_meta.get("lastCallUsage") or agent_meta.get("usage")
        if not isinstance(usage, dict):
            return None

        def _coerce(*keys: str) -> Optional[int]:
            for key in keys:
                value = usage.get(key)
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)
            return None

        return TokenCounts(
            input=_coerce("input", "promptTokens"),
            output=_coerce("output"),
            cache_read=_coerce("cacheRead"),
            cache_creation=_coerce("cacheWrite"),
        )
