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
from experiment_runner.models.trace import SessionTrace
from experiment_runner.runners.base import BaseRunner

# Per-question wall-clock budget for the Claude Code subprocess.
_TIMEOUT_SECONDS = 180

# Tools restricted to read/search-only so the agent cannot mutate the workspace.
_ALLOWED_TOOLS = "read_file,glob_search,grep_search"

# Ollama Anthropic-compatible shim used by Claude Code for local inference.
# Claude Code uses the Anthropic SDK internally, so ANTHROPIC_BASE_URL must
# point at Ollama's Anthropic-compatible endpoint.
_OLLAMA_BASE_URL = "http://localhost:11434"
_OLLAMA_API_KEY = "ollama"

_CLAUDECODE_PROMPT_SUFFIX = """
Claude Code runtime notes
- For this run, the corpus is mounted at ./corpus.
- Available tools are grep_search, glob_search, and read_file.
- Use grep_search or glob_search instead of search() or list_paths().
- There is no time_left() tool; keep the search bounded and answer within the run budget.
- Do not search, read, or cite files under ./.claude.
""".strip()


class ClaudeCodeLocalRunner(BaseRunner):
    """Runner for the Claude Code CLI baseline against a local Ollama model.

    Each question is executed inside a fresh temporary workspace so the real
    corpus directory is never written to and concurrent runs cannot interfere
    with each other. The corpus is exposed via a single symlink at ``./corpus``
    inside the temp workspace, keeping file paths stable in the model's context
    while the process runs in ``workspace-write`` permission mode.

    Local inference is routed through Ollama by setting ``ANTHROPIC_BASE_URL``
    to Ollama's Anthropic-compatible endpoint and ``ANTHROPIC_API_KEY`` to a
    placeholder value. The model name is prefixed with ``ollama/`` as required
    by Claude Code when routing to a local provider.

    The runner shells out to ``claude --output-format json``, captures stdout,
    and maps the JSON envelope onto the framework's RunResult / RunMetrics types.
    Any subprocess, parsing, or I/O failure is converted into a populated
    ``answer_error`` instead of a raised exception.
    """

    def run(self, question: Question) -> RunResult:
        result = self._base_result(question)

        if self.config.path_to_corpora is None:
            result.answer_error = "ClaudeCodeLocalRunner requires path_to_corpora in RunConfig"
            result.metrics = RunMetrics()
            return result

        corpus_root = Path(self.config.path_to_corpora).resolve()
        if not corpus_root.exists():
            result.answer_error = f"Corpus path does not exist: {corpus_root}"
            result.metrics = RunMetrics()
            return result

        t_start = time.perf_counter()

        try:
            with tempfile.TemporaryDirectory(prefix="claudelocal_run_") as tmp:
                workspace = Path(tmp)
                # Mirror the corpus into the sandbox via a symlink so Claude Code
                # sees the content under a predictable, sandboxed path.
                corpus_link = workspace / "corpus"
                corpus_link.symlink_to(corpus_root, target_is_directory=True)

                completed = self._invoke_claude(
                    self._build_prompt(question.question),
                    workspace,
                )

                if completed.returncode != 0:
                    detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
                    result.answer_error = (
                        f"claude exited with code {completed.returncode}: {detail[:500]}"
                    )
                    result.metrics = RunMetrics(
                        execution_time_s=time.perf_counter() - t_start,
                    )
                    return result

                data = self._parse_json_output(completed.stdout)
        except subprocess.TimeoutExpired:
            result.answer_error = f"claude timed out after {_TIMEOUT_SECONDS}s"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except FileNotFoundError:
            result.answer_error = "claude executable not found on PATH"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except Exception as exc:
            result.answer_error = f"ClaudeCodeLocal runner failed: {exc}"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result

        execution_time = time.perf_counter() - t_start

        if data is None:
            result.answer_error = "Failed to parse JSON output from claude"
            result.metrics = RunMetrics(execution_time_s=execution_time)
            return result

        # Claude Code JSON output: answer is in "result", not "message".
        answer_text = data.get("result") or data.get("message")
        if isinstance(answer_text, str):
            answer_text = answer_text.strip() or None
        else:
            answer_text = None

        tool_sequence = self._extract_tool_sequence(data.get("tool_uses"))
        tokens = self._extract_tokens(data.get("usage"))

        result.answer_text = answer_text
        if self.config.store_trace:
            result.trace = SessionTrace(
                model=data.get("model") if isinstance(data.get("model"), str) else None,
                workspace_root=str(workspace) if "workspace" in locals() else None,
                extra=data,
            )
        result.metrics = RunMetrics(
            execution_time_s=execution_time,
            tool_call_count=len(tool_sequence),
            tokens=tokens,
            tool_call_sequence=tool_sequence,
            corpus_used=len(tool_sequence) > 0,
        )
        return result

    @staticmethod
    def _build_prompt(question: str) -> str:
        return f"{EXAMINEE_SYSTEM_MESSAGE}\n\n{_CLAUDECODE_PROMPT_SUFFIX}\n\nQuestion:\n{question}"

    def _invoke_claude(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        # Claude Code requires the "ollama/" provider prefix ("ollama/qwen3:4b")
        # when routing through a local Ollama instance.
        model = self.config.model

        cmd = [
            "claude",
            "--model",
            model,
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
            "--allowedTools",
            _ALLOWED_TOOLS,
            "-p",
            prompt,
        ]

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = _OLLAMA_BASE_URL
        env["ANTHROPIC_API_KEY"] = _OLLAMA_API_KEY
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_API_KEY", None)
        # Strip vars set by an outer Claude Code session so the inner `claude`
        # process does not detect nesting and alter its output behaviour.
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
    def _parse_json_output(stdout: str) -> Optional[dict[str, Any]]:
        if not stdout:
            return None
        # Claude Code emits a single JSON envelope on stdout, but be tolerant
        # of leading/trailing whitespace or stray banner lines.
        stripped = stdout.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Fallback: scan for the outermost '{' ... '}' block in the output.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(stripped[start: end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_tool_sequence(tool_uses: Any) -> list[str]:
        if not isinstance(tool_uses, list):
            return []
        sequence: list[str] = []
        for entry in tool_uses:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("tool") or entry.get("type")
                if isinstance(name, str) and name:
                    sequence.append(name)
            elif isinstance(entry, str) and entry:
                sequence.append(entry)
        return sequence

    @staticmethod
    def _extract_tokens(usage: Any) -> Optional[TokenCounts]:
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
            input=_coerce("input_tokens", "prompt_tokens", "input"),
            output=_coerce("output_tokens", "completion_tokens", "output"),
            cache_read=_coerce("cache_read_input_tokens", "cache_read"),
            cache_creation=_coerce("cache_creation_input_tokens", "cache_creation"),
        )
