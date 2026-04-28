import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from experiment_runner.models.metrics import RunMetrics, TokenCounts
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult
from experiment_runner.runners.base import BaseRunner


# Per-question wall-clock budget for the ClawCode subprocess.
_TIMEOUT_SECONDS = 180

# Tools the EXAMINEE is allowed to invoke. Restricted to read/search-only so
# ClawCode cannot mutate the workspace even though it runs in workspace-write mode.
_ALLOWED_TOOLS = "read_file,glob_search,grep_search,list_files"

# Ollama-compatible OpenAI shim used by ClawCode for local inference.
_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
_OLLAMA_API_KEY = "ollama"


class ClawCodeRunner(BaseRunner):
    """Runner for the ClawCode CLI baseline.

    Each question is executed inside a fresh temporary workspace so the real
    corpus directory is never written to and concurrent runs cannot interfere
    with each other. The corpus is exposed to ClawCode via a single symlink
    inside the temp workspace, which keeps file paths stable in the model's
    context while still letting the framework operate in `workspace-write` mode.

    The runner shells out to `claw` with `--output-format json`, captures
    stdout, and maps the JSON envelope onto the framework's RunResult /
    RunMetrics types. Any subprocess, parsing, or I/O failure is converted into
    a populated `answer_error` instead of a raised exception.
    """

    def run(self, question: Question) -> RunResult:
        result = self._base_result(question)

        if self.config.path_to_corpora is None:
            result.answer_error = "ClawCodeRunner requires path_to_corpora in RunConfig"
            result.metrics = RunMetrics()
            return result

        corpus_root = Path(self.config.path_to_corpora).resolve()
        if not corpus_root.exists():
            result.answer_error = f"Corpus path does not exist: {corpus_root}"
            result.metrics = RunMetrics()
            return result

        t_start = time.perf_counter()

        try:
            with tempfile.TemporaryDirectory(prefix="clawcode_run_") as tmp:
                workspace = Path(tmp)
                # Mirror the corpus into the sandbox via a symlink so ClawCode
                # sees the content under a predictable, sandboxed path.
                corpus_link = workspace / "corpus"
                corpus_link.symlink_to(corpus_root, target_is_directory=True)

                completed = self._invoke_claw(question.question, workspace)

                if completed.returncode != 0:
                    result.answer_error = (
                        f"claw exited with code {completed.returncode}: "
                        f"{(completed.stderr or '').strip()[:500]}"
                    )
                    result.metrics = RunMetrics(
                        execution_time_s=time.perf_counter() - t_start,
                    )
                    return result

                data = self._parse_json_output(completed.stdout)
        except subprocess.TimeoutExpired:
            result.answer_error = f"claw timed out after {_TIMEOUT_SECONDS}s"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except FileNotFoundError:
            result.answer_error = "claw executable not found on PATH"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except Exception as exc:
            result.answer_error = f"ClawCode runner failed: {exc}"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result

        execution_time = time.perf_counter() - t_start

        if data is None:
            result.answer_error = "Failed to parse JSON output from claw"
            result.metrics = RunMetrics(execution_time_s=execution_time)
            return result

        answer_text = data.get("message")
        if isinstance(answer_text, str):
            answer_text = answer_text.strip() or None
        else:
            answer_text = None

        tool_sequence = self._extract_tool_sequence(data.get("tool_uses"))
        tokens = self._extract_tokens(data.get("usage"))

        result.answer_text = answer_text
        result.metrics = RunMetrics(
            execution_time_s=execution_time,
            tool_call_count=len(tool_sequence),
            tokens=tokens,
            tool_call_sequence=tool_sequence,
            corpus_used=len(tool_sequence) > 0,
        )
        return result

    def _invoke_claw(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        # ClawCode requires a provider-prefixed model identifier ("openai/qwen3:8b").
        # Bare Ollama model names ("qwen3:8b") are rejected as invalid model syntax.
        model = self.config.model
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        cmd = [
            "claw",
            "--model",
            model,
            "--output-format",
            "json",
            "--permission-mode",
            "workspace-write",
            "--allowedTools",
            _ALLOWED_TOOLS,
            prompt,
        ]

        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = _OLLAMA_BASE_URL
        env["OPENAI_API_KEY"] = _OLLAMA_API_KEY
        # Prevent ClawCode from falling back to cloud providers if these keys
        # happen to be set in the parent environment.
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("DASHSCOPE_API_KEY", None)

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
        # ClawCode is documented to emit a single JSON envelope on stdout, but
        # be tolerant of leading/trailing whitespace or stray banner lines.
        stripped = stdout.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Fallback: scan for the last '{' ... '}' block in the output.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(stripped[start : end + 1])
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
