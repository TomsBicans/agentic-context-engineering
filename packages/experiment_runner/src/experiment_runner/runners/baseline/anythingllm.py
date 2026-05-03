import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from agent.prompts import EXAMINEE_SYSTEM_MESSAGE
from experiment_runner.models.metrics import RunMetrics
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult
from experiment_runner.runners.base import BaseRunner


# Per-question wall-clock budget for the AnythingLLM CLI subprocess.
_TIMEOUT_SECONDS = 180

_CONTAINER_NAME = "anythingllm-thesis"
_STORAGE_DIR = Path.home() / "anythingllm-thesis-data"
_READY_URL = "http://localhost:3001/api/ping"
_READY_TIMEOUT_SECONDS = 90

# Replaces the ACE-specific tooling strategy section of EXAMINEE_SYSTEM_MESSAGE,
# which references tools (search(), list_paths(), time_left()) that do not exist
# in AnythingLLM. The citation format and offline constraints are inherited as-is.
_ANYTHINGLLM_PROMPT_SUFFIX = """
AnythingLLM runtime notes
- You are operating inside an AnythingLLM workspace with the corpus pre-loaded.
- Document retrieval is automatic — do not reference or call any external tools.
- Focus on producing a factual, cited answer using only the retrieved context.
- If the retrieved context does not contain enough information, say so explicitly.
""".strip()



class AnythingLLMRunner(BaseRunner):
    """Runner for the AnythingLLM baseline via the official `any` CLI.

    AnythingLLM workspaces are corpus-scoped (one workspace per corpus,
    slug matching the corpus name exactly). The workspace must be created and
    populated with the relevant documents before running experiments — the
    runner does not manage document ingestion.

    Each question is sent on a fresh thread (``--nt``) so that conversation
    history from previous questions cannot leak into the current answer.
    Streaming is disabled (``-S``) so the full response lands on stdout as a
    single clean string rather than an SSE stream.

    Because the `any` CLI exposes no structured output, token counts and
    tool-call sequences are unavailable; only ``execution_time_s`` and
    ``corpus_used`` are populated in ``RunMetrics``.
    """

    def setup(self) -> None:
        self._stop_container()
        self._start_container()
        self._wait_for_ready()

    def teardown(self) -> None:
        self._stop_container()

    def _stop_container(self) -> None:
        subprocess.run(["docker", "stop", _CONTAINER_NAME], capture_output=True, check=False)
        subprocess.run(["docker", "rm", _CONTAINER_NAME], capture_output=True, check=False)

    def _start_container(self) -> None:
        _STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        (_STORAGE_DIR / ".env").touch()
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", _CONTAINER_NAME,
                "-p", "3001:3001",
                "--add-host=host.docker.internal:host-gateway",
                "--cap-add", "SYS_ADMIN",
                "-v", f"{_STORAGE_DIR}:/app/server/storage",
                "-v", f"{_STORAGE_DIR}/.env:/app/server/.env",
                "-e", "STORAGE_DIR=/app/server/storage",
                "-e", "LLM_PROVIDER=ollama",
                "-e", f"OLLAMA_MODEL_PREF={self.config.model}",
                "-e", "OLLAMA_BASE_PATH=http://host.docker.internal:11434",
                "mintplexlabs/anythingllm:latest",
            ],
            check=True,
        )

    def _wait_for_ready(self) -> None:
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(_READY_URL, timeout=2)
                return
            except (urllib.error.URLError, OSError):
                time.sleep(2)
        raise RuntimeError(
            f"AnythingLLM did not become ready within {_READY_TIMEOUT_SECONDS}s"
        )

    def run(self, question: Question) -> RunResult:
        result = self._base_result(question)
        workspace = self.config.corpus.value
        t_start = time.perf_counter()

        try:
            completed = self._invoke_any(self._build_prompt(question.question), workspace)
        except subprocess.TimeoutExpired:
            result.answer_error = f"any timed out after {_TIMEOUT_SECONDS}s"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except FileNotFoundError:
            result.answer_error = "any executable not found on PATH"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result
        except Exception as exc:
            result.answer_error = f"AnythingLLM runner failed: {exc}"
            result.metrics = RunMetrics(execution_time_s=time.perf_counter() - t_start)
            return result

        execution_time = time.perf_counter() - t_start

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()[:500]
            result.answer_error = (
                f"any exited with code {completed.returncode}: {stderr}"
            )
            result.metrics = RunMetrics(execution_time_s=execution_time)
            return result

        answer_text = (completed.stdout or "").strip() or None
        result.answer_text = answer_text
        result.metrics = RunMetrics(
            execution_time_s=execution_time,
            # AnythingLLM queries a pre-loaded document workspace, so the
            # corpus is always consulted when the run succeeds.
            corpus_used=answer_text is not None,
        )
        return result

    @staticmethod
    def _build_prompt(question: str) -> str:
        return f"{EXAMINEE_SYSTEM_MESSAGE}\n\n{_ANYTHINGLLM_PROMPT_SUFFIX}\n\nQuestion:\n{question}"

    def _invoke_any(self, prompt: str, workspace: str) -> subprocess.CompletedProcess:
        cmd = [
            "any",
            "prompt",
            prompt,
            "--workspace",
            workspace,
            "--nt",
            "--no-stream",
        ]

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
