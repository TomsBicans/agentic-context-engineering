import subprocess
import time
from pathlib import Path

from experiment_runner.models.metrics import RunMetrics
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult
from experiment_runner.runners.base import BaseRunner


# Per-question wall-clock budget for the AnythingLLM CLI subprocess.
_TIMEOUT_SECONDS = 180



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

    def run(self, question: Question) -> RunResult:
        result = self._base_result(question)
        workspace = self.config.corpus.value
        t_start = time.perf_counter()

        try:
            completed = self._invoke_any(question.question, workspace)
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
