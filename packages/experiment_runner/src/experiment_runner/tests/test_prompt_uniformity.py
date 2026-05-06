from __future__ import annotations

from agent.prompts import EXAMINEE_SYSTEM_MESSAGE
from experiment_runner.runners.baseline.anythingllm import AnythingLLMRunner
from experiment_runner.runners.baseline.claudecodelocal import ClaudeCodeLocalRunner
from experiment_runner.runners.baseline.clawcode import ClawCodeRunner
from experiment_runner.runners.baseline.gptcodexlocal import GptCodexLocalRunner
from experiment_runner.runners.baseline.openclaw import OpenClawRunner


def test_baseline_runners_use_uniform_examinee_prompt() -> None:
    expected = f"{EXAMINEE_SYSTEM_MESSAGE}\n\nQuestion:\nWhat is Mars?"
    prompts = [
        AnythingLLMRunner._build_prompt("What is Mars?"),
        ClaudeCodeLocalRunner._build_prompt("What is Mars?"),
        ClawCodeRunner._build_prompt("What is Mars?"),
        GptCodexLocalRunner._build_prompt("What is Mars?"),
        OpenClawRunner._build_prompt("What is Mars?"),
    ]

    assert prompts == [expected] * len(prompts)


def test_baseline_prompts_do_not_include_integration_runtime_notes() -> None:
    prompt = "\n\n".join([
        AnythingLLMRunner._build_prompt("What is Mars?"),
        ClaudeCodeLocalRunner._build_prompt("What is Mars?"),
        ClawCodeRunner._build_prompt("What is Mars?"),
        GptCodexLocalRunner._build_prompt("What is Mars?"),
        OpenClawRunner._build_prompt("What is Mars?"),
    ])

    assert "AnythingLLM runtime notes" not in prompt
    assert "Claude Code runtime notes" not in prompt
    assert "ClawCode runtime notes" not in prompt
    assert "Codex runtime notes" not in prompt
    assert "OpenClaw runtime notes" not in prompt
