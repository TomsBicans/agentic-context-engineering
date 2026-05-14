from __future__ import annotations

from agent.prompts import EXAMINEE_SYSTEM_MESSAGE
from experiment_runner.runners.baseline.anythingllm import AnythingLLMRunner, build_anythingllm_prompt_command
from experiment_runner.runners.baseline.claudecodelocal import (
    ClaudeCodeLocalRunner,
    build_claude_command,
    claude_environment_overrides,
)
from experiment_runner.runners.baseline.clawcode import (
    ClawCodeRunner,
    build_claw_command,
    claw_environment_overrides,
)
from experiment_runner.runners.baseline.gptcodexlocal import GptCodexLocalRunner, build_codex_command
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


def test_baseline_cli_command_builders() -> None:
    prompt = "Prompt"

    assert build_codex_command(model="qwen3:4b", prompt=prompt) == [
        "codex", "exec",
        "--oss",
        "--local-provider", "ollama",
        "-m", "qwen3:4b",
        "--sandbox", "read-only",
        "--skip-git-repo-check",
        "--json",
        prompt,
    ]
    assert build_claw_command(model="qwen3:4b", prompt=prompt) == [
        "claw",
        "--model", "openai/qwen3:4b",
        "--output-format", "json",
        "--permission-mode", "workspace-write",
        "--allowedTools", "read_file,glob_search,grep_search",
        prompt,
    ]
    assert build_claude_command(model="qwen3:4b", prompt=prompt) == [
        "claude",
        "--model", "qwen3:4b",
        "--output-format", "json",
        "--permission-mode", "bypassPermissions",
        "--allowedTools", "read_file,glob_search,grep_search",
        "-p",
        prompt,
    ]
    assert build_anythingllm_prompt_command(prompt=prompt, workspace="solar_system_wiki") == [
        "any",
        "prompt",
        prompt,
        "--workspace",
        "solar_system_wiki",
        "--nt",
        "--no-stream",
    ]


def test_baseline_cli_environment_overrides() -> None:
    assert claw_environment_overrides() == {
        "OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
        "OPENAI_API_KEY": "ollama",
    }
    assert claude_environment_overrides() == {
        "ANTHROPIC_BASE_URL": "http://localhost:11434",
        "ANTHROPIC_API_KEY": "ollama",
    }
