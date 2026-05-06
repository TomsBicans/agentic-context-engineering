from __future__ import annotations

import json
import subprocess
from pathlib import Path

from experiment_runner.models.config import RunConfig
from experiment_runner.models.enums import AutomationLevel, Corpus, SystemName
from experiment_runner.models.question import Question
from experiment_runner.runners.baseline.gptcodexlocal import GptCodexLocalRunner
from agent.prompts import EXAMINEE_SYSTEM_MESSAGE


def _config(corpus_path: Path) -> RunConfig:
    return RunConfig(
        system=SystemName.CHATGPT_CODEX,
        corpus=Corpus.SOLAR_SYSTEM_WIKI,
        model="qwen3:4b",
        automation_level=AutomationLevel.FULL,
        path_to_corpora=corpus_path,
        inference_config={"num_ctx": 8192},
    )


def _jsonl(events: list[dict]) -> str:
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _question() -> Question:
    return Question(
        id="ss_L2_003",
        corpus="solar_system_wiki",
        question="What is Mars?",
        level=2,
        expected_facts=[],
    )


def test_codex_prompt_uses_shared_examinee_prompt() -> None:
    prompt = GptCodexLocalRunner._build_prompt("What is Mars?")

    assert prompt == f"{EXAMINEE_SYSTEM_MESSAGE}\n\nQuestion:\nWhat is Mars?"
    assert "Codex runtime notes" not in prompt


def test_codex_extracts_command_events_as_tool_steps() -> None:
    events = [
        {"type": "item.completed", "item": {"type": "reasoning", "text": "Need corpus evidence."}},
        {
            "type": "item.completed",
            "item": {"type": "local_shell_call", "command": "grep -R Mars ./corpus"},
        },
        {
            "type": "item.completed",
            "item": {"type": "local_shell_call_output", "output": "corpus/text/mars.md:Mars is..."},
        },
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Answer."}},
    ]

    steps = GptCodexLocalRunner._extract_steps(events)

    assert [step.type for step in steps] == ["reasoning", "tool_call", "tool_result", "agent_message"]
    assert GptCodexLocalRunner._extract_tool_sequence(events) == ["grep -R Mars ./corpus"]


def test_codex_run_marks_corpus_used_only_when_command_event_is_present(monkeypatch, tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    runner = GptCodexLocalRunner(_config(corpus))

    events = [
        {
            "type": "item.completed",
            "item": {"type": "local_shell_call", "command": "grep -R Mars ./corpus"},
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "[Mars is a planet.] [file:text/mars.md, lines:0-1]"},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 12}},
    ]

    def fake_invoke(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        assert (workspace / "corpus").is_symlink()
        return subprocess.CompletedProcess(args=["codex"], returncode=0, stdout=_jsonl(events), stderr="")

    monkeypatch.setattr(GptCodexLocalRunner, "_invoke_codex", fake_invoke)

    result = runner.run(_question())

    assert result.answer_error is None
    assert result.metrics.corpus_used is True
    assert result.metrics.tool_call_count == 1
    assert result.metrics.tool_call_sequence == ["grep -R Mars ./corpus"]


def test_codex_run_does_not_mark_corpus_used_without_command_events(monkeypatch, tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    runner = GptCodexLocalRunner(_config(corpus))

    events = [
        {"type": "item.completed", "item": {"type": "reasoning", "text": "No lookup."}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "No corpus data found."}},
    ]

    def fake_invoke(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["codex"], returncode=0, stdout=_jsonl(events), stderr="")

    monkeypatch.setattr(GptCodexLocalRunner, "_invoke_codex", fake_invoke)

    result = runner.run(_question())

    assert result.answer_error is None
    assert result.metrics.corpus_used is False
    assert result.metrics.tool_call_count == 0
    assert result.metrics.tool_call_sequence == []
