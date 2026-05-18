from __future__ import annotations

import json
import subprocess

from experiment_runner.models.config import RunConfig
from experiment_runner.models.enums import AutomationLevel, Corpus, SystemName
from experiment_runner.models.question import Question
from experiment_runner.runners.baseline.claudecodelocal import ClaudeCodeLocalRunner


def test_claude_json_error_envelope_sets_answer_error(tmp_path, monkeypatch) -> None:
    runner = ClaudeCodeLocalRunner(
        RunConfig(
            system=SystemName.CLAUDE_CODE_LOCAL,
            corpus=Corpus.SOLAR_SYSTEM_WIKI,
            model="qwen3.5:0.8b",
            automation_level=AutomationLevel.FULL,
            path_to_corpora=tmp_path,
            store_trace=True,
        )
    )
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "api_error_status": 404,
        "result": "There's an issue with the selected model (qwen3.5:0.8b).",
    }
    monkeypatch.setattr(
        runner,
        "_invoke_claude",
        lambda _prompt, _workspace: subprocess.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    result = runner.run(
        Question(
            id="ss_L1_001",
            corpus="solar_system_wiki",
            level=1,
            question="What is Mars?",
            expected_facts=[],
        )
    )

    assert result.answer_error == "claude returned error: There's an issue with the selected model (qwen3.5:0.8b)."
    assert result.answer_text is None
    assert result.trace is not None
    assert result.trace.extra["api_error_status"] == 404
