from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_runner import corpus_isolation
from experiment_runner.commands import run as run_command
from experiment_runner.corpus_isolation import capture_tree, isolated_corpus
from experiment_runner.models.enums import AutomationLevel, Corpus, SystemName
from experiment_runner.models.metrics import RunMetrics
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult


def _write_source_corpus(path: Path) -> None:
    (path / "text").mkdir(parents=True)
    (path / "text" / "planets.md").write_text("Earth\nMars\n", encoding="utf-8")
    (path / ".claw").mkdir()
    (path / ".claw" / "state.json").write_text("{}", encoding="utf-8")
    (path / "AGENTS.md").write_text("tool junk", encoding="utf-8")
    (path / "config.json").write_text(json.dumps({"name": "solar"}), encoding="utf-8")
    (path / "manifest.jsonl").write_text("{}\n{}\n", encoding="utf-8")


def _write_questions(path: Path, count: int = 2) -> None:
    rows = [
        {
            "id": f"ss_L1_00{i}",
            "corpus": "solar_system_wiki",
            "level": 1,
            "question": f"Question {i}?",
            "expected_facts": [],
        }
        for i in range(1, count + 1)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def _args(tmp_path: Path, source_corpus: Path, system: SystemName) -> argparse.Namespace:
    questions_file = tmp_path / "questions.json"
    _write_questions(questions_file)
    return argparse.Namespace(
        system=system.value,
        corpus=Corpus.SOLAR_SYSTEM_WIKI.value,
        questions_file=str(questions_file),
        output_dir=str(tmp_path / "results"),
        model="qwen3:4b",
        num_ctx=8192,
        path_to_corpora=str(source_corpus),
        automation_level=AutomationLevel.FULL.value,
        question_ids=None,
        reasoning_enabled=False,
        no_trace=True,
        dry_run=False,
    )


def test_isolated_corpus_copies_text_and_metadata_without_tool_artifacts(tmp_path) -> None:
    source = tmp_path / "solar_system_wiki"
    _write_source_corpus(source)

    with isolated_corpus(source) as (prepared, snapshot):
        assert (prepared / "text" / "planets.md").is_file()
        assert not (prepared / ".claw").exists()
        assert not (prepared / "AGENTS.md").exists()
        assert (prepared / "config.json").is_file()
        assert (prepared / "manifest.jsonl").is_file()

    assert snapshot.config_json == {"name": "solar"}
    assert snapshot.manifest_entry_count == 2
    assert snapshot.file_count == 3
    assert "text" in snapshot.copied_paths
    assert "planets.md" in (snapshot.pre_run_tree or "")
    assert "planets.md" in (snapshot.post_run_tree or "")
    assert not Path(snapshot.prepared_corpus_path).exists()


def test_capture_tree_uses_python_fallback_when_tree_is_missing(monkeypatch, tmp_path) -> None:
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "doc.md").write_text("content", encoding="utf-8")

    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("tree")

    monkeypatch.setattr(corpus_isolation.subprocess, "run", raise_missing)

    output = capture_tree(tmp_path)

    assert "text/" in output
    assert "doc.md" in output


def test_isolated_corpus_uses_configured_temp_root_when_usable(monkeypatch, tmp_path) -> None:
    source = tmp_path / "solar_system_wiki"
    temp_root = tmp_path / "ram_tmp"
    _write_source_corpus(source)
    monkeypatch.setenv("EXPERIMENT_RUNNER_CORPUS_TMP_DIR", str(temp_root))

    with isolated_corpus(source) as (prepared, snapshot):
        assert prepared.is_relative_to(temp_root)
        assert snapshot.temp_root_path == str(temp_root)
        assert (prepared / "text" / "planets.md").is_file()


def test_select_temp_root_auto_creates_named_directory(monkeypatch, tmp_path) -> None:
    ram_root = tmp_path / "dev_shm"
    disk_root = tmp_path / "tmp"
    monkeypatch.setattr(corpus_isolation, "_AUTO_TEMP_ROOTS", (ram_root, disk_root))

    selected = corpus_isolation._select_temp_root(1024)

    assert selected == ram_root / "ace_corpora"
    assert selected.is_dir()


def test_run_experiment_uses_clean_isolated_corpus_per_question(monkeypatch, tmp_path) -> None:
    source = tmp_path / "solar_system_wiki"
    _write_source_corpus(source)
    seen_paths: list[Path] = []

    class FakeRunner:
        def __init__(self, config) -> None:
            self.config = config

        def setup(self) -> None:
            pass

        def teardown(self) -> None:
            pass

        def run(self, question: Question) -> RunResult:
            corpus_path = Path(self.config.path_to_corpora)
            seen_paths.append(corpus_path)
            assert (corpus_path / "text" / "planets.md").is_file()
            assert not (corpus_path / ".claw").exists()
            return RunResult(
                system_name=self.config.system,
                automation_level=self.config.automation_level,
                corpus=self.config.corpus,
                question_id=question.id,
                question_text=question.question,
                model=self.config.model,
                metrics=RunMetrics(corpus_used=True),
                answer_text="answer",
            )

    monkeypatch.setattr(run_command, "get_runner", lambda config: FakeRunner(config))

    run_command.run_experiment(_args(tmp_path, source, SystemName.ACE))

    assert len(seen_paths) == 2
    assert len(set(seen_paths)) == 2
    assert all(not path.exists() for path in seen_paths)
    result_file = next((tmp_path / "results").glob("*.jsonl"))
    rows = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines()]
    assert all(row["corpus_snapshot"]["enabled"] for row in rows)
    assert all(row["corpus_snapshot"]["file_count"] == 3 for row in rows)
    assert all(".claw" not in row["corpus_snapshot"]["pre_run_tree"] for row in rows)


def test_run_experiment_does_not_isolate_anythingllm(monkeypatch, tmp_path) -> None:
    source = tmp_path / "solar_system_wiki"
    _write_source_corpus(source)
    setup_count = 0
    seen_paths: list[Path] = []

    class FakeRunner:
        def __init__(self, config) -> None:
            self.config = config

        def setup(self) -> None:
            nonlocal setup_count
            setup_count += 1

        def teardown(self) -> None:
            pass

        def run(self, question: Question) -> RunResult:
            seen_paths.append(Path(self.config.path_to_corpora))
            return RunResult(
                system_name=self.config.system,
                automation_level=self.config.automation_level,
                corpus=self.config.corpus,
                question_id=question.id,
                question_text=question.question,
                model=self.config.model,
                metrics=RunMetrics(corpus_used=True),
                answer_text="answer",
            )

    monkeypatch.setattr(run_command, "get_runner", lambda config: FakeRunner(config))

    run_command.run_experiment(_args(tmp_path, source, SystemName.ANYTHINGLLM))

    assert setup_count == 1
    assert seen_paths == [source, source]
    result_file = next((tmp_path / "results").glob("*.jsonl"))
    rows = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines()]
    assert all(row["corpus_snapshot"] is None for row in rows)
