from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from experiment_runner.corpus_isolation import isolated_corpus
from experiment_runner.models.config import RunConfig
from experiment_runner.models.enums import AutomationLevel, Corpus, SystemName
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult
from experiment_runner.runners.registry import get_runner


RESULT_PATH_PREFIX = "results → "

_ISOLATED_CORPUS_SYSTEMS: frozenset[SystemName] = frozenset({
    SystemName.ACE,
    SystemName.CLAUDE_CODE_LOCAL,
    SystemName.CHATGPT_CODEX,
    SystemName.CLAWCODE,
})


def load_questions(path: str, ids: list[str] | None) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = [Question(**q) for q in raw]
    if not ids:
        return questions
    id_set = set(ids)
    filtered = [q for q in questions if q.id in id_set]
    missing = id_set - {q.id for q in filtered}
    if missing:
        raise ValueError(f"Question IDs not found in {path}: {sorted(missing)}")
    return filtered


def _output_path(output_dir: str, config: RunConfig) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = (
        f"{config.system.value}__{config.corpus.value}__{ts}"
        f"__{uuid4().hex[:8]}.jsonl"
    )
    return Path(output_dir) / filename


def _uses_isolated_corpus(config: RunConfig) -> bool:
    return config.system in _ISOLATED_CORPUS_SYSTEMS and config.path_to_corpora is not None


def _run_one_question(config: RunConfig, question: Question) -> RunResult:
    runner = get_runner(config)
    runner.setup()
    try:
        return runner.run(question)
    finally:
        runner.teardown()


def _run_one_question_with_isolated_corpus(config: RunConfig, question: Question) -> RunResult:
    if config.path_to_corpora is None:
        raise ValueError("isolated corpus requires path_to_corpora to be set")
    with isolated_corpus(Path(config.path_to_corpora)) as (prepared_corpus_path, snapshot):
        isolated_config = config.model_copy(update={"path_to_corpora": prepared_corpus_path})
        result = _run_one_question(isolated_config, question)
    result.corpus_snapshot = snapshot
    return result


def run_experiment(args: argparse.Namespace) -> None:
    inference_config: dict = {"num_ctx": args.num_ctx}

    config = RunConfig(
        system=SystemName(args.system),
        corpus=Corpus(args.corpus),
        model=args.model,
        automation_level=AutomationLevel(args.automation_level),
        reasoning_enabled=args.reasoning_enabled,
        store_trace=not args.no_trace,
        path_to_corpora=Path(args.path_to_corpora) if args.path_to_corpora else None,
        inference_config=inference_config,
    )

    questions = load_questions(args.questions_file, args.question_ids)
    if not questions:
        raise ValueError("No questions to run after filtering")

    if args.dry_run:
        sys.stdout.write(
            f"[dry-run] system={config.system.value}  corpus={config.corpus.value}"
            f"  model={config.model}  questions={len(questions)}\n"
        )
        for q in questions:
            sys.stdout.write(f"  [{q.id}] {q.question}\n")
        return

    out_path = _output_path(args.output_dir, config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(questions)

    if _uses_isolated_corpus(config):
        with out_path.open("x", encoding="utf-8") as f:
            for i, question in enumerate(questions, 1):
                sys.stderr.write(f"[{i}/{total}] {question.id}: {question.question[:72]}\n")
                result = _run_one_question_with_isolated_corpus(config, question)
                f.write(result.model_dump_json() + "\n")
                f.flush()
    else:
        runner = get_runner(config)
        runner.setup()
        try:
            with out_path.open("x", encoding="utf-8") as f:
                for i, question in enumerate(questions, 1):
                    sys.stderr.write(f"[{i}/{total}] {question.id}: {question.question[:72]}\n")
                    result = runner.run(question)
                    f.write(result.model_dump_json() + "\n")
                    f.flush()
        finally:
            runner.teardown()

    sys.stderr.write(f"{RESULT_PATH_PREFIX}{out_path}\n")
