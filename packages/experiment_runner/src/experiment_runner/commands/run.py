from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiment_runner.models.config import RunConfig
from experiment_runner.models.enums import AutomationLevel, Corpus, SystemName
from experiment_runner.models.question import Question
from experiment_runner.runners.registry import get_runner


def _load_questions(path: str, ids: list[str] | None) -> list[Question]:
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
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{config.system.value}__{config.corpus.value}__{ts}.jsonl"
    return Path(output_dir) / filename


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

    questions = _load_questions(args.questions_file, args.question_ids)
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

    runner = get_runner(config)
    total = len(questions)

    runner.setup()
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for i, question in enumerate(questions, 1):
                sys.stderr.write(f"[{i}/{total}] {question.id}: {question.question[:72]}\n")
                result = runner.run(question)
                f.write(result.model_dump_json() + "\n")
                f.flush()
    finally:
        runner.teardown()

    sys.stderr.write(f"results → {out_path}\n")
