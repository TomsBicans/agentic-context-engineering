from __future__ import annotations

import argparse
import sys

from experiment_runner.models.enums import AutomationLevel, Corpus, SystemName


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--questions-file",
        required=True,
        dest="questions_file",
        help="Path to a question-set JSON file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Directory where RunResult JSONL files are written",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        choices=[c.value for c in Corpus],
        help="Corpus identifier",
    )
    parser.add_argument(
        "--model",
        default="qwen3:4b",
        help="LLM model name used by the system under test",
    )
    parser.add_argument(
        "--question-ids",
        dest="question_ids",
        nargs="*",
        default=None,
        help="Optional subset of question IDs to run (runs all if omitted)",
    )
    parser.add_argument(
        "--no-trace",
        dest="no_trace",
        action="store_true",
        default=False,
        help="Skip storing the full session trace to reduce output size",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Validate configuration and print plan without executing any runs",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="experiment-runner",
        description="Baseline experiment runner for agentic context engineering evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute experiment runs for a system")
    _add_common_options(run_parser)
    run_parser.add_argument(
        "--system",
        required=True,
        choices=[s.value for s in SystemName],
        help="System under test",
    )
    run_parser.add_argument(
        "--path-to-corpora",
        dest="path_to_corpora",
        default=None,
        help="Root corpora directory (required for locally-run systems such as ace)",
    )
    run_parser.add_argument(
        "--automation-level",
        dest="automation_level",
        choices=[a.value for a in AutomationLevel],
        default=AutomationLevel.FULL.value,
        help="How much of the run is automated",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "run":
        from experiment_runner.commands.run import run_experiment
        try:
            run_experiment(args)
        except (ValueError, OSError) as exc:
            sys.stderr.write(f"error: {exc}\n")
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
