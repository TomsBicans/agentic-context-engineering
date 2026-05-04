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
    run_parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        dest="num_ctx",
        help="Context window size passed to the local inference engine",
    )
    run_parser.add_argument(
        "--reasoning-enabled",
        dest="reasoning_enabled",
        action="store_true",
        default=False,
        help="Enable chain-of-thought / reasoning mode for the model",
    )

    suite = subparsers.add_parser("suite", help="Plan, run, and resume experiment suites")
    suite_subparsers = suite.add_subparsers(dest="suite_command", required=True)

    suite_plan = suite_subparsers.add_parser("plan", help="Print the expanded suite task list")
    suite_plan.add_argument("--config", required=True, help="Experiment suite config JSON")
    suite_plan.add_argument("--json", action="store_true", help="Print planned tasks as JSON")

    suite_run = suite_subparsers.add_parser("run", help="Execute or resume a suite")
    suite_run.add_argument("--config", required=True, help="Experiment suite config JSON")
    suite_run.add_argument("--state", default=None, help="Suite state JSON path")

    suite_status = suite_subparsers.add_parser("status", help="Print suite progress state")
    suite_status.add_argument("--state", required=True, help="Suite state JSON path")

    suite_cancel = suite_subparsers.add_parser("cancel", help="Request cooperative suite cancellation")
    suite_cancel.add_argument("--state", required=True, help="Suite state JSON path")

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
    elif args.command == "suite":
        from experiment_runner.commands.suite import (
            run_suite_cancel,
            run_suite_plan,
            run_suite_run,
            run_suite_status,
        )
        try:
            if args.suite_command == "plan":
                run_suite_plan(args)
            elif args.suite_command == "run":
                run_suite_run(args)
            elif args.suite_command == "status":
                run_suite_status(args)
            elif args.suite_command == "cancel":
                run_suite_cancel(args)
        except (ValueError, OSError) as exc:
            sys.stderr.write(f"error: {exc}\n")
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
