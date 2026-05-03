from __future__ import annotations

import argparse
import sys


def _add_analyze_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-results-dir",
        dest="experiment_results_dir",
        default="./data/experiment_results",
        help="Directory containing RunResult JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="./data/analysis_results",
        help="Directory where AnalysisResult JSONL files are written",
    )
    parser.add_argument(
        "--path-to-corpora",
        dest="path_to_corpora",
        required=True,
        help="Root corpora directory used to resolve cited file paths",
    )
    parser.add_argument(
        "--examiner-model",
        dest="examiner_model",
        default="qwen3:4b",
        help="Ollama model name used by the A2 examiner",
    )
    parser.add_argument(
        "--num-ctx",
        dest="num_ctx",
        type=int,
        default=8192,
        help="Context window size for the examiner model",
    )
    parser.add_argument(
        "--input-files",
        dest="input_files",
        nargs="*",
        default=None,
        help="Specific JSONL files to analyse (default: all under experiment-results-dir)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Re-analyse runs that already have an AnalysisResult on disk",
    )


def _add_visualize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-results-dir",
        dest="experiment_results_dir",
        default="./data/experiment_results",
        help="Directory containing RunResult JSONL files",
    )
    parser.add_argument(
        "--analysis-results-dir",
        dest="analysis_results_dir",
        default="./data/analysis_results",
        help="Directory containing AnalysisResult JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="./data/figures",
        help="Directory where Plotly figures and LaTeX tables are written",
    )
    parser.add_argument(
        "--formats",
        dest="formats",
        nargs="+",
        default=["png", "html"],
        choices=["png", "html", "pdf", "svg"],
        help="Image formats to export Plotly figures in",
    )


def _add_dashboard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--experiment-results-dir",
        dest="experiment_results_dir",
        default="./data/experiment_results",
    )
    parser.add_argument(
        "--analysis-results-dir",
        dest="analysis_results_dir",
        default="./data/analysis_results",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="result-processor",
        description="Post-experiment analysis: A2 examiner, charts, LaTeX tables, dashboard",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Run the A2 examiner over RunResult JSONL files",
    )
    _add_analyze_args(analyze)

    visualize = subparsers.add_parser(
        "visualize",
        help="Generate Plotly figures and LaTeX tables from analysis results",
    )
    _add_visualize_args(visualize)

    dashboard = subparsers.add_parser(
        "dashboard",
        help="Launch the Streamlit dashboard",
    )
    _add_dashboard_args(dashboard)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        if args.command == "analyze":
            from result_processor.commands.analyze import run_analyze
            run_analyze(args)
        elif args.command == "visualize":
            from result_processor.commands.visualize import run_visualize
            run_visualize(args)
        elif args.command == "dashboard":
            from result_processor.commands.dashboard import run_dashboard
            run_dashboard(args)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
