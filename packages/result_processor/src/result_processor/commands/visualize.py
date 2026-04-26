from __future__ import annotations

import argparse


def run_visualize(args: argparse.Namespace) -> None:
    from result_processor.visualization.pipeline import visualize_results

    visualize_results(
        experiment_results_dir=args.experiment_results_dir,
        analysis_results_dir=args.analysis_results_dir,
        output_dir=args.output_dir,
        formats=args.formats,
    )
