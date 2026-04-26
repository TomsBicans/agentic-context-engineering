from __future__ import annotations

import argparse


def run_analyze(args: argparse.Namespace) -> None:
    from result_processor.analysis.pipeline import analyze_directory

    analyze_directory(
        experiment_results_dir=args.experiment_results_dir,
        output_dir=args.output_dir,
        path_to_corpora=args.path_to_corpora,
        examiner_model=args.examiner_model,
        num_ctx=args.num_ctx,
        input_files=args.input_files,
        resume=args.resume,
    )
