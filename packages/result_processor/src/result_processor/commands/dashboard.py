from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import streamlit.web.cli as stcli


def run_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard.

    Streamlit is invoked in-process via its CLI module so that the standard
    `streamlit run` argv parsing applies (theme, port, etc).
    """
    app_path = Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py"

    os.environ["RP_EXPERIMENT_RESULTS_DIR"] = str(Path(args.experiment_results_dir).resolve())
    os.environ["RP_ANALYSIS_RESULTS_DIR"] = str(Path(args.analysis_results_dir).resolve())

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
    ]
    sys.exit(stcli.main())
