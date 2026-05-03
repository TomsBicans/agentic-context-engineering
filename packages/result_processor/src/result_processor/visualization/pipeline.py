"""Orchestrator for the visualize command."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rich.console import Console

from result_processor.visualization.loader import build_dataframe
from result_processor.visualization.plots import ALL_PLOTS
from result_processor.visualization.tables import ALL_TABLES


def visualize_results(
    experiment_results_dir: str,
    analysis_results_dir: str,
    output_dir: str,
    formats: Iterable[str] = ("png", "html"),
) -> None:
    console = Console()

    in_dir = Path(experiment_results_dir).resolve()
    analysis_dir = Path(analysis_results_dir).resolve()
    out_dir = Path(output_dir).resolve()

    if not in_dir.is_dir():
        raise ValueError(f"experiment_results_dir not found: {in_dir}")

    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(in_dir, analysis_dir)
    if df.empty:
        console.print("[yellow]No data to visualize — experiment_results is empty.[/yellow]")
        return

    console.print(f"[bold]Loaded {len(df)} runs[/bold]")

    for name, builder in ALL_PLOTS.items():
        try:
            fig = builder(df)
        except Exception as exc:
            console.print(f"  [red]plot {name} failed: {exc}[/red]")
            continue
        _export_figure(fig, plots_dir / name, formats, console)

    for name, builder in ALL_TABLES.items():
        try:
            latex = builder(df)
        except Exception as exc:
            console.print(f"  [red]table {name} failed: {exc}[/red]")
            continue
        if not latex:
            continue
        path = tables_dir / f"{name}.tex"
        path.write_text(latex, encoding="utf-8")
        console.print(f"  [dim]→[/dim] {path.relative_to(out_dir.parent)}")

    console.print("[bold green]Visualization complete.[/bold green]")


def _export_figure(fig, base_path: Path, formats: Iterable[str], console: Console) -> None:
    for fmt in formats:
        path = base_path.with_suffix(f".{fmt}")
        try:
            if fmt == "html":
                fig.write_html(str(path), include_plotlyjs="cdn")
            else:
                # PNG/PDF/SVG require kaleido; fall back gracefully if missing.
                fig.write_image(str(path))
        except Exception as exc:
            console.print(f"  [yellow]skipped {path.name}: {exc}[/yellow]")
            continue
        console.print(f"  [dim]→[/dim] {path.name}")
