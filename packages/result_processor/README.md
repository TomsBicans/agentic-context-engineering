# result-processor

Post-experiment analysis, visualization, LaTeX table export, and Streamlit dashboard tools.

## Static chart export

The visualization pipeline writes interactive HTML charts and thesis-ready PDF charts. PDF export uses Plotly Kaleido, which requires Google Chrome to be available on the machine.

If PDF export fails with `Kaleido requires Google Chrome to be installed`, install Chrome for Kaleido from the project environment:

```bash
uv run --package result-processor plotly_get_chrome
```

You can also install Google Chrome manually for your operating system. HTML chart export does not require Chrome.
