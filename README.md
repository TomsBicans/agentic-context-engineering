# agentic-context-engineering

# Software requirements:

- Python 3.10+
- uv, high performance python package manager (https://github.com/astral-sh/uv)
- Pandoc CLI tool (https://github.com/jgm/pandoc)
- Ollama (https://github.com/ollama/ollama/releases)
    - download a model with thinking and tool support `ollama pull {model_name}` (https://ollama.com/)
- Docker Desktop
    - for running the anythingLLM system (one of the baseline systems)
- Google Chrome for Plotly/Kaleido static PDF chart export
    - install with `uv run --package result-processor plotly_get_chrome`, or install Chrome manually for your operating
      system

# Setup guide

1. Clone this repository
2. create a python virtual environment using `uv` tool

```
uv sync --all-packages --all-groups
```

3. Acquire the core datasets

```
make corpus_scraper_solar_system
make corpus_scraper_oblivion
make corpus_scraper_scipy
```

4. Launch the agent sample program

```
make agent_test
```

# Architecture diagrams

High-level PlantUML diagrams for the thesis and project review are stored in `docs/diagrams`.
Generate the SVG versions with:

```
make diagrams
```
