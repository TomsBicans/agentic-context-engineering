# agentic-context-engineering

# Software requirements:

- Python 3.10+
- uv, high performance python package manager (https://github.com/astral-sh/uv)
- Ollama (https://github.com/ollama/ollama/releases)
    - download a model with thinking and tool support `ollama pull {model_name}` (https://ollama.com/)

# Setup guide

1. Clone this repository
2. create a python virtual environment using `uv` tool

```
uv sync --all-packages --all-groups
```

3. Launch the agent sample program

```
make agent_test
```