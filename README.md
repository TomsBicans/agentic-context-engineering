# agentic-context-engineering

# Software requirements:
- Python 3.10+
- uv, high performance python package manager (https://github.com/astral-sh/uv)
- Ollama (https://github.com/ollama/ollama/releases)
  - download a model with thinking and tool support `ollama pull {model_name}` (https://ollama.com/)

# Setup guide
1. Clone this repository
2. create python virtual environment
```
python -m venv .venv
```
3. Activate python virtual environment
On Windows:
```
.venv/Scripts/activate
```
On Linux (todo: need to validate if this is the real command):
```
.venv/scripts/activate
```

4. Install dependencies
```
python -m pip install -r requirements.txt
```
5. Launch the agent program
```
make agent_test
```