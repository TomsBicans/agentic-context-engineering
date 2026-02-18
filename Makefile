# Setup
install_dependencies:
	uv sync --all-packages --all-groups

run_tests:
	uv run --package agent py -m pytest -q

install_tools:
	uv tool install ./packages/cli
	uv tool install ./packages/agent

model=gemma3:4b # no tool call support on Ollama
model=deepseek-r1:8b # no tool call support on Ollama
model=deepseek-r1:14b # no tool call support on Ollama

# There are some issues for these models, such as them not calling tools.
model=qwen3:8b
model=cogito:8b
model=functiongemma:270m
model=mistral-nemo:12b
model=llama3.1:8b

# These models somewhat work correctly
model=qwen3-vl:4b
model=qwen3-vl:8b
model=qwen3:14b
model=gpt-oss:20b
model=qwen3:4b

q="how many Extreme trans-Neptunian objects are there?"
q="Where can i find the list of minor planets?"
q="What is the Earth's mass?"
q="How many comets are in the solar system?"
q="What is the Sun's mass?"

ctx=2048
ctx=16384
ctx=32768
ctx=6144
ctx=4096
ctx=8192

agent_h:
	uv run --package agent py -m agent.core -h

agent_test:
	uv run --package agent py -m agent.core ${q} --model ${model} --num_ctx ${ctx} --role examinee --path-to-corpora "./corpora/scraped_data/solar_system_wiki" --no-stream --reasoning-enabled

agent:
	uv run --package agent py -m agent.core

main_h:
	uv run --package cli py -m cli.main -h

main:
	uv run --package cli py -m cli.main
