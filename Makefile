PYTHON := python3

# Setup
install_dependencies:
	uv sync --all-packages --all-groups

run_tests:
	uv run --package agent ${PYTHON} -m pytest -q

install_tools:
	uv tool install ./packages/cli
	uv tool install ./packages/agent


corpus_scraper_h:
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main -h

corpus_scraper:
	uv run --package corpus_scraper py -m corpus_scraper.main
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main crawl -h

CORPORA_OUTPUT_DIR := ./corpora/scraped_data
SOLAR_SYSTEM_URLS_FILE := ./corpora/seed_urls/solar_system_wiki.txt
SOLAR_SYSTEM_CORPUS_NAME := solar_system_wiki

corpus_scraper_solar_system:
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main list \
		--output-dir ${CORPORA_OUTPUT_DIR} \
		--corpus-name ${SOLAR_SYSTEM_CORPUS_NAME} \
		--input-file ${SOLAR_SYSTEM_URLS_FILE} \
		--fetcher http \
		--store-text

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
	uv run --package agent ${PYTHON} -m agent.core -h

agent_test:
	uv run --package agent ${PYTHON} -m agent.core ${q} --model ${model} --num_ctx ${ctx} --role examinee --path-to-corpora "./corpora/scraped_data/solar_system_wiki" --no-stream --reasoning-enabled

agent:
	uv run --package agent ${PYTHON} -m agent.core

main_h:
	uv run --package cli ${PYTHON} -m cli.main -h

main:
	uv run --package cli ${PYTHON} -m cli.main
