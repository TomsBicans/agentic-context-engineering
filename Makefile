PYTHON := python3

# Setup
install_dependencies:
	uv sync --all-packages --all-groups

i:
	make install_dependencies

run_tests:
	uv run --package agent ${PYTHON} -m pytest -q

install_tools:
	uv tool install ./packages/cli
	uv tool install ./packages/agent
	uv tool install ./packages/corpus_scraper


corpus_scraper_h:
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main -h

corpus_scraper:
	uv run --package corpus_scraper py -m corpus_scraper.main
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main crawl -h

CORPORA_OUTPUT_DIR := ./corpora/scraped_data
SOLAR_SYSTEM_URLS_FILE := ./corpora/seed_urls/solar_system_wiki.txt
SOLAR_SYSTEM_CORPUS_NAME := solar_system_wiki
SOLAR_SYSTEM_START_URL := https://en.wikipedia.org/wiki/Solar_System
SOLAR_SYSTEM_ALLOWED_DOMAIN := en.wikipedia.org
OBLIVION_CORPUS_NAME := oblivion_wiki
OBLIVION_START_URL := https://en.uesp.net/wiki/Oblivion:Oblivion
OBLIVION_ALLOWED_DOMAIN := en.uesp.net

corpus_scraper_solar_system:
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main crawl \
		--output-dir ${CORPORA_OUTPUT_DIR} \
		--corpus-name ${SOLAR_SYSTEM_CORPUS_NAME} \
		--start-url ${SOLAR_SYSTEM_START_URL} \
		--allowed-domain ${SOLAR_SYSTEM_ALLOWED_DOMAIN} \
		--page-limit 500 \
		--fetcher http \
		--download-delay 0.75 \
		--store-text \
		--text-format markdown \
		--markdown-converter pandoc \
		--log-level INFO

corpus_scraper_oblivion:
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main crawl \
		--output-dir ${CORPORA_OUTPUT_DIR} \
		--corpus-name ${OBLIVION_CORPUS_NAME} \
		--start-url ${OBLIVION_START_URL} \
		--allowed-domain ${OBLIVION_ALLOWED_DOMAIN} \
		--page-limit 500 \
		--download-delay 1.0 \
		--allow-pattern '/wiki/Oblivion:' \
		--fetcher http \
		--store-text \
		--text-format markdown \
		--markdown-converter pandoc \
		--log-level INFO

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
