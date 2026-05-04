include setup/*.mk

PYTHON := python3

# Documentation
PLANTUML_VERSION := 1.2026.2
PLANTUML_DIR := docs/diagrams
PLANTUML_JAR := ${PLANTUML_DIR}/plantuml.jar
PLANTUML_URL := https://github.com/plantuml/plantuml/releases/download/v${PLANTUML_VERSION}/plantuml-${PLANTUML_VERSION}.jar
DIAGRAM_OUTPUT_DIR := ${PLANTUML_DIR}/generated

diagrams:
	@mkdir -p ${DIAGRAM_OUTPUT_DIR}
	@command -v java >/dev/null || { echo "java is required to render PlantUML diagrams"; exit 1; }
	@if [ ! -f "${PLANTUML_JAR}" ]; then \
		command -v curl >/dev/null || { echo "curl is required to download PlantUML"; exit 1; }; \
		echo "Downloading PlantUML ${PLANTUML_VERSION}..."; \
		curl -L --fail --show-error --output "${PLANTUML_JAR}" "${PLANTUML_URL}"; \
	fi
	@java -Djava.awt.headless=true -jar "${PLANTUML_JAR}" -tsvg -o generated "${PLANTUML_DIR}"/*.puml
	@echo "Generated diagrams in ${DIAGRAM_OUTPUT_DIR}"

# Setup
install_dependencies:
	uv sync --all-packages --all-groups

i:
	make install_dependencies

run_tests:
	uv run --package agent ${PYTHON} -m pytest -q

test_cli:
	uv run --package cli ${PYTHON} -m pytest packages/cli/src/cli/tests/ -v

test_agent:
	uv run --package agent ${PYTHON} -m pytest -v

test_result_processor:
	uv run --package result-processor ${PYTHON} -m pytest packages/result_processor/src/result_processor/tests/ -v

test_all:
	uv run --package agent ${PYTHON} -m pytest -v
	uv run --package cli ${PYTHON} -m pytest packages/cli/src/cli/tests/ -v
	uv run --package result-processor ${PYTHON} -m pytest packages/result_processor/src/result_processor/tests/ -v

install_tools:
	uv tool install ./packages/cli
	uv tool install ./packages/agent
	uv tool install ./packages/corpus_scraper


corpus_scraper_h:
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main -h

corpus_scraper:
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main crawl -h

# Scrape commands for the 3 main data corpora
CORPORA_OUTPUT_DIR := ./corpora/scraped_data
SOLAR_SYSTEM_CORPUS_NAME := solar_system_wiki
SOLAR_SYSTEM_START_URL := https://en.wikipedia.org/wiki/Solar_System
SOLAR_SYSTEM_ALLOWED_DOMAIN := en.wikipedia.org
OBLIVION_CORPUS_NAME := oblivion_wiki
OBLIVION_START_URL := https://en.uesp.net/wiki/Oblivion:Oblivion
OBLIVION_ALLOWED_DOMAIN := en.uesp.net
SCIPY_CORPUS_NAME := scipy_repo
SCIPY_REPO_URL := https://github.com/scipy/scipy
SCIPY_REF := HEAD
SCIPY_SUBPATH := scipy

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

corpus_scraper_scipy:
	uv run --package corpus_scraper ${PYTHON} -m corpus_scraper.main repo \
		--output-dir ${CORPORA_OUTPUT_DIR} \
		--corpus-name ${SCIPY_CORPUS_NAME} \
		--repo-url ${SCIPY_REPO_URL} \
		--ref ${SCIPY_REF} \
		--subpath ${SCIPY_SUBPATH} \
		--include '**/*.py' \
		--exclude '**/tests/**' \
		--max-files 500 \
		--store-text \
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
model=qwen3:8b
model=qwen2.5-coder:14b-instruct
model=qwen3:14b

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

# ace CLI - single query
corpus=./corpora/scraped_data/solar_system_wiki
fmt=table
k=10

ace_h:
	uv run ace -h

ace_query:
	uv run ace query ${q} \
		--path-to-corpora "${corpus}" \
		--format ${fmt} \
		--k ${k} \
		--model ${model} \
		--num-ctx ${ctx}

ace_query_json:
	uv run ace query ${q} \
		--path-to-corpora "${corpus}" \
		--format json \
		--k ${k} \
		--model ${model} \
		--num-ctx ${ctx}

ace_query_no_stream:
	uv run ace query ${q} \
		--path-to-corpora "${corpus}" \
		--format ${fmt} \
		--k ${k} \
		--model ${model} \
		--num-ctx ${ctx} \
		--no-stream

# ace CLI - interactive REPL
# Usage: make ace [corpus=./path]
ace:
	uv run ace --path-to-corpora "${corpus}"

# Corpora release — archives all scraped corpora and publishes them as a GitHub Release.
# Usage: make release_corpora tag=v1.0-corpora
# Requires: gh auth login with repo scope
corpora_dir=./corpora/scraped_data
corpora_build_dir=./corpora/release_build
release_tag=v1.0-corpora

release_corpora:
	mkdir -p ${corpora_build_dir}
	tar --exclude="*.claw*" --exclude=".gitignore" --exclude="CLAUDE.md" -czf ${corpora_build_dir}/oblivion_wiki.tar.gz     -C ${corpora_dir} oblivion_wiki/
	tar --exclude="*.claw*" --exclude=".gitignore" --exclude="CLAUDE.md" -czf ${corpora_build_dir}/solar_system_wiki.tar.gz -C ${corpora_dir} solar_system_wiki/
	tar --exclude="*.claw*" --exclude=".gitignore" --exclude="CLAUDE.md" -czf ${corpora_build_dir}/scipy_repo.tar.gz        -C ${corpora_dir} scipy_repo/
	cd ${corpora_build_dir} && sha256sum oblivion_wiki.tar.gz solar_system_wiki.tar.gz scipy_repo.tar.gz > corpora_checksums.sha256
	cp ${corpora_build_dir}/corpora_checksums.sha256 ./corpora/corpora_checksums.sha256
	gh release create $(or ${tag},${release_tag}) \
		${corpora_build_dir}/oblivion_wiki.tar.gz \
		${corpora_build_dir}/solar_system_wiki.tar.gz \
		${corpora_build_dir}/scipy_repo.tar.gz \
		${corpora_build_dir}/corpora_checksums.sha256 \
		--title "Experiment corpora $(or ${tag},${release_tag})" \
		--notes "Fixed corpus snapshots used in thesis experiments. Verify integrity with corpora_checksums.sha256."
	rm -rf ${corpora_build_dir}

# Download corpora from GitHub Release and extract locally.
# No authentication required — uses public release URLs via curl.
# Usage: make download_corpora
#        make download_corpora tag=v1.1-corpora
github_repo=TomsBicans/agentic-context-engineering
release_base_url=https://github.com/${github_repo}/releases/download

download_corpora:
	mkdir -p ${corpora_dir}
	curl -L --progress-bar -o ${corpora_dir}/oblivion_wiki.tar.gz     ${release_base_url}/$(or ${tag},${release_tag})/oblivion_wiki.tar.gz
	curl -L --progress-bar -o ${corpora_dir}/solar_system_wiki.tar.gz ${release_base_url}/$(or ${tag},${release_tag})/solar_system_wiki.tar.gz
	curl -L --progress-bar -o ${corpora_dir}/scipy_repo.tar.gz        ${release_base_url}/$(or ${tag},${release_tag})/scipy_repo.tar.gz
	curl -L --progress-bar -o ${corpora_dir}/corpora_checksums.sha256 ${release_base_url}/$(or ${tag},${release_tag})/corpora_checksums.sha256
	cd ${corpora_dir} && sha256sum --check corpora_checksums.sha256
	tar -xzf ${corpora_dir}/oblivion_wiki.tar.gz     -C ${corpora_dir}
	tar -xzf ${corpora_dir}/solar_system_wiki.tar.gz -C ${corpora_dir}
	tar -xzf ${corpora_dir}/scipy_repo.tar.gz        -C ${corpora_dir}
	rm ${corpora_dir}/oblivion_wiki.tar.gz ${corpora_dir}/solar_system_wiki.tar.gz ${corpora_dir}/scipy_repo.tar.gz

# Experiment runner
# Results are written to experiment_results/ as timestamped JSONL files.
# Every invocation produces a new file — no run is ever overwritten.
#
# Variables (override on command line):
#   corpus_id   — which corpus to query  (default: solar_system_wiki)
#   q_id        — single question ID to run (e.g. ss_L1_001)
#   model       — LLM model name          (default: qwen3:4b, set above)
#   ctx         — context window size     (default: 8192, set above)
#   corpus      — path to corpus data dir (default: ./corpora/scraped_data/solar_system_wiki)

data_dir=./data
experiment_results_dir=${data_dir}/experiment_results
questions_dir=./corpora/questions
corpus_id=solar_system_wiki
questions_file=${questions_dir}/solar_system.json
q_id=ss_L1_001 ss_L2_001

experiment_runner_h:
	uv run --package experiment_runner experiment-runner --help

# Run a single question through the ACE system.
# Usage:
#   make ace_experiment_single
#   make ace_experiment_single q_id=ss_L2_003
#   make ace_experiment_single corpus_id=oblivion_wiki questions_file=./corpora/questions/oblivion.json corpus=./corpora/scraped_data/oblivion_wiki q_id=ob_L1_001
ace_experiment_single:
	uv run --package experiment_runner experiment-runner run \
		--system ace \
		--corpus ${corpus_id} \
		--questions-file ${questions_file} \
		--question-ids ${q_id} \
		--output-dir ${experiment_results_dir} \
		--model ${model} \
		--num-ctx ${ctx} \
		--path-to-corpora "${corpus}"

# Dry-run: print the plan without executing.
ace_experiment_dry:
	uv run --package experiment_runner experiment-runner run \
		--system ace \
		--corpus ${corpus_id} \
		--questions-file ${questions_file} \
		--question-ids ${q_id} \
		--output-dir ${experiment_results_dir} \
		--model ${model} \
		--num-ctx ${ctx} \
		--path-to-corpora "${corpus}" \
		--dry-run

# Result processor — A2 examiner analysis, charts, LaTeX tables, dashboard.
# All artefacts are produced from the JSONL files written by experiment_runner.
#
# Variables (override on command line):
#   examiner_model — model used for A2 verification (default: qwen3:4b)
#   ctx            — context window for the examiner (default: 8192, set above)
#   corpora_root   — root containing oblivion_wiki/, solar_system_wiki/, scipy_repo/
#   figures_dir    — output directory for charts/tables (default: ./data/figures)
#   port           — Streamlit port (default: 8501)

analysis_results_dir=${data_dir}/analysis_results
figures_dir=${data_dir}/figures
corpora_root=./corpora/scraped_data
examiner_model=qwen3:4b
port=8501

result_processor_h:
	uv run --package result_processor result-processor --help

# Analyze: run the A2 examiner over experiment_results/*.jsonl.
# Idempotent — already-analyzed runs are skipped unless `resume=0` is set.
analyze:
	uv run --package result_processor result-processor analyze \
		--experiment-results-dir ${experiment_results_dir} \
		--output-dir ${analysis_results_dir} \
		--path-to-corpora ${corpora_root} \
		--examiner-model ${examiner_model} \
		--num-ctx ${ctx}

# Re-analyze everything from scratch.
analyze_force:
	uv run --package result_processor result-processor analyze \
		--experiment-results-dir ${experiment_results_dir} \
		--output-dir ${analysis_results_dir} \
		--path-to-corpora ${corpora_root} \
		--examiner-model ${examiner_model} \
		--num-ctx ${ctx} \
		--no-resume

# Visualize: generate Plotly figures (HTML by default) and LaTeX tables.
# Override formats with: make visualize formats="html png pdf"
formats=html
visualize:
	uv run --package result_processor result-processor visualize \
		--experiment-results-dir ${experiment_results_dir} \
		--analysis-results-dir ${analysis_results_dir} \
		--output-dir ${figures_dir} \
		--formats ${formats}

# Streamlit dashboard — interactive UI for browsing runs and dispatching analyze/visualize.
dashboard:
	uv run --package result_processor result-processor dashboard \
		--port ${port} \
		--experiment-results-dir ${experiment_results_dir} \
		--analysis-results-dir ${analysis_results_dir}
