# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Senior Engineer Persona (always follow)

**You are a senior staff software engineer with 15+ years of experience.**  
Write production-grade, clean, idiomatic Python.

**Core principles (never violate):**

- Follow PEP 8 + modern Python 3.11+ idioms (type hints everywhere, dataclasses/Pydantic v2, context managers, etc.).
- Prefer simplicity and readability over cleverness. "Explicit is better than implicit."
- Self-documenting code first. Comments only for *why*, not *what*.
- Small, focused functions/classes with single responsibilities.
- Excellent naming, early returns, clear error handling, observability.
- Strong separation of concerns and extensibility (Strategy, Factory, Adapter patterns where appropriate).
- Always consider performance, testability, maintainability, and reproducibility.
- Fail fast, never swallow exceptions silently.

**When editing code:**

- Preserve existing style unless there is clear improvement.
- Remove unnecessary comments, banners, and meta-commentary about the agent/process.
- Keep changes focused and justified.

## Commands

**Setup:**

```bash
make i                    # Install all dependencies (uv sync --all-packages --all-groups)
make install_tools        # Install ace/agent/corpus_scraper as uv tools
```

**Testing:**

```bash
make test_all             # Run all tests (agent + cli)
make test_agent           # pytest for agent package only
make test_cli             # pytest for cli package only

# Single test file:
uv run --package cli python3 -m pytest packages/cli/src/cli/tests/test_rich_render.py -v
uv run --package agent python3 -m pytest packages/agent/src/agent/tests/test_tools.py -v
```

**Running the CLI:**

```bash
make ace                                  # Interactive REPL
make ace_query q="What is the Sun's mass?" model=qwen3:4b ctx=8192
make ace_query_json q="List key planets"  # JSON output
make ace_query_no_stream q="..."          # Non-streaming

# Directly:
uv run ace --path-to-corpora ./corpora/scraped_data/solar_system_wiki
uv run ace query "What is the Earth's mass?" --path-to-corpora ./corpora/scraped_data/solar_system_wiki
```

**Corpus ingestion:**

```bash
make corpus_scraper_solar_system   # Wikipedia solar system (~500 pages)
make corpus_scraper_oblivion       # UESP Oblivion wiki
make corpus_scraper_scipy          # scipy GitHub repo source
```

## Architecture

This is a monorepo with three independent `uv` workspace packages. Dependency direction: `cli → agent` (workspace dep);
`corpus_scraper` is standalone.

### `packages/agent` — LLM agent engine

The core agent is a LangGraph `CompiledStateGraph` initialized via `agent.core.initialize_agent()`. Two roles exist:

- **EXAMINEE** (`AgentRole.EXAMINEE`): answers questions using a local text corpus via tools. System prompt mandates
  inline citations in the format `statement [file: relative/path, lines:a-b]`.
- **EXAMINER** (`AgentRole.EXAMINER`): verifies claims against cited sources; rates each as SUPPORTED /
  PARTIALLY_SUPPORTED / NOT_SUPPORTED / BAD_REFERENCE.

Tool sets are role-specific (`tools.py`):

- **PerformerTools** (EXAMINEE): `list_paths`, `read_lines` (capped 80 lines/12KB), `search` (regex, ≤200 matches),
  `file_meta`, `time_elapsed`, `time_left`
- **ValidatorTools** (EXAMINER): `resolve_reference`

Streaming is event-driven: `interface/events.py` exposes `iter_stream_events(agent, prompt)` which yields
`("token", str)`, `("tool_call", str)`, `("tool_result", {"name", "snippet"})`, `("done", AgentResponse)`. The older
`interface/streaming.py` prints directly to stdout (kept for backward compat).

### `packages/cli` — Terminal interface

Two modes via argparse in `main.py`:

- **`ace query "<prompt>"`** — single-shot; delegates to `commands/query.py`
- **`ace`** (no subcommand) — interactive REPL; delegates to `repl/session.py`

All Rich rendering lives in `ui/rich_render.py`:

**`render_stream_live(agent, prompt)`** — opens a `Rich.Live` context and processes events from `iter_stream_events`.
Maintains a `segments` list of alternating `_AceSegment` (AI turns) and tool-dicts, rendered chronologically as panels.

**`_AceSegment`** — per-turn think-block filter. Key semantics:

- `reasoning` is `True` only while actively inside an unclosed `<think>` block
- `think_chars` = `max(0, _last_close_end)` — position past the last `</think>` tag; 0 if no `</think>` seen
- `display` returns: post-`</think>` content (if think block completed), `""` (if inside open `<think>`), or full
  content (if no think blocks at all)

**`render_statements(response, k, fmt)`** — extracts statements from `AgentResponse` using a four-tier fallback: (1)
structured `ExamineeResponse.statements`, (2) embedded `<examinee-response>[…]</examinee-response>` JSON, (3) inline
citations regex, (4) paragraph split.

### `packages/corpus_scraper` — Corpus ingestion

Four modes: `crawl` (Scrapy link-following), `list` (fetch URLs from file), `repo` (git clone + file iteration),
`mediawiki` (API-based). All modes write to `<output_dir>/<corpus_name>/` with a `manifest.jsonl` tracking per-item
metadata (URL, hash, paths, status). Pydantic models in `config/models.py` validate all config inputs.

## Key conventions

- Lines in tool arguments are **0-based, half-open `[a, b)`** — consistent across `read_lines`, `search` output, and
  EXAMINER's `resolve_reference`.
- The `--require-tools` flag enforces that the EXAMINEE calls at least one tool before answering (reproducibility).
- The REPL lazy-initializes the agent on first real query; agent is reused across queries in the same session.
- REPL history is persisted to `~/.ace_history`.

# Code style rules

- Do NOT add decorative banner comments (e.g. lines of dashes) or “meta” comments about the agent/process.
- Only add/edit comments when they explain non-obvious intent, constraints, or tricky behavior.
- Never add comments that describe the conversation, turns, thinking, prompts, or tooling.
- If you feel a banner would help, use meaningful function/section names instead.