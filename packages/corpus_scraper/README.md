# corpus-scraper

`corpus-scraper` is a CLI for building reproducible text corpora from:

- web pages discovered by crawling (`crawl`)
- explicit URL lists (`list`)
- git repositories (`repo`)

The command always writes a run config and manifest, and can optionally store raw payloads, extracted text, and outlink metadata.

## Current status

Implemented and usable now:

- `crawl` + `http` (Scrapy-based)
- `list` + `http` (Scrapy-based)
- `repo` (git clone + checkout + file ingestion)

`mediawiki` command exists, but dedicated MediaWiki API execution is not wired yet.

## Quick start

Show CLI help:

```bash
uv run --package corpus_scraper python -m corpus_scraper.main -h
```

Run with URL list (`list`):

```bash
uv run --package corpus_scraper python -m corpus_scraper.main list \
  --output-dir ./corpora/scraped_data \
  --corpus-name demo_list \
  --input-file ./urls.txt \
  --store-text
```

Run domain-bounded crawl (`crawl`):

```bash
uv run --package corpus_scraper python -m corpus_scraper.main crawl \
  --output-dir ./corpora/scraped_data \
  --corpus-name demo_crawl \
  --start-url https://en.wikipedia.org/wiki/Solar_System \
  --allowed-domain en.wikipedia.org \
  --page-limit 500 \
  --download-delay 0.75 \
  --store-text
```

Run git repo ingestion (`repo`):

```bash
uv run --package corpus_scraper python -m corpus_scraper.main repo \
  --output-dir ./corpora/scraped_data \
  --corpus-name scipy_repo \
  --repo-url https://github.com/scipy/scipy \
  --subpath scipy \
  --include '**/*.py' \
  --exclude '**/tests/**' \
  --max-files 500 \
  --store-text
```

## Markdown text conversion

When `--store-text` is enabled, text output defaults to plain text.

To convert HTML to Markdown using Pandoc:

```bash
uv run --package corpus_scraper python -m corpus_scraper.main list \
  --output-dir ./corpora/scraped_data \
  --corpus-name solar_system_markdown \
  --input-file ./corpora/seed_urls/solar_system_wiki.txt \
  --store-text \
  --text-format markdown \
  --markdown-converter pandoc
```

Notes:

- `pandoc` must be installed and available in `PATH`.
- HTML is pruned before conversion to reduce Wikipedia/MediaWiki chrome in output.

## Output layout

For each run, output is written to:

- `<output-dir>/<corpus-name>/config.json`
- `<output-dir>/<corpus-name>/manifest.jsonl`
- `<output-dir>/<corpus-name>/raw/` (if `--store-raw`)
- `<output-dir>/<corpus-name>/text/` (if `--store-text`)
- `<output-dir>/<corpus-name>/outlinks/` (if `--store-outlinks`)

`manifest.jsonl` is append-only for a run and is intended for downstream indexing/inspection.

## Useful options

- `--page-limit`: hard cap for pages/items
- `--time-limit`: stop after N seconds
- `--download-delay`: pause between requests
- `--timeout`: per-request timeout
- `--log-level`: controls Scrapy terminal logs (`DEBUG|INFO|WARNING|ERROR`)
- `--dry-run`: validate and print config only (no files written)

## Makefile targets (repo root)

From the repository root, these targets are available:

- `make corpus_scraper_solar_system`
- `make corpus_scraper_oblivion`
- `make corpus_scraper_scipy`

## Troubleshooting

No text files for Markdown runs:

- check `pandoc --version`
- check run logs (`--log-level INFO` or `DEBUG`)
- inspect `<corpus>/manifest.jsonl` for failed entries

Git repo ingestion fails:

- check `git --version`
- confirm `--repo-url` and `--ref` are valid
