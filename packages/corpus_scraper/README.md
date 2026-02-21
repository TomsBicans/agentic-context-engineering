# corpus-scraper

Generic corpus scraper CLI. It validates CLI arguments, builds a config object, and prints it as pretty JSON.

Implemented execution path:
- `list` mode with `http` fetcher uses Scrapy to read URLs from `--input-file`, fetch pages, and write corpus artifacts.

Other modes currently write only stub artifacts (config + empty manifest).

## Usage

Show help:

```bash
uv run --package corpus_scraper py -m corpus_scraper.main -h
```

Crawl (link-following):

```bash
uv run --package corpus_scraper py -m corpus_scraper.main crawl \
  --output-dir ./corpora/scraped_data \
  --corpus-name demo \
  --start-url https://example.com \
  --allowed-domain example.com
```

List (explicit URL file):

```bash
uv run --package corpus_scraper py -m corpus_scraper.main list \
  --output-dir ./corpora/scraped_data \
  --corpus-name demo \
  --input-file ./urls.txt
```

Repo (git repo file listing):

```bash
uv run --package corpus_scraper py -m corpus_scraper.main repo \
  --output-dir ./corpora/scraped_data \
  --corpus-name scipy \
  --repo-url https://github.com/scipy/scipy \
  --subpath scipy/optimize \
  --include '**/*.py' \
  --exclude '**/tests/**'
```

MediaWiki (API listing):

```bash
uv run --package corpus_scraper py -m corpus_scraper.main mediawiki \
  --output-dir ./corpora/scraped_data \
  --corpus-name wiki-demo \
  --api-url https://en.wikipedia.org/w/api.php \
  --category "Machine learning"
```

Each command prints the parsed configuration as deterministic JSON.

When execution writes artifacts, output structure is:

- `<output-dir>/<corpus-name>/config.json`
- `<output-dir>/<corpus-name>/manifest.jsonl`
- `<output-dir>/<corpus-name>/raw/` (if `--store-raw`)
- `<output-dir>/<corpus-name>/text/` (if `--store-text`)
- `<output-dir>/<corpus-name>/outlinks/` (if `--store-outlinks`)

Use `--dry-run` to skip writing files.
