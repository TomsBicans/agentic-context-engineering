# corpus-scraper

Generic corpus scraper CLI. For now it only validates CLI arguments, builds a config object, and prints it as pretty JSON for reproducibility.

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

Each command prints the parsed configuration as deterministic JSON. Crawling/fetching is not implemented yet.
