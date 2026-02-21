from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pydantic import ValidationError

from corpus_scraper.config import (
    CrawlConfig,
    ListConfig,
    MediaWikiConfig,
    RepoConfig,
    ScrapeJob,
    format_validation_error,
    job_to_json,
)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    parser.add_argument(
        "--corpus-name",
        required=True,
        help="Corpus name (letters, numbers, _ and -)",
    )
    parser.add_argument("--page-limit", type=int, default=500, help="Max pages/items to fetch")
    parser.add_argument("--time-limit", type=int, default=None, help="Time limit in seconds")
    parser.add_argument(
        "--fetcher",
        choices=["http", "playwright", "mediawiki"],
        default="http",
        help="Fetcher strategy",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent requests (1-64)",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=0.0,
        help="Delay between requests in seconds",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    parser.add_argument("--user-agent", default=None, help="Custom user agent")
    parser.add_argument(
        "--store-raw",
        dest="store_raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store raw responses",
    )
    parser.add_argument(
        "--store-text",
        dest="store_text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store extracted text",
    )
    parser.add_argument(
        "--store-outlinks",
        dest="store_outlinks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store outlinks",
    )
    parser.add_argument(
        "--compress",
        dest="compress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compress stored data",
    )
    parser.add_argument(
        "--deduplicate-content",
        dest="deduplicate_content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate content",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config only",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corpus scraper CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    crawl_parser = subparsers.add_parser("crawl", help="Discover via link-follow crawl")
    _add_common_options(crawl_parser)
    crawl_parser.add_argument("--start-url", required=True, help="Start URL for crawling")
    crawl_parser.add_argument(
        "--allowed-domain",
        action="append",
        required=True,
        help="Allowed domain(s)",
    )
    crawl_parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max crawl depth (0-50)",
    )
    crawl_parser.add_argument("--allow-pattern", default=None, help="Allow regex pattern")
    crawl_parser.add_argument("--deny-pattern", default=None, help="Deny regex pattern")

    list_parser = subparsers.add_parser("list", help="Discover via URL list file")
    _add_common_options(list_parser)
    list_parser.add_argument("--input-file", required=True, help="File with URLs (one per line)")
    list_parser.add_argument(
        "--skip-empty",
        dest="skip_empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip empty lines",
    )

    repo_parser = subparsers.add_parser("repo", help="Discover via git repo files copy")
    _add_common_options(repo_parser)
    repo_parser.add_argument("--repo-url", required=True, help="Git repository URL")
    repo_parser.add_argument("--ref", default="HEAD", help="Branch/tag/commit ref")
    repo_parser.add_argument("--subpath", default=None, help="Subpath within repo")
    repo_parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Glob(s) to include",
    )
    repo_parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Glob(s) to exclude",
    )
    repo_parser.add_argument("--max-files", type=int, default=100, help="Max files to include")

    mediawiki_parser = subparsers.add_parser("mediawiki", help="Discover via MediaWiki API listing")
    _add_common_options(mediawiki_parser)
    mediawiki_parser.add_argument("--api-url", required=True, help="MediaWiki API URL")
    mediawiki_parser.add_argument("--category", default=None, help="Category to list")
    mediawiki_parser.add_argument("--namespace", type=int, default=0, help="Namespace ID")
    mediawiki_parser.add_argument(
        "--allpages-prefix",
        default=None,
        help="Allpages prefix",
    )
    mediawiki_parser.set_defaults(fetcher="mediawiki")

    return parser.parse_args(argv)


def _common_config_kwargs(args: argparse.Namespace) -> dict:
    return {
        "output_dir": args.output_dir,
        "corpus_name": args.corpus_name,
        "page_limit": args.page_limit,
        "time_limit": args.time_limit,
        "fetcher": args.fetcher,
        "concurrency": args.concurrency,
        "download_delay": args.download_delay,
        "timeout": args.timeout,
        "user_agent": args.user_agent,
        "store_raw": args.store_raw,
        "store_text": args.store_text,
        "store_outlinks": args.store_outlinks,
        "compress": args.compress,
        "deduplicate_content": args.deduplicate_content,
        "log_level": args.log_level,
        "dry_run": args.dry_run,
    }


def _build_job(args: argparse.Namespace) -> ScrapeJob:
    if args.mode == "crawl":
        config = CrawlConfig(
            start_url=args.start_url,
            allowed_domain=args.allowed_domain,
            max_depth=args.max_depth,
            allow_pattern=args.allow_pattern,
            deny_pattern=args.deny_pattern,
            **_common_config_kwargs(args),
        )
    elif args.mode == "list":
        config = ListConfig(
            input_file=args.input_file,
            skip_empty=args.skip_empty,
            **_common_config_kwargs(args),
        )
    elif args.mode == "repo":
        include = args.include or ["**/*"]
        exclude = args.exclude or []
        config = RepoConfig(
            repo_url=args.repo_url,
            ref=args.ref,
            subpath=args.subpath,
            include=include,
            exclude=exclude,
            max_files=args.max_files,
            **_common_config_kwargs(args),
        )
    elif args.mode == "mediawiki":
        config = MediaWikiConfig(
            api_url=args.api_url,
            category=args.category,
            namespace=args.namespace,
            allpages_prefix=args.allpages_prefix,
            **_common_config_kwargs(args),
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return ScrapeJob(mode=args.mode, config=config)


def _job_paths(job: ScrapeJob) -> tuple[Path, Path, Path]:
    corpus_dir = Path(job.config.output_dir) / job.config.corpus_name
    return (
        corpus_dir,
        corpus_dir / "config.json",
        corpus_dir / "manifest.jsonl",
    )


def run_job(job: ScrapeJob) -> None:
    corpus_dir, config_path, manifest_path = _job_paths(job)

    if job.config.dry_run:
        return

    corpus_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(job_to_json(job) + "\n", encoding="utf-8")
    # Manifest stays empty until discovery/fetching is wired in.
    manifest_path.write_text("", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        job = _build_job(args)
        run_job(job)
    except ValidationError as exc:
        sys.stderr.write(format_validation_error(exc) + "\n")
        raise SystemExit(1) from exc
    except OSError as exc:
        sys.stderr.write(f"filesystem error: {exc}\n")
        raise SystemExit(1) from exc

    sys.stdout.write(job_to_json(job) + "\n")


if __name__ == "__main__":
    main()
