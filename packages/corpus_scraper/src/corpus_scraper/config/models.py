from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

FetcherType = Literal["http", "playwright", "mediawiki"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class CommonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str
    corpus_name: str
    page_limit: int = 500
    time_limit: int | None = None
    fetcher: FetcherType = "http"
    concurrency: int = 4
    download_delay: float = 0.0
    timeout: float = 30.0
    user_agent: str | None = None
    store_raw: bool = True
    store_text: bool = False
    store_outlinks: bool = True
    compress: bool = False
    deduplicate_content: bool = True
    log_level: LogLevel = "INFO"
    dry_run: bool = False

    @field_validator("corpus_name")
    @classmethod
    def validate_corpus_name(cls, value: str) -> str:
        if re.fullmatch(r"[a-zA-Z0-9_-]+", value) is None:
            raise ValueError("corpus-name must match [a-zA-Z0-9_-]+")
        return value

    @field_validator("concurrency")
    @classmethod
    def validate_concurrency(cls, value: int) -> int:
        if not 1 <= value <= 64:
            raise ValueError("concurrency must be between 1 and 64")
        return value

    @field_validator("download_delay")
    @classmethod
    def validate_download_delay(cls, value: float) -> float:
        if value < 0:
            raise ValueError("download-delay must be >= 0")
        return value

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout must be > 0")
        return value

    @field_validator("time_limit")
    @classmethod
    def validate_time_limit(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("time-limit must be > 0")
        return value


class CrawlConfig(CommonConfig):
    start_url: str
    allowed_domain: list[str] = Field(default_factory=list)
    max_depth: int = 3
    allow_pattern: str | None = None
    deny_pattern: str | None = None

    @field_validator("max_depth")
    @classmethod
    def validate_max_depth(cls, value: int) -> int:
        if not 0 <= value <= 50:
            raise ValueError("max-depth must be between 0 and 50")
        return value

    @model_validator(mode="after")
    def validate_allowed_domain(self) -> CrawlConfig:
        if not self.allowed_domain:
            raise ValueError("at least one allowed-domain is required")
        return self


class ListConfig(CommonConfig):
    input_file: str
    skip_empty: bool = True


class RepoConfig(CommonConfig):
    repo_url: str
    ref: str = "HEAD"
    subpath: str | None = None
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude: list[str] = Field(default_factory=list)
    max_files: int = 100

    @field_validator("max_files")
    @classmethod
    def validate_max_files(cls, value: int) -> int:
        if not 1 <= value <= 1_000_000:
            raise ValueError("max-files must be between 1 and 1_000_000")
        return value


class MediaWikiConfig(CommonConfig):
    api_url: str
    category: str | None = None
    namespace: int = 0
    allpages_prefix: str | None = None

    @model_validator(mode="after")
    def validate_listing_scope(self) -> MediaWikiConfig:
        if not self.category and not self.allpages_prefix:
            raise ValueError("mediawiki requires --category or --allpages-prefix")
        return self


class ScrapeJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["crawl", "list", "repo", "mediawiki"]
    config: CrawlConfig | ListConfig | RepoConfig | MediaWikiConfig

    @model_validator(mode="after")
    def validate_mode_config(self) -> ScrapeJob:
        expected = {
            "crawl": CrawlConfig,
            "list": ListConfig,
            "repo": RepoConfig,
            "mediawiki": MediaWikiConfig,
        }
        if not isinstance(self.config, expected[self.mode]):
            raise ValueError("config does not match mode")
        return self


def job_to_json(job: ScrapeJob) -> str:
    data = job.model_dump()
    return json.dumps(data, sort_keys=True, indent=2)


def format_validation_error(error: ValidationError) -> str:
    lines: list[str] = []
    for entry in error.errors():
        loc = ".".join(str(part) for part in entry.get("loc", [])) or "config"
        msg = entry.get("msg", "Invalid value")
        lines.append(f"{loc}: {msg}")
    return "\n".join(lines)
