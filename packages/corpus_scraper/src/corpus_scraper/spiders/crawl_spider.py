from __future__ import annotations

import re
import time
from urllib.parse import urldefrag, urljoin, urlparse

import scrapy
from scrapy import Request
from scrapy.crawler import Crawler
from scrapy.exceptions import CloseSpider
from scrapy.http import Response

_NON_ARTICLE_NAMESPACE_RE = re.compile(
    r"^/wiki/(Special|Talk|Category|File|Template|Help|Wikipedia|Portal|Draft|TimedText|Module|User|User_talk|Book):",
    re.IGNORECASE,
)


class CrawlSpider(scrapy.Spider):
    name = "crawl_spider"

    def __init__(
        self,
        start_url: str,
        allowed_domains: list[str] | None = None,
        page_limit: int = 500,
        allow_pattern: str | None = None,
        deny_pattern: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.start_url = start_url
        self.allowed_domains_input = allowed_domains or []
        self.page_limit = page_limit
        self.allow_pattern = re.compile(allow_pattern) if allow_pattern else None
        self.deny_pattern = re.compile(deny_pattern) if deny_pattern else None
        self._started_at = time.monotonic()
        self._scheduled_urls: set[str] = set()
        self._scheduled_count = 0

    @classmethod
    def from_crawler(cls, crawler: Crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider.time_limit = crawler.settings.getint("CORPUS_TIME_LIMIT") or None
        return spider

    def _is_timed_out(self) -> bool:
        if self.time_limit is None:
            return False
        return (time.monotonic() - self._started_at) >= self.time_limit

    def _is_allowed_domain(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        for allowed in self.allowed_domains_input:
            domain = allowed.lower().lstrip(".")
            if host == domain or host.endswith(f".{domain}"):
                return True
        return False

    def _is_article_path(self, url: str) -> bool:
        path = urlparse(url).path or ""
        if not path.startswith("/wiki/"):
            return False
        if _NON_ARTICLE_NAMESPACE_RE.match(path):
            return False
        return True

    def _normalize_url(self, base_url: str, href: str) -> str | None:
        absolute = urljoin(base_url, href)
        clean, _fragment = urldefrag(absolute)
        parsed = urlparse(clean)
        if parsed.scheme not in {"http", "https"}:
            return None
        return clean

    def _should_follow(self, url: str) -> bool:
        if not self._is_allowed_domain(url):
            return False
        if not self._is_article_path(url):
            return False
        if self.allow_pattern and not self.allow_pattern.search(url):
            return False
        if self.deny_pattern and self.deny_pattern.search(url):
            return False
        return True

    def _outlinks(self, response: Response) -> list[str]:
        links: set[str] = set()
        for href in response.css("a::attr(href)").getall():
            normalized = self._normalize_url(response.url, href)
            if normalized and normalized.startswith(("http://", "https://")):
                links.add(normalized)
        return sorted(links)

    def _text(self, response: Response) -> str:
        chunks = [chunk.strip() for chunk in response.xpath("//body//text()").getall() if chunk.strip()]
        return "\n".join(chunks)

    def _requests_from_links(self, response: Response):
        for href in sorted(response.css("a::attr(href)").getall()):
            if self._scheduled_count >= self.page_limit:
                return
            normalized = self._normalize_url(response.url, href)
            if not normalized or normalized in self._scheduled_urls:
                continue
            if not self._should_follow(normalized):
                continue

            index = self._scheduled_count
            self._scheduled_count += 1
            self._scheduled_urls.add(normalized)
            yield Request(
                url=normalized,
                callback=self.parse,
                errback=self.handle_error,
                dont_filter=True,
                meta={"corpus_index": index, "handle_httpstatus_all": True},
            )

    def _initial_request(self):
        index = self._scheduled_count
        self._scheduled_count += 1
        self._scheduled_urls.add(self.start_url)
        return Request(
            url=self.start_url,
            callback=self.parse,
            errback=self.handle_error,
            dont_filter=True,
            meta={"corpus_index": index, "handle_httpstatus_all": True},
        )

    async def start(self):
        if self._is_timed_out() or self.page_limit <= 0:
            return
        yield self._initial_request()

    def start_requests(self):
        if self._is_timed_out() or self.page_limit <= 0:
            return
        yield self._initial_request()

    def parse(self, response: Response):
        if self._is_timed_out():
            raise CloseSpider("time_limit_reached")
        self.logger.info("Fetched [%s] %s", response.status, response.url)

        yield {
            "content_bytes": bytes(response.body),
            "content_type": response.headers.get("Content-Type", b"").decode("latin-1", errors="ignore"),
            "error": None,
            "index": int(response.meta.get("corpus_index", 0)),
            "outlinks": self._outlinks(response),
            "status_code": int(response.status),
            "text": self._text(response),
            "url": response.url,
        }

        if self._scheduled_count >= self.page_limit:
            return

        yield from self._requests_from_links(response)

    def handle_error(self, failure):
        request = failure.request
        response = getattr(failure.value, "response", None)
        self.logger.warning("Request failed: %s (%s)", request.url, failure.value)

        yield {
            "content_bytes": b"",
            "content_type": "",
            "error": str(failure.value),
            "index": int(request.meta.get("corpus_index", 0)),
            "outlinks": [],
            "status_code": getattr(response, "status", None),
            "text": "",
            "url": request.url,
        }
