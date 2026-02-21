from __future__ import annotations

import time
from urllib.parse import urldefrag, urljoin

import scrapy
from scrapy import Request
from scrapy.crawler import Crawler
from scrapy.exceptions import CloseSpider
from scrapy.http import Response


class ListSpider(scrapy.Spider):
    name = "list_spider"

    def __init__(self, urls: list[str] | None = None, page_limit: int = 500, **kwargs) -> None:
        super().__init__(**kwargs)
        self.urls = urls or []
        self.page_limit = page_limit
        self._started_at = time.monotonic()

    @classmethod
    def from_crawler(cls, crawler: Crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider.time_limit = crawler.settings.getint("CORPUS_TIME_LIMIT") or None
        return spider

    def _is_timed_out(self) -> bool:
        if self.time_limit is None:
            return False
        return (time.monotonic() - self._started_at) >= self.time_limit

    def _requests(self):
        for index, url in enumerate(self.urls[: self.page_limit]):
            if self._is_timed_out():
                break
            yield Request(
                url=url,
                callback=self.parse,
                errback=self.handle_error,
                dont_filter=True,
                meta={"corpus_index": index, "handle_httpstatus_all": True},
            )

    async def start(self):
        for request in self._requests():
            yield request

    def start_requests(self):
        yield from self._requests()

    def parse(self, response: Response):
        if self._is_timed_out():
            raise CloseSpider("time_limit_reached")

        outlinks = sorted(
            {
                urldefrag(urljoin(response.url, href))[0]
                for href in response.css("a::attr(href)").getall()
                if urldefrag(urljoin(response.url, href))[0].startswith(("http://", "https://"))
            }
        )

        text_chunks = [chunk.strip() for chunk in response.xpath("//body//text()").getall() if chunk.strip()]
        text = "\n".join(text_chunks)

        yield {
            "content_bytes": bytes(response.body),
            "content_type": response.headers.get("Content-Type", b"").decode("latin-1", errors="ignore"),
            "error": None,
            "index": int(response.meta.get("corpus_index", 0)),
            "outlinks": outlinks,
            "status_code": int(response.status),
            "text": text,
            "url": response.url,
        }

    def handle_error(self, failure):
        request = failure.request
        response = getattr(failure.value, "response", None)

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
