from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scrapy.crawler import Crawler

from corpus_scraper.pipelines.storage import write_payload


class ListHttpManifestPipeline:
    def __init__(
        self,
        corpus_dir: Path,
        manifest_path: Path,
        store_raw: bool,
        store_text: bool,
        store_outlinks: bool,
        compress: bool,
        deduplicate_content: bool,
    ) -> None:
        self.corpus_dir = corpus_dir
        self.manifest_path = manifest_path
        self.store_raw = store_raw
        self.store_text = store_text
        self.store_outlinks = store_outlinks
        self.compress = compress
        self.deduplicate_content = deduplicate_content
        self.seen_hashes: dict[str, int] = {}
        self.raw_dir = corpus_dir / "raw"
        self.text_dir = corpus_dir / "text"
        self.outlinks_dir = corpus_dir / "outlinks"

    @classmethod
    def from_crawler(cls, crawler: Crawler):
        settings = crawler.settings
        return cls(
            corpus_dir=Path(settings["CORPUS_DIR"]),
            manifest_path=Path(settings["CORPUS_MANIFEST_PATH"]),
            store_raw=settings.getbool("CORPUS_STORE_RAW"),
            store_text=settings.getbool("CORPUS_STORE_TEXT"),
            store_outlinks=settings.getbool("CORPUS_STORE_OUTLINKS"),
            compress=settings.getbool("CORPUS_COMPRESS"),
            deduplicate_content=settings.getbool("CORPUS_DEDUPLICATE_CONTENT"),
        )

    def open_spider(self, _spider=None) -> None:
        if self.store_raw:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.store_text:
            self.text_dir.mkdir(parents=True, exist_ok=True)
        if self.store_outlinks:
            self.outlinks_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.manifest_path.open("a", encoding="utf-8")

    def close_spider(self, _spider=None) -> None:
        self.manifest_file.close()

    def _base_entry(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "error": item.get("error"),
            "index": int(item.get("index", 0)),
            "outlinks_path": None,
            "raw_path": None,
            "status_code": item.get("status_code"),
            "text_path": None,
            "url": item.get("url"),
        }

    def process_item(self, item: dict[str, Any], _spider=None):
        entry = self._base_entry(item)
        content_bytes = item.get("content_bytes", b"")

        if entry["error"] is not None or not content_bytes:
            self.manifest_file.write(json.dumps(entry, sort_keys=True) + "\n")
            return item

        content_hash = hashlib.sha256(content_bytes).hexdigest()
        entry["content_sha256"] = content_hash

        if self.deduplicate_content and content_hash in self.seen_hashes:
            entry["duplicate_of"] = self.seen_hashes[content_hash]
            self.manifest_file.write(json.dumps(entry, sort_keys=True) + "\n")
            return item

        index = int(entry["index"])
        self.seen_hashes[content_hash] = index

        content_type = str(item.get("content_type") or "")
        extension = ".html" if "html" in content_type.lower() else ".bin"
        slug = f"{index:06d}_{content_hash[:12]}"

        if self.store_raw:
            raw_path = write_payload(self.raw_dir / f"{slug}{extension}", content_bytes, self.compress)
            entry["raw_path"] = raw_path.relative_to(self.corpus_dir).as_posix()

        if self.store_text:
            text_payload = (str(item.get("text", "")) + "\n").encode("utf-8")
            text_path = write_payload(self.text_dir / f"{slug}.txt", text_payload, self.compress)
            entry["text_path"] = text_path.relative_to(self.corpus_dir).as_posix()

        if self.store_outlinks:
            outlinks = sorted(item.get("outlinks", []))
            outlinks_payload = (json.dumps(outlinks, sort_keys=True, indent=2) + "\n").encode("utf-8")
            outlinks_path = write_payload(self.outlinks_dir / f"{slug}.json", outlinks_payload, self.compress)
            entry["outlinks_path"] = outlinks_path.relative_to(self.corpus_dir).as_posix()
            entry["outlinks_count"] = len(outlinks)

        self.manifest_file.write(json.dumps(entry, sort_keys=True) + "\n")
        return item
