import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from corpus_scraper.main import main
from corpus_scraper.pipelines.manifest import ListHttpManifestPipeline


def test_help_runs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Corpus scraper CLI" in out


def test_crawl_parses_json(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    main(
        [
            "crawl",
            "--output-dir",
            str(output_dir),
            "--corpus-name",
            "demo",
            "--start-url",
            "https://example.com",
            "--allowed-domain",
            "example.com",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "crawl"
    assert payload["config"]["start_url"] == "https://example.com"


def test_list_parses_json(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    input_file = tmp_path / "urls.txt"
    input_file.write_text("https://example.com\n", encoding="utf-8")
    main(
        [
            "list",
            "--output-dir",
            str(output_dir),
            "--corpus-name",
            "demo",
            "--input-file",
            str(input_file),
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "list"
    assert payload["config"]["input_file"] == str(input_file)


def test_repo_parses_json(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    main(
        [
            "repo",
            "--output-dir",
            str(output_dir),
            "--corpus-name",
            "scipy",
            "--repo-url",
            "https://github.com/scipy/scipy",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "repo"
    assert payload["config"]["repo_url"].endswith("scipy/scipy")
    assert payload["config"]["include"] == ["**/*"]


def test_mediawiki_parses_json(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    main(
        [
            "mediawiki",
            "--output-dir",
            str(output_dir),
            "--corpus-name",
            "wiki-demo",
            "--api-url",
            "https://en.wikipedia.org/w/api.php",
            "--category",
            "Machine learning",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "mediawiki"
    assert payload["config"]["fetcher"] == "mediawiki"


def test_invalid_corpus_name_fails(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "crawl",
                "--output-dir",
                str(output_dir),
                "--corpus-name",
                "bad name",
                "--start-url",
                "https://example.com",
                "--allowed-domain",
                "example.com",
                "--dry-run",
            ]
        )
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "corpus-name" in err


def test_mediawiki_requires_scope(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "mediawiki",
                "--output-dir",
                str(output_dir),
                "--corpus-name",
                "wiki-demo",
                "--api-url",
                "https://en.wikipedia.org/w/api.php",
                "--dry-run",
            ]
        )
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "mediawiki requires" in err


def test_pandoc_requires_markdown_text_format(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    input_file = tmp_path / "urls.txt"
    input_file.write_text("https://example.com\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "list",
                "--output-dir",
                str(output_dir),
                "--corpus-name",
                "demo",
                "--input-file",
                str(input_file),
                "--markdown-converter",
                "pandoc",
                "--dry-run",
            ]
        )
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "requires --text-format markdown" in err


def test_writes_stub_artifacts(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    main(
        [
            "repo",
            "--output-dir",
            str(output_dir),
            "--corpus-name",
            "scipy",
            "--repo-url",
            "https://github.com/scipy/scipy",
        ]
    )
    capsys.readouterr()
    corpus_dir = output_dir / "scipy"
    assert corpus_dir.exists()

    config_path = corpus_dir / "config.json"
    manifest_path = corpus_dir / "manifest.jsonl"
    assert config_path.exists()
    assert manifest_path.exists()
    assert manifest_path.read_text(encoding="utf-8") == ""

    written_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert written_config["mode"] == "repo"
    assert written_config["config"]["repo_url"] == "https://github.com/scipy/scipy"


def test_dry_run_does_not_write_files(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    main(
        [
            "crawl",
            "--output-dir",
            str(output_dir),
            "--corpus-name",
            "demo",
            "--start-url",
            "https://example.com",
            "--allowed-domain",
            "example.com",
            "--dry-run",
        ]
    )
    capsys.readouterr()
    assert not (output_dir / "demo").exists()


def test_list_http_ingestion_writes_manifest_and_artifacts(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            payload = (
                b"<html><body><h1>Solar System</h1>"
                b"<a href='/wiki/Planet'>Planet</a></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output_dir = tmp_path / "corpora"
    input_file = tmp_path / "urls.txt"
    input_file.write_text(f"http://127.0.0.1:{server.server_port}/solar\n", encoding="utf-8")

    try:
        main(
            [
                "list",
                "--output-dir",
                str(output_dir),
                "--corpus-name",
                "solar",
                "--input-file",
                str(input_file),
                "--store-text",
            ]
        )
        capsys.readouterr()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    corpus_dir = output_dir / "solar"
    manifest_path = corpus_dir / "manifest.jsonl"
    manifest_lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 1
    entry = json.loads(manifest_lines[0])
    assert entry["status_code"] == 200
    assert entry["raw_path"].startswith("raw/")
    assert entry["text_path"].startswith("text/")
    assert entry["outlinks_path"].startswith("outlinks/")
    assert entry["outlinks_count"] == 1

    assert (corpus_dir / entry["raw_path"]).exists()
    assert (corpus_dir / entry["text_path"]).exists()
    assert (corpus_dir / entry["outlinks_path"]).exists()


def test_pipeline_markdown_with_pandoc(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pandoc(*args, **kwargs):
        assert args[0] == ["pandoc", "--from", "html", "--to", "gfm"]
        pruned_html = kwargs["input"]
        assert "vector-header-container" not in pruned_html
        assert "mw-jump-link" not in pruned_html
        assert "mw-aria-live-region" not in pruned_html
        assert "Solar System" in pruned_html
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="# Solar System\n\nMass and orbit.\n",
        )

    monkeypatch.setattr("corpus_scraper.pipelines.manifest.subprocess.run", fake_pandoc)

    corpus_dir = tmp_path / "solar_md"
    manifest_path = corpus_dir / "manifest.jsonl"
    pipeline = ListHttpManifestPipeline(
        corpus_dir=corpus_dir,
        manifest_path=manifest_path,
        store_raw=True,
        store_text=True,
        store_outlinks=True,
        compress=False,
        deduplicate_content=True,
        text_format="markdown",
        markdown_converter="pandoc",
    )
    pipeline.open_spider()
    pipeline.process_item(
        {
            "content_bytes": (
                b"<html><body>"
                b"<div id='mw-aria-live-region'>status updates</div>"
                b"<a class='mw-jump-link' href='#content'>Jump to content</a>"
                b"<div class='vector-header-container'>Header chrome</div>"
                b"<h1>Solar System</h1><p>Mass and orbit.</p>"
                b"</body></html>"
            ),
            "content_type": "text/html; charset=utf-8",
            "error": None,
            "index": 0,
            "outlinks": ["https://en.wikipedia.org/wiki/Planet"],
            "status_code": 200,
            "text": "Solar System\nMass and orbit.",
            "url": "https://en.wikipedia.org/wiki/Solar_System",
        }
    )
    pipeline.close_spider()

    entry = json.loads(manifest_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert entry["text_path"].endswith(".md")
    markdown_text = (corpus_dir / entry["text_path"]).read_text(encoding="utf-8")
    assert "# Solar System" in markdown_text
