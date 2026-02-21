import json

import pytest

from corpus_scraper.main import main


def test_help_runs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Corpus scraper CLI" in out


def test_crawl_parses_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "crawl",
            "--output-dir",
            "./corpora/scraped_data",
            "--corpus-name",
            "demo",
            "--start-url",
            "https://example.com",
            "--allowed-domain",
            "example.com",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "crawl"
    assert payload["config"]["start_url"] == "https://example.com"


def test_list_parses_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "list",
            "--output-dir",
            "./corpora/scraped_data",
            "--corpus-name",
            "demo",
            "--input-file",
            "./urls.txt",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "list"
    assert payload["config"]["input_file"] == "./urls.txt"


def test_repo_parses_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "repo",
            "--output-dir",
            "./corpora/scraped_data",
            "--corpus-name",
            "scipy",
            "--repo-url",
            "https://github.com/scipy/scipy",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "repo"
    assert payload["config"]["repo_url"].endswith("scipy/scipy")
    assert payload["config"]["include"] == ["**/*"]


def test_mediawiki_parses_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "mediawiki",
            "--output-dir",
            "./corpora/scraped_data",
            "--corpus-name",
            "wiki-demo",
            "--api-url",
            "https://en.wikipedia.org/w/api.php",
            "--category",
            "Machine learning",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "mediawiki"
    assert payload["config"]["fetcher"] == "mediawiki"


def test_invalid_corpus_name_fails(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "crawl",
                "--output-dir",
                "./corpora/scraped_data",
                "--corpus-name",
                "bad name",
                "--start-url",
                "https://example.com",
                "--allowed-domain",
                "example.com",
            ]
        )
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "corpus-name" in err


def test_mediawiki_requires_scope(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "mediawiki",
                "--output-dir",
                "./corpora/scraped_data",
                "--corpus-name",
                "wiki-demo",
                "--api-url",
                "https://en.wikipedia.org/w/api.php",
            ]
        )
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "mediawiki requires" in err
