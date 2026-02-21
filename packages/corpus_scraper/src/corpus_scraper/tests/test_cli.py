import json

import pytest

from corpus_scraper.main import main


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
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "crawl"
    assert payload["config"]["start_url"] == "https://example.com"


def test_list_parses_json(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "corpora"
    main(
        [
            "list",
            "--output-dir",
            str(output_dir),
            "--corpus-name",
            "demo",
            "--input-file",
            "./urls.txt",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "list"
    assert payload["config"]["input_file"] == "./urls.txt"


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
            ]
        )
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "mediawiki requires" in err


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
