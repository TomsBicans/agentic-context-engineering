from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiment_runner.models.result import RunResult
from result_processor.analysis.citation_parser import extract_citations, split_sentences, strip_reasoning
from result_processor.analysis.examiner import ExaminerLLM, _ClaimVerdict
from result_processor.analysis.excerpt_resolver import ExcerptResolver
from result_processor.analysis.io import append_analysis, iter_run_results, load_existing_run_ids
from result_processor.analysis.pipeline import _aggregate, _analyze_one
from result_processor.models.analysis import ClaimAnalysis, ClaimStatus, Verdict
from result_processor.tests.conftest import analysis_result, run_payload, write_jsonl


def test_citation_parser_ignores_reasoning_and_normalizes_claims() -> None:
    answer = """
    <think>[draft claim] [file:hidden.md, lines:0-1]</think>
    [Jupiter
    is a planet.] [file: planets.md, lines: 2 - 4]
    """

    assert strip_reasoning(answer).startswith("[Jupiter")
    citations = extract_citations(answer)

    assert len(citations) == 1
    assert citations[0].statement == "Jupiter is a planet."
    assert citations[0].file_path == "planets.md"
    assert citations[0].line_start == 2
    assert citations[0].line_end == 4


def test_split_sentences_removes_citations_and_reasoning() -> None:
    text = "<think>hidden.</think>Jupiter is large. It has moons! [Claim.] [file:x.md, lines:0-1]"

    assert split_sentences(text) == ["Jupiter is large.", "It has moons!"]


def test_excerpt_resolver_reads_clamped_ranges_and_rejects_bad_references(tmp_path) -> None:
    corpus_file = tmp_path / "solar_system_wiki" / "planets.md"
    corpus_file.parent.mkdir(parents=True)
    corpus_file.write_text("Mercury\nVenus\nEarth\n", encoding="utf-8")
    resolver = ExcerptResolver(tmp_path)

    assert resolver.resolve(RunResult.model_validate(run_payload()).corpus, "planets.md", 1, 10) == "Venus\nEarth"
    assert resolver.resolve(RunResult.model_validate(run_payload()).corpus, "missing.md", 0, 1) is None
    assert resolver.resolve(RunResult.model_validate(run_payload()).corpus, "planets.md", 2, 2) is None


def test_jsonl_io_round_trips_runs_and_existing_analysis_ids(tmp_path) -> None:
    run_path = tmp_path / "runs.jsonl"
    analysis_path = tmp_path / "analysis" / "runs.jsonl"
    write_jsonl(run_path, [run_payload(run_id="r1"), run_payload(run_id="r2")])
    append_analysis(analysis_path, analysis_result(run_id="r1"))
    analysis_path.write_text(
        analysis_path.read_text(encoding="utf-8") + "{bad json}\n{\"run_id\":\"r2\"}\n",
        encoding="utf-8",
    )

    assert [run.run_id for run in iter_run_results(run_path)] == ["r1", "r2"]
    assert load_existing_run_ids(analysis_path) == {"r1", "r2"}


def test_iter_run_results_reports_invalid_json_line(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(run_payload()) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"runs\.jsonl:2: invalid JSON"):
        list(iter_run_results(path))


def test_analyze_one_uses_citations_and_aggregates_examiner_results() -> None:
    run = RunResult.model_validate(run_payload())

    class Resolver:
        def resolve(self, corpus, relative_path, a, b):
            return "Jupiter is a planet."

    class Examiner:
        def classify_claim(self, **kwargs):
            return SimpleNamespace(status=ClaimStatus.SUPPORTED, justification="ok")

        def summarize(self, **kwargs):
            return SimpleNamespace(helpfulness_rating=5, notes="Useful answer.")

    result = _analyze_one(run, Resolver(), Examiner(), "qwen3:4b")

    assert result.run_id == run.run_id
    assert result.claims_total == 1
    assert result.claims_supported == 1
    assert result.support_rate == 1.0
    assert result.verdict == Verdict.PASS
    assert result.helpfulness_rating == 5


def test_analyze_one_marks_uncited_sentences_as_bad_references() -> None:
    run = RunResult.model_validate(run_payload(answer_text="Jupiter is large. It has moons."))

    class Examiner:
        def summarize(self, **kwargs):
            return SimpleNamespace(helpfulness_rating=2, notes="Missing citations.")

    result = _analyze_one(run, resolver=SimpleNamespace(), examiner=Examiner(), examiner_model="qwen3:4b")

    assert result.claims_total == 2
    assert result.claims_bad_reference == 2
    assert result.claims_without_citation_count == 2
    assert result.verdict == Verdict.FAIL


def test_aggregate_counts_statuses_and_pass_threshold() -> None:
    run = RunResult.model_validate(run_payload())
    claims = [
        ClaimAnalysis(statement="a", status=ClaimStatus.SUPPORTED),
        ClaimAnalysis(statement="b", status=ClaimStatus.PARTIALLY_SUPPORTED),
        ClaimAnalysis(statement="c", status=ClaimStatus.NOT_SUPPORTED),
        ClaimAnalysis(statement="d", status=ClaimStatus.BAD_REFERENCE),
    ]

    result = _aggregate(run, claims, 1, "qwen3:4b", helpfulness=3, notes="Mixed.")

    assert result.claims_total == 4
    assert result.claims_supported == 1
    assert result.claims_partially_supported == 1
    assert result.claims_not_supported == 1
    assert result.claims_bad_reference == 1
    assert result.support_rate == 0.25
    assert result.error_rate == 0.5
    assert result.overclaim_rate == 0.25
    assert result.verdict == Verdict.FAIL


def test_examiner_llm_classifies_missing_excerpt_without_network() -> None:
    examiner = ExaminerLLM.__new__(ExaminerLLM)

    verdict = examiner.classify_claim("claim", None, "missing.md", 0, 1)

    assert verdict.status == ClaimStatus.BAD_REFERENCE
    assert "could not be resolved" in verdict.justification


def test_examiner_llm_invoke_json_validates_response_and_uses_fallback() -> None:
    class Client:
        def __init__(self, content: str) -> None:
            self.content = content

        def invoke(self, _prompt: str):
            return SimpleNamespace(content=self.content)

    examiner = ExaminerLLM.__new__(ExaminerLLM)
    fallback = _ClaimVerdict(status=ClaimStatus.BAD_REFERENCE, justification="fallback")

    examiner._client = Client('{"status":"supported","justification":"ok"}')
    verdict = examiner._invoke_json("prompt", _ClaimVerdict, fallback=fallback)
    assert verdict.status == ClaimStatus.SUPPORTED

    examiner._client = Client("not-json")
    verdict = examiner._invoke_json("prompt", _ClaimVerdict, fallback=fallback)
    assert verdict == fallback
