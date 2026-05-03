"""Orchestrator for the analyze command.

Walks input experiment_results JSONL files, applies the A2 examiner to each
RunResult, and writes parallel AnalysisResult JSONL files. Idempotent by
default — runs already present in the output are skipped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from experiment_runner.models.result import RunResult
from rich.console import Console
from tqdm import tqdm

from result_processor.analysis.citation_parser import (
    ParsedCitation,
    extract_citations,
    split_sentences,
)
from result_processor.analysis.examiner import ExaminerLLM
from result_processor.analysis.excerpt_resolver import ExcerptResolver
from result_processor.analysis.io import (
    append_analysis,
    iter_run_results,
    load_existing_run_ids,
)
from result_processor.models.analysis import (
    AnalysisResult,
    ClaimAnalysis,
    ClaimStatus,
    Verdict,
)


_PASS_SUPPORT_THRESHOLD = 0.7


def analyze_directory(
    experiment_results_dir: str,
    output_dir: str,
    path_to_corpora: str,
    examiner_model: str,
    num_ctx: int = 8192,
    input_files: Optional[Iterable[str]] = None,
    resume: bool = True,
) -> None:
    console = Console()

    in_dir = Path(experiment_results_dir).resolve()
    out_dir = Path(output_dir).resolve()
    corpora_root = Path(path_to_corpora).resolve()

    if not in_dir.is_dir():
        raise ValueError(f"experiment_results_dir not found: {in_dir}")
    if not corpora_root.is_dir():
        raise ValueError(f"path_to_corpora not found: {corpora_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if input_files:
        targets = [Path(f).resolve() for f in input_files]
    else:
        targets = sorted(p for p in in_dir.glob("*.jsonl") if p.is_file())

    if not targets:
        console.print("[yellow]No JSONL files found to analyze.[/yellow]")
        return

    resolver = ExcerptResolver(corpora_root=corpora_root)
    examiner = ExaminerLLM(model=examiner_model, num_ctx=num_ctx)

    console.print(f"[bold]Analyzing {len(targets)} file(s) with examiner={examiner_model}[/bold]")

    for target in targets:
        out_path = out_dir / target.name
        already_done = load_existing_run_ids(out_path) if resume else set()

        runs = list(iter_run_results(target))
        pending = [r for r in runs if r.run_id not in already_done]
        skipped = len(runs) - len(pending)

        console.print(
            f"  → {target.name}: {len(pending)} to analyze, {skipped} cached"
        )
        if not pending:
            continue

        for run in tqdm(pending, desc=target.name, unit="run", leave=False):
            analysis = _analyze_one(run, resolver, examiner, examiner_model)
            append_analysis(out_path, analysis)

    console.print("[bold green]Analysis complete.[/bold green]")


def _analyze_one(
    run: RunResult,
    resolver: ExcerptResolver,
    examiner: ExaminerLLM,
    examiner_model: str,
) -> AnalysisResult:
    answer = run.answer_text or ""

    citations = extract_citations(answer)
    uncited_sentences = split_sentences(answer)

    claim_analyses: list[ClaimAnalysis] = []

    for citation in citations:
        excerpt = resolver.resolve(
            run.corpus,
            citation.file_path,
            citation.line_start,
            citation.line_end,
        )
        verdict = examiner.classify_claim(
            claim=citation.statement,
            excerpt=excerpt,
            file_path=citation.file_path,
            a=citation.line_start,
            b=citation.line_end,
        )
        claim_analyses.append(_to_claim_analysis(citation, excerpt, verdict.status, verdict.justification))

    # Sentences with no citation get auto-classified as BAD_REFERENCE without
    # an LLM call — there is nothing to verify against.
    for sentence in uncited_sentences:
        claim_analyses.append(
            ClaimAnalysis(
                statement=sentence,
                status=ClaimStatus.BAD_REFERENCE,
                justification="No citation provided for this claim.",
            )
        )

    helpfulness, notes = _summarize(run, examiner, claim_analyses)
    return _aggregate(run, claim_analyses, len(uncited_sentences), examiner_model, helpfulness, notes)


def _to_claim_analysis(
    citation: ParsedCitation,
    excerpt: Optional[str],
    status: ClaimStatus,
    justification: str,
) -> ClaimAnalysis:
    return ClaimAnalysis(
        statement=citation.statement,
        cited_file=citation.file_path,
        cited_line_start=citation.line_start,
        cited_line_end=citation.line_end,
        excerpt=excerpt,
        status=status,
        justification=justification,
    )


def _summarize(
    run: RunResult,
    examiner: ExaminerLLM,
    claims: list[ClaimAnalysis],
) -> tuple[Optional[int], str]:
    if not claims:
        return None, "No claims to summarize."
    summary_lines = [
        f"- [{c.status.value}] {c.statement[:120]}" for c in claims[:30]
    ]
    overall = examiner.summarize(
        question=run.question_text,
        answer=run.answer_text or "",
        claim_summary="\n".join(summary_lines),
    )
    return overall.helpfulness_rating, overall.notes


def _aggregate(
    run: RunResult,
    claims: list[ClaimAnalysis],
    claims_without_citation_count: int,
    examiner_model: str,
    helpfulness: Optional[int],
    notes: str,
) -> AnalysisResult:
    total = len(claims)
    supported = sum(1 for c in claims if c.status == ClaimStatus.SUPPORTED)
    partial = sum(1 for c in claims if c.status == ClaimStatus.PARTIALLY_SUPPORTED)
    not_supported = sum(1 for c in claims if c.status == ClaimStatus.NOT_SUPPORTED)
    bad = sum(1 for c in claims if c.status == ClaimStatus.BAD_REFERENCE)

    if total > 0:
        support_rate = supported / total
        error_rate = (not_supported + bad) / total
        overclaim_rate = partial / total
        unsupported_ratio = (not_supported + bad) / total
    else:
        support_rate = error_rate = overclaim_rate = unsupported_ratio = 0.0

    verdict = (
        Verdict.PASS
        if support_rate >= _PASS_SUPPORT_THRESHOLD and not_supported == 0 and total > 0
        else Verdict.FAIL
    )

    return AnalysisResult(
        run_id=run.run_id,
        examiner_model=examiner_model,
        claims=claims,
        claims_total=total,
        claims_supported=supported,
        claims_partially_supported=partial,
        claims_not_supported=not_supported,
        claims_bad_reference=bad,
        claims_without_citation_count=claims_without_citation_count,
        support_rate=support_rate,
        error_rate=error_rate,
        overclaim_rate=overclaim_rate,
        unsupported_claim_ratio=unsupported_ratio,
        verdict=verdict,
        helpfulness_rating=helpfulness,
        examiner_notes=notes,
    )
