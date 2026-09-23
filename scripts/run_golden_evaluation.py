"""Run retrieval and answer-level metrics against the evidence-linked QA set.

Default mode is a cheap structural smoke test. ``--judge`` enables an optional
LLM judge through the project's configured provider for faithfulness, answer
relevance, completeness, and citation correctness. Structural grounding is
reported separately and must not be called semantic faithfulness.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strategic_graphrag.response_contract import (
    classify_outcome,
    is_abstention,
    response_state,
)
from strategic_graphrag.evaluation.metric_spec import metric_registry, wilson_interval
MODES = ("vector", "graph", "hybrid", "hybrid_temporal")
JUDGE_PROMPT_VERSION = "answer-level-judge-v1"
SYNTHESIS_PROMPT_VERSION = "graph-rag-report-contract-v1"
SYNTHESIS_TEMPERATURE = 0.3


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 2)
    index = (len(values) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 2)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    return round(statistics.mean(usable), 4) if usable else None


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    """Return the harmonic mean of two already-aggregated metrics."""
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return round(2 * precision * recall / (precision + recall), 4)


def _bootstrap_ci(
    values: Iterable[Optional[float]],
    *,
    seed: int = 20260916,
    resamples: int = 2000,
) -> Optional[Dict[str, Any]]:
    """Return a deterministic percentile bootstrap CI for a question metric.

    This is deliberately a row-level engineering interval.  The report labels
    it explicitly because the current Gold set contains duplicate question
    texts and is too small for a publication-grade clustered estimate.
    """
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    if len(usable) == 1:
        point = round(usable[0], 4)
        return {
            "estimate": point,
            "lower": point,
            "upper": point,
            "n": 1,
            "resamples": 0,
            "seed": seed,
            "method": "deterministic percentile bootstrap; row-level",
        }
    rng = random.Random(seed)
    means = [
        statistics.mean(rng.choice(usable) for _ in usable)
        for _ in range(resamples)
    ]
    return {
        "estimate": round(statistics.mean(usable), 4),
        "lower": round(_percentile_raw(means, 2.5), 4),
        "upper": round(_percentile_raw(means, 97.5), 4),
        "n": len(usable),
        "resamples": resamples,
        "seed": seed,
        "method": "deterministic percentile bootstrap; row-level",
    }


def _percentile_raw(values: Iterable[float], percentile: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _post_query(base_url: str, payload: Dict[str, Any]) -> tuple[Any, float]:
    """POST one evaluation query while respecting the Demo rate limiter.

    The wait is returned separately so callers can exclude server-enforced
    backoff from engine latency. This keeps a full four-mode run reproducible
    under the default local limit.
    """
    waited_ms = 0.0
    for attempt in range(4):
        response = requests.post(
            f"{base_url.rstrip('/')}/query",
            json=payload,
            timeout=180,
        )
        if response.status_code != 429 or attempt == 3:
            return response, round(waited_ms, 2)
        retry_after = response.headers.get("Retry-After", "60")
        try:
            delay_seconds = min(max(float(retry_after), 1.0), 60.0)
        except (TypeError, ValueError):
            delay_seconds = 60.0
        time.sleep(delay_seconds)
        waited_ms += delay_seconds * 1000
    return response, round(waited_ms, 2)


def _gold_evidence_ids(item: Dict[str, Any]) -> Set[str]:
    """Read human gold IDs, while retaining compatibility with the old draft schema."""

    if "gold_evidence_ids" in item:
        values = item.get("gold_evidence_ids") or []
    else:
        values = item.get("evidence_claim_ids") or []
    return {str(value) for value in values if value}


def _gold_pages(item: Dict[str, Any]) -> Set[int]:
    """Read human gold pages, falling back to the legacy candidate field."""

    values = item.get("gold_pages") if "gold_pages" in item else item.get("pages")
    return {int(page) for page in (values or []) if str(page).isdigit()}


def _source_filing(item: Dict[str, Any]) -> Optional[str]:
    return item.get("source_filing") or item.get("candidate_source_filing")


def _answerable(item: Dict[str, Any]) -> bool:
    if item.get("answerable") is not None:
        return bool(item.get("answerable"))
    return bool(item.get("candidate_answerable", True))


def _is_abstention(result: Dict[str, Any]) -> bool:
    """Prefer the explicit outcome; use the shared legacy parser otherwise."""
    return is_abstention(result)


def _dataset_status(dataset: List[Dict[str, Any]]) -> str:
    statuses = {row.get("review_status") for row in dataset}
    if dataset and statuses == {"HUMAN_REVIEWED"}:
        return "HUMAN_REVIEWED"
    if dataset and statuses == {"HUMAN_REVIEWED_DERIVED_QUESTION_LEVEL"}:
        return "HUMAN_REVIEWED_DERIVED_QUESTION_LEVEL"
    if dataset and statuses == {"AUTO_GENERATED_REGRESSION_CANDIDATE"}:
        return "AUTO_GENERATED_REGRESSION_CANDIDATE_NOT_HUMAN_GOLD"
    return "MIXED_OR_UNREVIEWED"


def _collapse_question_variants(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a question-level view without editing candidate annotations.

    The human form reviews evidence candidates, so one question can appear
    more than once with different candidate passages and labels.  Answer-level
    evaluation should treat repeated question text as one unit and union the
    human-approved evidence variants instead of treating conflicting rows as
    independent questions.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in dataset:
        question = " ".join(str(row.get("question") or "").split())
        key = question.casefold()
        if key not in groups:
            grouped = dict(row)
            grouped.update({
                "question": question,
                "variant_ids": [],
                "variant_count": 0,
                "variant_label_conflict": False,
                "gold_evidence_ids": [],
                "gold_pages": [],
                "answerable": False,
                "requires_abstention": True,
                "reference_answer": "",
            })
            groups[key] = grouped
            order.append(key)

        grouped = groups[key]
        grouped["variant_ids"].append(str(row.get("id") or ""))
        grouped["variant_count"] += 1
        row_answerable = _answerable(row)
        if row_answerable != grouped["answerable"] and grouped["variant_count"] > 1:
            grouped["variant_label_conflict"] = True
        grouped["answerable"] = bool(grouped["answerable"] or row_answerable)
        grouped["requires_abstention"] = not grouped["answerable"]
        if row_answerable:
            grouped["gold_evidence_ids"] = sorted(
                set(grouped["gold_evidence_ids"]) | _gold_evidence_ids(row)
            )
            grouped["gold_pages"] = sorted(
                set(grouped["gold_pages"]) | _gold_pages(row)
            )
            reference = str(row.get("reference_answer") or "").strip()
            if reference and not grouped["reference_answer"]:
                grouped["reference_answer"] = reference

    result = []
    for key in order:
        grouped = groups[key]
        grouped["id"] = "QG-" + str(grouped["variant_ids"][0] or key[:8])
        grouped["review_status"] = "HUMAN_REVIEWED_DERIVED_QUESTION_LEVEL"
        result.append(grouped)
    return result


def _retrieved_ids(result: Dict[str, Any], *, include_variants: bool = False) -> Set[str]:
    """Return graph EvidenceClaim IDs surfaced by the response.

    A semantic path may be deduplicated while retaining alternative evidence
    variants.  The primary IDs are useful for strict top-path diagnostics;
    variants are also valid surfaced provenance and must be included in the
    Gold coverage view.
    """
    values: Set[str] = set()
    for path in result.get("paths", []) or []:
        values.update(
            str(evidence_id)
            for evidence_id in path.get("evidence_ids", []) or []
            if evidence_id
        )
        if include_variants:
            for hop_variants in path.get("evidence_variants", []) or []:
                values.update(
                    str(variant.get("evidence_id"))
                    for variant in hop_variants or []
                    if isinstance(variant, dict) and variant.get("evidence_id")
                )
    return values


def _retrieved_pages(
    result: Dict[str, Any],
    *,
    mode: str = "graph",
    include_variants: bool = False,
) -> Set[int]:
    """Return page numbers surfaced by graph paths or vector hits."""
    if mode == "vector":
        retrieval = (result.get("metadata") or {}).get("retrieval") or {}
        pages: Set[int] = set()
        for hit in retrieval.get("hits", []) or []:
            page = (hit.get("metadata") or {}).get("page")
            if str(page).isdigit():
                pages.add(int(page))
        return pages

    pages: Set[int] = set()
    for path in result.get("paths", []) or []:
        pages.update(
            int(page)
            for page in path.get("pages", []) or []
            if str(page).isdigit()
        )
        if include_variants:
            for hop_variants in path.get("evidence_variants", []) or []:
                pages.update(
                    int(variant.get("page"))
                    for variant in hop_variants or []
                    if isinstance(variant, dict) and str(variant.get("page")).isdigit()
                )
    return pages


def _judge_answer(
    question: str,
    context: str,
    answer: str,
    llm: Any,
    *,
    answerable: bool,
    reference_answer: str,
    gold_evidence_ids: Iterable[str],
    gold_pages: Iterable[int],
) -> Optional[Dict[str, Any]]:
    prompt = f"""Evaluate this evidence-grounded financial RAG answer for an academic benchmark. Return JSON only.
Score faithfulness, answer_relevance, completeness, and citation_correctness from 1 to 5.
For an unanswerable question, completeness must be null because there is no expected answer to complete.
Do not reward unsupported detail, confident tone, or stylistic fluency.

Definitions:
- faithfulness: every material factual statement in the answer is supported by the retrieved context.
- answer_relevance: the answer directly addresses the question and does not drift into unrelated context.
- completeness: for an answerable question, the answer covers the material facts in the reference answer without omitting the requested relation, entity, or time scope.
- citation_correctness: cited EvidenceClaim IDs and pages identify the context that supports the statements; missing, mismatched, or background-only citations score low.

QUESTION:
{question}

GOLD ANSWERABLE: {answerable}
REFERENCE ANSWER (evaluation reference, not a source):
{reference_answer or '[none — the Gold label requires abstention]'}

GOLD EVIDENCE IDS: {', '.join(str(value) for value in gold_evidence_ids) or '[none]'}
GOLD PAGES: {', '.join(str(value) for value in gold_pages) or '[none]'}

RETRIEVED CONTEXT:
{context[:12000]}

ANSWER:
{answer[:8000]}

JSON shape:
{{"faithfulness": 1, "answer_relevance": 1, "completeness": 1, "citation_correctness": 1, "justification": "short reason"}}
"""
    try:
        content = llm.chat_with_fallback(
            prompt=prompt,
            system_prompt="You are a strict academic RAG evaluator.",
            temperature=0.0,
            max_tokens=350,
        )
        raw = (content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()
        parsed = json.loads(raw)

        def score(name: str, *, nullable: bool = False) -> Optional[float]:
            value = parsed.get(name)
            if value is None and nullable:
                return None
            numeric = float(value)
            return round(max(1.0, min(5.0, numeric)), 2)

        return {
            "faithfulness": score("faithfulness"),
            "answer_relevance": score("answer_relevance"),
            "completeness": score("completeness", nullable=not answerable),
            "citation_correctness": score("citation_correctness"),
            "justification": str(parsed.get("justification", "")),
        }
    except Exception as exc:
        return {"error": type(exc).__name__}


def _judge_context(result: Dict[str, Any]) -> str:
    """Render evidence with claim IDs/pages so citation scoring is auditable."""
    lines: List[str] = []
    seen: Set[tuple[str, str, str]] = set()
    for path in result.get("paths", []) or []:
        for index, evidence in enumerate(path.get("evidence", []) or []):
            if not evidence:
                continue
            evidence_id = (path.get("evidence_ids", []) or [])[index] if index < len(path.get("evidence_ids", []) or []) else ""
            page = (path.get("pages", []) or [])[index] if index < len(path.get("pages", []) or []) else ""
            role = path.get("evidence_role", "BACKGROUND_CONTEXT")
            key = (str(evidence_id), str(page), str(evidence))
            if key not in seen:
                seen.add(key)
                lines.append(f"[EvidenceClaim: {evidence_id}; p.{page}; role={role}; primary] {evidence}")
        for hop_variants in path.get("evidence_variants", []) or []:
            for variant in hop_variants or []:
                if not isinstance(variant, dict) or not variant.get("evidence"):
                    continue
                evidence_id = str(variant.get("evidence_id") or "")
                page = str(variant.get("page") or "")
                evidence = str(variant.get("evidence"))
                role = variant.get("evidence_role") or path.get("evidence_role", "BACKGROUND_CONTEXT")
                key = (evidence_id, page, evidence)
                if key not in seen:
                    seen.add(key)
                    lines.append(f"[EvidenceClaim: {evidence_id}; p.{page}; role={role}; variant] {evidence}")
    if not lines:
        lines.extend(str(value) for value in result.get("evidence_sentences", []) or [] if value)
    return "\n".join(lines)


def evaluate(
    dataset: List[Dict[str, Any]],
    base_url: str,
    limit: Optional[int],
    judge: bool,
    modes: Iterable[str] = MODES,
    synthesize: bool = False,
    question_level: bool = False,
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    judge_llm = None
    if judge:
        from strategic_graphrag.llm_provider import get_llm

        judge_llm = get_llm()

    source_dataset_size = len(dataset)
    dataset = _collapse_question_variants(dataset) if question_level else dataset
    rows = dataset[:limit] if limit else dataset
    selected_modes = tuple(dict.fromkeys(str(mode).strip().lower() for mode in modes))
    invalid_modes = sorted(set(selected_modes) - set(MODES))
    if invalid_modes:
        raise ValueError(f"unsupported retrieval mode(s): {', '.join(invalid_modes)}")

    results_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    metrics_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in selected_modes:
        results = []
        for item in rows:
            started = time.perf_counter()
            row: Dict[str, Any] = {
                "id": item["id"],
                "question_type": item.get("question_type") or item.get("candidate_question_type"),
                "retrieval_mode": mode,
                "expected_answerable": _answerable(item),
                "expected_source_filing": _source_filing(item),
                "expected_gold_evidence_ids": sorted(_gold_evidence_ids(item)),
                "expected_gold_pages": sorted(_gold_pages(item)),
            }
            try:
                response, rate_limit_wait_ms = _post_query(
                    base_url,
                    {
                        "question": item["question"],
                        "max_paths": 5,
                        "retrieval_mode": mode,
                        "vector_top_k": 5,
                        "source_filing": _source_filing(item),
                        "synthesize": bool(synthesize),
                        "use_cache": False,
                    },
                )
                elapsed_ms = (time.perf_counter() - started) * 1000 - rate_limit_wait_ms
                row["http_status"] = response.status_code
                row["latency_ms"] = round(elapsed_ms, 2)
                row["rate_limit_wait_ms"] = round(rate_limit_wait_ms, 2)
                payload = response.json()
                state = response_state(payload, http_status=response.status_code)
                row.update({
                    "execution_status": state["execution_status"],
                    "answer_status": state["answer_status"],
                    "grounding_status": state["grounding_status"],
                    "status_provenance": state["status_provenance"],
                })
                row["outcome"] = classify_outcome(
                    payload,
                    http_status=response.status_code,
                )
                if state["execution_status"] != "SUCCEEDED":
                    row["error"] = {
                        "type": state["execution_status"],
                        "payload": payload,
                    }
                    results.append(row)
                    continue

                gold_ids = _gold_evidence_ids(item)
                gold_pages = _gold_pages(item)
                answerable = _answerable(item)
                primary_ids = _retrieved_ids(payload)
                surfaced_ids = _retrieved_ids(payload, include_variants=True)
                primary_pages = _retrieved_pages(payload, mode=mode)
                surfaced_pages = _retrieved_pages(payload, mode=mode, include_variants=True)
                overlap_ids = gold_ids & surfaced_ids
                primary_overlap_ids = gold_ids & primary_ids
                overlap_pages = gold_pages & surfaced_pages
                primary_overlap_pages = gold_pages & primary_pages
                retrieval_unit = "chunk" if mode == "vector" else "evidence_claim"
                row.update(
                    {
                        "answerable": answerable,
                        "retrieval_unit": retrieval_unit,
                        "gold_evidence_ids": sorted(gold_ids),
                        "retrieved_evidence_ids": sorted(primary_ids),
                        "surfaced_evidence_ids": sorted(surfaced_ids),
                        "evidence_recall": round(len(overlap_ids) / len(gold_ids), 4) if gold_ids else None,
                        "evidence_recall_primary": round(len(primary_overlap_ids) / len(gold_ids), 4) if gold_ids else None,
                        "evidence_precision": round(len(overlap_ids) / len(surfaced_ids), 4) if surfaced_ids else 0.0,
                        "evidence_precision_primary": round(len(primary_overlap_ids) / len(primary_ids), 4) if primary_ids else 0.0,
                        "gold_pages": sorted(gold_pages),
                        "retrieved_pages": sorted(primary_pages),
                        "surfaced_pages": sorted(surfaced_pages),
                        "page_recall": round(len(overlap_pages) / len(gold_pages), 4) if gold_pages else None,
                        "page_recall_primary": round(len(primary_overlap_pages) / len(gold_pages), 4) if gold_pages else None,
                        "page_precision": round(len(gold_pages & surfaced_pages) / len(surfaced_pages), 4) if surfaced_pages else 0.0,
                        "grounding_status": state["grounding_status"],
                        "outcome": row["outcome"],
                        "answer_trace": {
                            "answer": payload.get("answer", ""),
                            "structured_report_status": (payload.get("structured_report") or {}).get("status"),
                            "evidence_context": _judge_context(payload)[:12000],
                            "cited_evidence_ids": (payload.get("metadata", {}).get("grounding") or {}).get("cited_evidence_ids", []),
                            "cited_pages": (payload.get("metadata", {}).get("grounding") or {}).get("cited_pages", []),
                            "llm": (payload.get("metadata") or {}).get("llm"),
                            "reference_answer": item.get("reference_answer", ""),
                            "gold_evidence_ids": sorted(gold_ids),
                            "gold_pages": sorted(gold_pages),
                        },
                        "abstention_correct": (
                            (not answerable and _is_abstention(payload))
                            if synthesize and not answerable
                            else None
                        ),
                    }
                )
                if judge_llm is not None:
                    row["judge"] = _judge_answer(
                        item["question"],
                        row["answer_trace"]["evidence_context"],
                        payload.get("answer", ""),
                        judge_llm,
                        answerable=answerable,
                        reference_answer=str(item.get("reference_answer", "")),
                        gold_evidence_ids=sorted(gold_ids),
                        gold_pages=sorted(gold_pages),
                    )
            except Exception as exc:
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                row["outcome"] = "TIMEOUT" if isinstance(exc, requests.Timeout) else "DEPENDENCY_ERROR"
                row["execution_status"] = row["outcome"]
                row["answer_status"] = "NOT_REQUESTED"
                row["grounding_status"] = "NOT_EXECUTED"
                row["error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)

        results_by_mode[mode] = results
        answerable_rows = [row for row in results if row.get("answerable") and row.get("execution_status") == "SUCCEEDED"]
        evidence_rows = [row for row in answerable_rows if row.get("retrieval_unit") == "evidence_claim"]
        judge_rows = [row.get("judge") for row in answerable_rows if isinstance(row.get("judge"), dict) and "faithfulness" in row["judge"]]
        latency = [row["latency_ms"] for row in results if "latency_ms" in row]
        metrics_by_mode[mode] = {
            "answerable_questions": len(answerable_rows),
            "unsupported_questions": sum(not row.get("answerable") and row.get("execution_status") == "SUCCEEDED" for row in results),
            "error_count": sum(row.get("execution_status") != "SUCCEEDED" for row in results),
            "outcome_counts": {
                outcome: sum(row.get("outcome") == outcome for row in results)
                for outcome in sorted({row.get("outcome") for row in results if row.get("outcome")})
            },
            "successful_execution_questions": sum(
                row.get("execution_status") == "SUCCEEDED" for row in results
            ),
            "execution_success_rate": round(
                sum(row.get("execution_status") == "SUCCEEDED" for row in results) / len(results), 4
            ) if results else None,
            "evidence_recall": _mean([row.get("evidence_recall") for row in evidence_rows]),
            "evidence_recall_primary": _mean([row.get("evidence_recall_primary") for row in evidence_rows]),
            "evidence_precision": _mean([row.get("evidence_precision") for row in evidence_rows]),
            "evidence_precision_primary": _mean([row.get("evidence_precision_primary") for row in evidence_rows]),
            "page_recall": _mean([row.get("page_recall") for row in answerable_rows]),
            "page_recall_primary": _mean([row.get("page_recall_primary") for row in answerable_rows]),
            "page_precision": _mean([row.get("page_precision") for row in answerable_rows]),
            "faithfulness_structural_proxy": _mean([
                1.0 if row.get("grounding_status") == "VERIFIED" else 0.0
                for row in answerable_rows
                if row.get("retrieval_unit") == "evidence_claim"
            ]),
            "faithfulness_llm_1_to_5": _mean([row.get("faithfulness") for row in judge_rows]),
            "answer_relevance_llm_1_to_5": _mean([row.get("answer_relevance") for row in judge_rows]),
            "completeness_llm_1_to_5": _mean([row.get("completeness") for row in judge_rows]),
            "citation_correctness_llm_1_to_5": _mean([row.get("citation_correctness") for row in judge_rows]),
            "abstention_accuracy": (
                _mean([
                    1.0 if row.get("abstention_correct") else 0.0
                    for row in results
                    if row.get("answerable") is False
                ])
                if synthesize
                else None
            ),
            "latency_ms": {
                "p50": _percentile(latency, 50),
                "p95": _percentile(latency, 95),
                "mean": _mean(latency),
            },
        }
        metrics_by_mode[mode]["evidence_f1"] = _f1(
            metrics_by_mode[mode]["evidence_precision"],
            metrics_by_mode[mode]["evidence_recall"],
        )
        metrics_by_mode[mode]["evidence_f1_primary"] = _f1(
            metrics_by_mode[mode]["evidence_precision_primary"],
            metrics_by_mode[mode]["evidence_recall_primary"],
        )
        metric_values = {
            "faithfulness_llm_1_to_5": [row.get("faithfulness") for row in judge_rows],
            "answer_relevance_llm_1_to_5": [row.get("answer_relevance") for row in judge_rows],
            "completeness_llm_1_to_5": [row.get("completeness") for row in judge_rows],
            "citation_correctness_llm_1_to_5": [row.get("citation_correctness") for row in judge_rows],
            "abstention_accuracy": [
                1.0 if row.get("abstention_correct") else 0.0
                for row in results
                if row.get("answerable") is False
            ],
        }
        metrics_by_mode[mode]["confidence_intervals_95"] = {
            name: _bootstrap_ci(values)
            for name, values in metric_values.items()
        }
        execution_successes = sum(
            row.get("execution_status") == "SUCCEEDED" for row in results
        )
        metrics_by_mode[mode]["execution_success_interval_95"] = {
            "successes": execution_successes,
            "total": len(results),
            "interval": wilson_interval(execution_successes, len(results)),
            "method": "Wilson score interval",
        }

    return {
        "dataset": "golden_qa",
        "dataset_status": _dataset_status(dataset),
        "source_filing": "2025-10-K.pdf",
        "evaluation_protocol": (
            "four explicit retrieval modes against human-reviewed Gold QA; "
            + ("answer synthesis and optional LLM judging enabled" if synthesize else "retrieval-only structural metrics")
        ),
        "evaluation_unit": "question_text_group" if question_level else "candidate_evidence_row",
        "source_dataset_size": source_dataset_size,
        "dataset_size": len(dataset),
        "evaluated": len(rows),
        "modes": list(selected_modes),
        "judge_enabled": judge,
        "judge_config": ({
            "provider": getattr(judge_llm, "provider", None),
            "model": getattr(judge_llm, "default_model", None),
            "temperature": 0.0,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "scores": ["faithfulness", "answer_relevance", "completeness", "citation_correctness"],
        } if judge_llm is not None else None),
        "synthesis_config": {
            "enabled": bool(synthesize),
            "temperature": SYNTHESIS_TEMPERATURE,
            "prompt_version": SYNTHESIS_PROMPT_VERSION,
            "context_scope": "top-five graph paths; vector hits are excluded from synthesis; judge trace retains primary and evidence variants",
            "routes": sorted({
                (
                    str(((row.get("answer_trace") or {}).get("llm") or {}).get("success_provider") or "none"),
                    str(((row.get("answer_trace") or {}).get("llm") or {}).get("success_model") or "none"),
                )
                for mode_rows in results_by_mode.values()
                for row in mode_rows
                if row.get("answer_trace")
            }),
        },
        "metrics": metrics_by_mode,
        "metric_specs": metric_registry(),
        "rows": results_by_mode,
        "metric_notes": {
            "evidence_recall_precision": "Exact EvidenceClaim ID overlap against human Gold. Graph-family metrics include surfaced evidence_variants; primary metrics exclude variants and show the strict representative-path view.",
            "evidence_f1": "Harmonic mean of the corresponding macro evidence precision and recall; surfaced and strict primary views use matched units.",
            "page_recall_precision": "Page overlap against human Gold. Vector uses chunk-hit pages; graph-family uses surfaced path pages and evidence variants.",
            "faithfulness_structural_proxy": "Runtime citation/grounding check; not a substitute for semantic faithfulness.",
            "faithfulness_llm_1_to_5": "Optional configured-provider judge; run with --judge and report model/version.",
            "answer_relevance_llm_1_to_5": "Optional configured-provider judge; run with --judge and report model/version.",
            "completeness_llm_1_to_5": "Reference-answer coverage for answerable Gold rows; not computed for unanswerable rows.",
            "citation_correctness_llm_1_to_5": "Judge score for whether cited EvidenceClaim IDs/pages support the answer; inspect answer_trace for the evidence shown.",
            "latency": "End-to-end HTTP latency, including retrieval and answer synthesis; cold and warm runs should be separated for publication.",
            "abstention_accuracy": "Only computed when answer synthesis is enabled; retrieval-only traces are not answers and cannot support a formal abstention score.",
            "confidence_intervals_95": "Deterministic percentile bootstrap over evaluated rows (2,000 resamples, seed 20260916). Because the 30-row set contains duplicate question texts and only 7 answerable rows, these are engineering intervals, not publication-grade clustered confidence intervals.",
            "question_level_view": "When enabled, repeated question text is grouped. Answerability is true if any human-reviewed evidence variant is answerable; Gold IDs/pages are the union of the answerable variants. This derived view does not edit the original candidate-conditioned annotations.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the single-filing Golden QA set")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "evaluation" / "golden_qa_v2.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only the first N questions")
    parser.add_argument("--judge", action="store_true", help="Enable configured LLM judge for semantic metrics")
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Enable answer synthesis; --judge enables it automatically",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Retrieval modes to evaluate (default: all four)",
    )
    parser.add_argument(
        "--question-level",
        action="store_true",
        help="Group repeated question text into one answer-level unit while preserving the original candidate-conditioned dataset.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation" / "golden_qa_v2_results.json")
    args = parser.parse_args()
    report = evaluate(
        _load_jsonl(args.dataset),
        args.base_url,
        args.limit,
        args.judge,
        args.modes,
        synthesize=bool(args.synthesize or args.judge),
        question_level=bool(args.question_level),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluated": report["evaluated"], "modes": report["modes"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
