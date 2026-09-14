"""Evaluate four retrieval modes on a frozen automatic Silver benchmark.

Graph and vector retrieval expose different native identifiers, so the primary
comparison unit is the canonical filing-page key ``doc_id#page``. Exact
EvidenceClaim identifiers are retained as a secondary diagnostic for graph
and hybrid modes. This report is explicitly a Silver/proxy evaluation and must
not be presented as independent human QA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODES = ("vector", "graph", "hybrid", "hybrid_temporal")
DEFAULT_DATASET = ROOT / "evaluation" / "silver_retrieval_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "retrieval_benchmark_silver_2026-09-09.json"
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20260909


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _page_key(filing: Any, page: Any) -> str | None:
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        return None
    if page_number <= 0 or not filing:
        return None
    return f"{str(filing).removesuffix('.pdf')}#{page_number}"


def _expected_pages(item: dict[str, Any]) -> set[str]:
    explicit = item.get("expected_page_keys") or []
    if explicit:
        return {str(value) for value in explicit if value}
    filing = item.get("source_filing")
    return {
        key
        for page in item.get("expected_pages") or []
        if (key := _page_key(filing, page))
    }


def _vector_pages(result: dict[str, Any]) -> list[str]:
    retrieval = (result.get("metadata") or {}).get("retrieval") or {}
    pages: list[str] = []
    for hit in retrieval.get("hits") or []:
        metadata = hit.get("metadata") or {}
        key = _page_key(metadata.get("source_filing") or metadata.get("doc_id"), metadata.get("page"))
        if key and key not in pages:
            pages.append(key)
    return pages


def _graph_pages(result: dict[str, Any]) -> list[str]:
    pages: list[str] = []
    for path in result.get("paths") or []:
        filings = path.get("filings") or []
        path_pages = path.get("pages") or []
        for filing, page in zip(filings, path_pages):
            key = _page_key(filing, page)
            if key and key not in pages:
                pages.append(key)
    return pages


def _graph_evidence_ids(result: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for path in result.get("paths") or []:
        for evidence_id in path.get("evidence_ids") or []:
            if evidence_id and str(evidence_id) not in ids:
                ids.append(str(evidence_id))
    return ids


def _ranked_pages(mode: str, result: dict[str, Any]) -> list[str]:
    return _vector_pages(result) if mode == "vector" else _graph_pages(result)


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.mean(usable), 4) if usable else None


def _bootstrap_ci(
    values: Iterable[float | None],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any] | None:
    """Return a deterministic percentile bootstrap CI for a macro mean.

    The resampling unit is one question score. This interval is descriptive for
    the Silver regression and must not be interpreted as human-gold evidence.
    """
    if resamples < 1:
        raise ValueError("resamples must be positive")
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    rng = random.Random(seed)
    sample_size = len(usable)
    means = [
        statistics.mean(rng.choice(usable) for _ in range(sample_size))
        for _ in range(resamples)
    ]
    return {
        "lower": _percentile(means, 2.5),
        "upper": _percentile(means, 97.5),
        "confidence_level": 0.95,
        "unit": "question_id",
        "question_count": sample_size,
        "resamples": resamples,
        "seed": seed,
    }


def _mrr(ranked: list[str], relevant: set[str]) -> float:
    for rank, page in enumerate(ranked, start=1):
        if page in relevant:
            return round(1.0 / rank, 4)
    return 0.0


def _ndcg(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    actual = 0.0
    for rank, page in enumerate(ranked[:k], start=1):
        if page in relevant:
            actual += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(actual / ideal, 4) if ideal else 0.0


def _is_abstention(result: dict[str, Any]) -> bool:
    structured = result.get("structured_report") or {}
    status = str(structured.get("status") or "").upper()
    answer = str(result.get("answer") or "").upper()
    if status.startswith("INSUFFICIENT") or status in {"NEGATIVE_CLAIM_GUARD", "NO_HITS", "EMPTY"}:
        return True
    return any(marker in answer for marker in ("INSUFFICIENT EVIDENCE", "INSUFFICIENT_DIRECT_EVIDENCE", "GROUNDING FAILURE", "NO VECTOR CHUNKS"))


def _latency(result: dict[str, Any]) -> float | None:
    value = ((result.get("metadata") or {}).get("latency_ms") or {}).get("total_ms")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_record(item: dict[str, Any], mode: str, result: dict[str, Any], elapsed_ms: float, max_k: int) -> dict[str, Any]:
    relevant = _expected_pages(item)
    ranked_pages = _ranked_pages(mode, result)
    row: dict[str, Any] = {
        "id": item["id"],
        "question_type": item.get("question_type"),
        "answerable": bool(item.get("answerable")),
        "expected_page_keys": sorted(relevant),
        "ranked_page_keys": ranked_pages,
        "retrieved_evidence_ids": _graph_evidence_ids(result),
        "abstained": _is_abstention(result),
        "runtime_latency_ms": round(elapsed_ms, 2),
        "engine_latency_ms": _latency(result),
        "retrieval_status": ((result.get("metadata") or {}).get("retrieval") or {}).get("status"),
        "grounding_status": ((result.get("metadata") or {}).get("grounding") or {}).get("status"),
        "error": None,
    }
    if item.get("answerable"):
        row["first_relevant_rank"] = next((rank for rank, page in enumerate(ranked_pages, 1) if page in relevant), None)
        row["mrr"] = _mrr(ranked_pages, relevant)
        for k in (1, 3, 5, 10):
            top = ranked_pages[:k]
            overlap = len(set(top) & relevant)
            row[f"precision_at_{k}"] = round(overlap / k, 4)
            row[f"recall_at_{k}"] = round(overlap / len(relevant), 4) if relevant else None
            row[f"ndcg_at_{k}"] = _ndcg(ranked_pages, relevant, k)
    else:
        row["abstention_correct"] = bool(item.get("requires_abstention")) == row["abstained"]
    return row


def _metrics(
    rows: list[dict[str, Any]],
    max_k: int,
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    answerable = [row for row in rows if row.get("answerable") and not row.get("error")]
    unsupported = [row for row in rows if not row.get("answerable") and not row.get("error")]
    output: dict[str, Any] = {
        "answerable_questions": len(answerable),
        "unsupported_questions": len(unsupported),
        "error_count": sum(bool(row.get("error")) for row in rows),
        "precision_at_k": {},
        "recall_at_k": {},
        "ndcg_at_k": {},
        "mrr": _mean([row.get("mrr") for row in answerable]),
        "abstention_accuracy": _mean([1.0 if row.get("abstention_correct") else 0.0 for row in unsupported]),
        "latency_ms": {
            "mean": _mean([row.get("runtime_latency_ms") for row in rows]),
            "p50": _percentile([row.get("runtime_latency_ms") for row in rows], 50),
            "p95": _percentile([row.get("runtime_latency_ms") for row in rows], 95),
        },
        "bootstrap_ci_95": {
            "precision_at_k": {},
            "recall_at_k": {},
            "ndcg_at_k": {},
            "mrr": _bootstrap_ci(
                [row.get("mrr") for row in answerable],
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            ),
        },
    }
    for k in (1, 3, 5, 10):
        output["precision_at_k"][str(k)] = _mean([row.get(f"precision_at_{k}") for row in answerable])
        output["recall_at_k"][str(k)] = _mean([row.get(f"recall_at_{k}") for row in answerable])
        output["ndcg_at_k"][str(k)] = _mean([row.get(f"ndcg_at_{k}") for row in answerable])
        output["bootstrap_ci_95"]["precision_at_k"][str(k)] = _bootstrap_ci(
            [row.get(f"precision_at_{k}") for row in answerable],
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        output["bootstrap_ci_95"]["recall_at_k"][str(k)] = _bootstrap_ci(
            [row.get(f"recall_at_{k}") for row in answerable],
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        output["bootstrap_ci_95"]["ndcg_at_k"][str(k)] = _bootstrap_ci(
            [row.get(f"ndcg_at_{k}") for row in answerable],
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
    return output


def _numeric_score(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _paired_comparison(
    rows_by_mode: dict[str, list[dict[str, Any]]],
    mode_a: str,
    mode_b: str,
    metric: str,
) -> dict[str, Any]:
    """Compare two modes on the same answerable question IDs."""
    rows_a = {str(row.get("id")): row for row in rows_by_mode.get(mode_a, [])}
    rows_b = {str(row.get("id")): row for row in rows_by_mode.get(mode_b, [])}
    common_ids = sorted(set(rows_a) & set(rows_b))
    paired: list[tuple[float, float]] = []
    for question_id in common_ids:
        row_a, row_b = rows_a[question_id], rows_b[question_id]
        if row_a.get("error") or row_b.get("error"):
            continue
        if not row_a.get("answerable") or not row_b.get("answerable"):
            continue
        score_a = _numeric_score(row_a, metric)
        score_b = _numeric_score(row_b, metric)
        if score_a is not None and score_b is not None:
            paired.append((score_a, score_b))

    mode_a_wins = sum(score_a > score_b for score_a, score_b in paired)
    mode_b_wins = sum(score_b > score_a for score_a, score_b in paired)
    ties = len(paired) - mode_a_wins - mode_b_wins
    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "metric": metric,
        "unit_of_comparison": "question_id",
        "eligible_questions": len(paired),
        "excluded_questions": len(common_ids) - len(paired),
        "win_tie_loss": {
            "mode_a_wins": mode_a_wins,
            "ties": ties,
            "mode_b_wins": mode_b_wins,
        },
        "mean_score": {
            mode_a: _mean(score_a for score_a, _ in paired),
            mode_b: _mean(score_b for _, score_b in paired),
        },
        "mean_delta_mode_b_minus_mode_a": _mean(score_b - score_a for score_a, score_b in paired),
    }


def _stratified_metrics(
    rows_by_mode: dict[str, list[dict[str, Any]]],
    max_k: int,
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, dict[str, dict[str, Any]]]:
    question_types = sorted(
        {
            row.get("question_type")
            for mode_rows in rows_by_mode.values()
            for row in mode_rows
        },
        key=lambda value: str(value),
    )
    return {
        mode: {
            question_type: _metrics(
                [row for row in mode_rows if row.get("question_type") == question_type],
                max_k,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            )
            for question_type in question_types
        }
        for mode, mode_rows in rows_by_mode.items()
    }


def _percentile(values: Iterable[float | None], percentile: float) -> float | None:
    usable = sorted(float(value) for value in values if value is not None)
    if not usable:
        return None
    if len(usable) == 1:
        return round(usable[0], 2)
    position = (len(usable) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(usable) - 1)
    weight = position - lower
    return round(usable[lower] * (1 - weight) + usable[upper] * weight, 2)


def evaluate(
    dataset: list[dict[str, Any]],
    top_k: int,
    limit: int | None,
    offline_anchors: bool,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    if offline_anchors:
        # Keep the first benchmark focused on the four retrieval algorithms.
        # Optional Cross-Encoder reranking is a separate ablation and can make
        # a CPU-only run extremely slow while adding another variable.
        os.environ["CROSS_ENCODER_ENABLED"] = "false"
    from strategic_graphrag.engine.graph_rag_engine import GraphRAGEngine
    from strategic_graphrag.engine.vector_rag_baseline import VectorRAGBaseline

    rows = dataset[:limit] if limit else dataset
    engine = GraphRAGEngine()
    vector = VectorRAGBaseline()
    if offline_anchors:
        # The generated questions contain canonical entity tokens. Disabling
        # optional LLM anchor expansion keeps retrieval scoring deterministic
        # and avoids turning a retrieval benchmark into an LLM-cost benchmark.
        engine._has_llm = False
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        for item in rows:
            for mode in MODES:
                started = time.perf_counter()
                try:
                    result = engine.query(
                        item["question"],
                        top_k=max(top_k, 10),
                        source_filing=item.get("source_filing"),
                        cross_filing=bool(item.get("cross_filing")),
                        retrieval_mode=mode,
                        vector_engine=vector,
                        vector_top_k=max(top_k, 10),
                        synthesize=False,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    results[mode].append(_score_record(item, mode, result, elapsed_ms, top_k))
                except Exception as exc:
                    results[mode].append({
                        "id": item["id"],
                        "question_type": item.get("question_type"),
                        "answerable": bool(item.get("answerable")),
                        "error": f"{type(exc).__name__}: {exc}",
                        "runtime_latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    })
    finally:
        engine.close()

    return {
        "schema": "strategic-graphrag-retrieval-benchmark/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_status": "AUTO_GENERATED_SILVER_NOT_HUMAN_GOLD",
        "dataset_size": len(dataset),
        "evaluated_questions": len(rows),
        "modes": list(MODES),
        "unit_of_comparison": "canonical_filing_page_key",
        "paired_comparison_unit": "question_id",
        "bootstrap_unit": "question_id",
        "run_config": {
            "top_k": top_k,
            "metric_cutoffs": [1, 3, 5, 10],
            "bootstrap": {
                "method": "percentile bootstrap over per-question scores",
                "confidence_level": 0.95,
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
            },
            "synthesize": False,
            "llm_anchor_assist_disabled": offline_anchors,
            "cross_encoder_disabled": offline_anchors,
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_model": os.getenv("LLM_MODEL") or os.getenv("LLM_QUERY_MODEL"),
            "embedding_model": os.getenv("GRAPH_EMBEDDING_MODEL"),
        },
        "metrics": {
            mode: _metrics(
                mode_rows,
                top_k,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            )
            for mode, mode_rows in results.items()
        },
        "by_question_type": _stratified_metrics(
            results,
            top_k,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "paired_comparisons": {
            metric: {
                f"{mode_a}_vs_{mode_b}": _paired_comparison(results, mode_a, mode_b, metric)
                for mode_a, mode_b in combinations(MODES, 2)
            }
            for metric in ("recall_at_5", "ndcg_at_5")
        },
        "rows": {mode: mode_rows for mode, mode_rows in results.items()},
        "dataset_notes": {
            "label_source": "auto-generated Silver from graph-derived strict EvidenceClaims and deterministic unsupported questions",
            "self_test_bias": "Expected pages are derived from the strict graph under test, so retrieval scores are proxy regression measurements and can inherit graph construction bias.",
            "human_gold_status": "missing; no independent human-reviewed Golden QA set is present",
        },
        "limitations": [
            "Expected evidence is generated from the same strict graph under test.",
            "Page-level comparison is fair across vector and graph modes but is not exact EvidenceClaim retrieval.",
            "The unsupported set is deterministic and synthetic, not human-authored Golden QA.",
            "LLM anchor expansion and answer synthesis are disabled for the reproducible retrieval-only run.",
            "Bootstrap intervals are descriptive uncertainty summaries over question-level scores, not human-gold confidence or a proof of superiority.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vector, graph, hybrid, and hybrid-temporal retrieval")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--allow-llm-anchors", action="store_true", help="Include optional remote LLM anchor expansion")
    args = parser.parse_args()
    report = evaluate(
        _load_jsonl(args.dataset),
        args.top_k,
        args.limit,
        not args.allow_llm_anchors,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
