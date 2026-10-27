"""Summarize raw retrieval timings and PDF-audit signals without inventing QA accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_METHODS = (
    "keyword_bm25",
    "dense_semantic",
    "keyword_dense_fusion_rrf",
    "fusion_graph_expansion",
    "fusion_graph_expansion_temporal",
    "fusion_temporal_only",
)
FROZEN_INPUT_KEYS = {"runner", "query_parser", "summarizer", "protocol", "dataset", "dependency_lock"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], p: float) -> float | str:
    if not values:
        return "NOT_RUN"
    if p == 0.5:
        return round(statistics.median(values), 3)
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(p * len(ordered)) - 1)], 3)


def _cluster_ci(values_by_family: dict[str, list[float]], *, seed: int = 20260924, iterations: int = 2000) -> dict[str, Any]:
    families = sorted(values_by_family)
    if not families:
        return {"status": "NOT_RUN_NO_LABELED_FAMILIES", "n_families": 0}
    family_means = [statistics.mean(values_by_family[family]) for family in families]
    estimate = statistics.mean(family_means)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        drawn = [rng.choice(family_means) for _ in family_means]
        samples.append(statistics.mean(drawn))
    samples.sort()
    lo = samples[max(0, math.floor(0.025 * iterations))]
    hi = samples[min(iterations - 1, math.ceil(0.975 * iterations) - 1)]
    return {
        "status": "DESCRIPTIVE_FAMILY_CLUSTER_BOOTSTRAP",
        "n_families": len(families),
        "seed": seed,
        "iterations": iterations,
        "family_weighted_estimate": round(estimate, 4),
        "ci_95": [round(lo, 4), round(hi, 4)],
        "interpretation": "Development-only uncertainty interval; labels are AI/PDF source diagnostics, not human Gold or an independent test.",
    }


def _table_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def count(field: str, predicate) -> int:
        return sum(bool(predicate(str(row.get(field) or ""))) for row in rows)

    joint = Counter(str(row.get("ai_joint_diagnostic_status") or "NOT_RECORDED") for row in rows)
    return {
        "tier": "AI_PDF_VISUAL_DIAGNOSTIC_NOT_HUMAN_REVIEW",
        "candidate_rows": len(rows),
        "fields": {
            "numeric_cell_visual_match": {"n": count("numeric_value_cell_visual_status", lambda x: x == "VISUAL_MATCH"), "denominator": len(rows)},
            "sign_visual_match": {"n": count("sign_visual_status", lambda x: x.startswith("VISUAL_MATCH")), "denominator": len(rows)},
            "unit_scale_visual_match": {"n": count("unit_scale_visual_status", lambda x: x == "VISUAL_MATCH"), "denominator": len(rows)},
            "year_column_visual_match": {"n": count("fiscal_year_column_visual_status", lambda x: x == "VISUAL_MATCH"), "denominator": len(rows)},
            "metric_semantics_alias_match": {"n": count("metric_semantics_visual_status", lambda x: x == "VISUAL_ROW_LABEL_OR_NORMALIZED_ALIAS_MATCH"), "denominator": len(rows)},
            "source_physical_page_visual_match": {"n": count("source_page_visual_status", lambda x: x == "VISUAL_MATCH"), "denominator": len(rows)},
        },
        "joint_ai_diagnostic_counts_not_accuracy": dict(sorted(joint.items())),
        "human_primary_review": "NOT_RUN",
        "human_secondary_review": "NOT_RUN",
        "adjudication": "NOT_RUN",
        "interpretation": "These are AI visual review statuses, not estimated accuracy. Duplicate-key rows remain preserved; repeated facts can be multiple source items.",
    }


def summarize(raw_path: Path, table_path: Path, dataset_path: Path) -> dict[str, Any]:
    raw = _jsonl(raw_path)
    table = _jsonl(table_path)
    run_manifest_path = raw_path.with_suffix(".manifest.json")
    if not run_manifest_path.is_file():
        raise ValueError(f"Raw-run manifest is required for reproducible scoring: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    raw_hash = _sha256(raw_path)
    if run_manifest.get("raw_jsonl_sha256") != raw_hash:
        raise ValueError("Raw JSONL SHA-256 does not match its run manifest")
    if int(run_manifest.get("records", -1)) != len(raw):
        raise ValueError("Raw request count does not match the run manifest")
    if _sha256(dataset_path) != run_manifest.get("candidate_dataset_sha256"):
        raise ValueError("Development dataset SHA-256 does not match the run manifest")
    frozen_before = run_manifest.get("frozen_input_sha256_before") or {}
    frozen_after = run_manifest.get("frozen_input_sha256_after") or {}
    if not FROZEN_INPUT_KEYS.issubset(frozen_before) or frozen_before != frozen_after:
        raise ValueError("Run manifest lacks a complete, stable pre/post freeze hash set")
    if tuple(run_manifest.get("matrix_modes") or ()) != EXPECTED_METHODS:
        raise ValueError("Run manifest does not declare the fixed six-method matrix")
    if run_manifest.get("limited_smoke_run") is not False:
        raise ValueError("A smoke/partial run cannot be scored as the full frozen development matrix")
    current_source_paths = {
        "runner": Path(__file__).resolve().parents[1] / "scripts/run_financial_retrieval_matrix.py",
        "query_parser": Path(__file__).resolve().parents[1] / "strategic_graphrag/engine/query_understanding.py",
        "summarizer": Path(__file__).resolve(),
        "protocol": Path(__file__).resolve().parents[1] / "docs/financial_qa_candidate_protocol_v4.md",
        "dependency_lock": Path(__file__).resolve().parents[1] / "requirements-lock-2026-09-19.txt",
    }
    for key, path in current_source_paths.items():
        if frozen_before.get(key) != _sha256(path):
            raise ValueError(f"Current {key} file differs from the file frozen for this run")
    if frozen_before.get("dataset") != _sha256(dataset_path):
        raise ValueError("Current development dataset differs from the dataset frozen for this run")
    for field, frozen_key in (
        ("runner_sha256", "runner"),
        ("protocol_sha256", "protocol"),
        ("candidate_dataset_sha256", "dataset"),
        ("dependency_lock_sha256", "dependency_lock"),
    ):
        if run_manifest.get(field) != frozen_before[frozen_key]:
            raise ValueError(f"Run manifest {field} conflicts with its frozen source hash")

    family_rows = _jsonl(dataset_path)
    if not family_rows or any(row.get("partition") != "development" for row in family_rows):
        raise ValueError("Scoring requires the non-empty, development-only frozen dataset")
    expected_rows: list[dict[str, Any]] = []
    for family in family_rows:
        base = {key: value for key, value in family.items() if key != "candidate_variants"}
        expected_rows.append(base)
        for variant in family.get("candidate_variants") or []:
            expected_rows.append({**base, **variant, "family_id": family["family_id"]})
    expected_by_item = {str(row["item_id"]): row for row in expected_rows}
    if len(expected_by_item) != len(expected_rows):
        raise ValueError("Frozen dataset contains duplicate item IDs")
    expected_family_ids = sorted({str(row["family_id"]) for row in family_rows})
    if (
        int(run_manifest.get("candidate_question_count", -1)) != len(expected_rows)
        or int(run_manifest.get("candidate_family_count", -1)) != len(family_rows)
        or list(run_manifest.get("selected_family_ids") or []) != expected_family_ids
    ):
        raise ValueError("Run manifest scope/count does not match the frozen development dataset")

    request_ids = [str(row.get("request_id") or f"{row.get('item_id')}:{row.get('method')}") for row in raw]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Raw JSONL contains duplicate request IDs")
    expected_methods = set(EXPECTED_METHODS)
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in raw:
        item_id = str(row.get("item_id") or "")
        method = str(row.get("method") or "")
        if not item_id or not method or method in by_item[item_id]:
            raise ValueError("Raw JSONL has a missing item/method or duplicate method for one item")
        by_item[item_id][method] = row
    if set(by_item) != set(expected_by_item):
        missing = sorted(set(expected_by_item) - set(by_item))
        extra = sorted(set(by_item) - set(expected_by_item))
        raise ValueError(f"Raw schedule differs from frozen dataset; missing={missing[:5]}, extra={extra[:5]}")
    invariant_fields = (
        "family_id", "question", "question_type", "partition", "build_id",
        "gold_page_grades", "retrieval_scoring_status", "label_tier",
        "answerability_label", "reference_answer", "candidate_filing_hint", "candidate_pages_hint",
    )
    for item_id, methods_for_item in by_item.items():
        if set(methods_for_item) != expected_methods:
            raise ValueError(f"Incomplete fixed six-method schedule for item_id={item_id}")
        reference = next(iter(methods_for_item.values()))
        expected = expected_by_item[item_id]
        expected_values = {
            "family_id": expected.get("family_id"),
            "question": expected.get("question"),
            "question_type": expected.get("question_type"),
            "partition": expected.get("partition"),
            "build_id": run_manifest.get("build_id"),
            "gold_page_grades": expected.get("gold_page_grades", {}),
            "retrieval_scoring_status": expected.get("retrieval_scoring_status", "NOT_RUN"),
            "label_tier": expected.get("label_tier", "NOT_REVIEWED"),
            "answerability_label": expected.get("answerability_label", "NOT_REVIEWED"),
            "reference_answer": expected.get("reference_answer", "NOT_AVAILABLE_PENDING_PDF_REVIEW"),
            "candidate_filing_hint": expected.get("candidate_filing"),
            "candidate_pages_hint": expected.get("candidate_pages"),
        }
        for method_row in methods_for_item.values():
            for field in invariant_fields:
                if method_row.get(field) != reference.get(field) or method_row.get(field) != expected_values[field]:
                    raise ValueError(f"Request labels differ across methods for item_id={item_id}, field={field}")

    modes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        modes[row["method"]].append(row)
    method_names = {row["method"]: str(row.get("method_name_zh") or row["method"]) for row in raw}
    latency: dict[str, dict[str, Any]] = {}
    for mode, rows in sorted(modes.items()):
        all_values = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
        success_values = [float(row["latency_ms"]) for row in rows if row.get("execution_status") == "SUCCESS" and row.get("latency_ms") is not None]
        latency[mode] = {
            "method_name_zh": method_names[mode],
            "n_scheduled": len(rows),
            "n_success": len(success_values),
            "n_failed": len(rows) - len(success_values),
            "success_rate_all_requests": round(len(success_values) / len(rows), 4) if rows else "NOT_RUN",
            "p50_ms_all_requests": _percentile(all_values, 0.50),
            "p95_ms_all_requests": _percentile(all_values, 0.95),
            "p50_ms_success_subset": _percentile(success_values, 0.50),
            "p95_ms_success_subset": _percentile(success_values, 0.95),
            "result_statuses": dict(Counter(str(row.get("result_status")) for row in rows)),
        }

    candidate_pool_by_method: dict[str, dict[str, Any]] = {}
    pool_fields = (
        "bm25_retrieved_chunks", "bm25_unique_candidate_pages",
        "dense_retrieved_chunks", "dense_unique_candidate_pages",
        "fusion_unique_candidate_pages", "graph_seed_pages",
        "graph_expansion_candidate_pages", "graph_added_candidate_pages",
        "temporal_candidate_pages_before_filter", "temporal_candidate_pages_after_filter",
        "returned_unique_pages",
    )
    for mode, rows in sorted(modes.items()):
        valid = [row.get("candidate_counts") for row in rows if isinstance(row.get("candidate_counts"), dict)]
        candidate_pool_by_method[mode] = {
            "method_name_zh": method_names[mode],
            "requests_with_candidate_counts": len(valid),
            "counts": {
                field: {
                    "n": len(values),
                    "min": min(values) if values else "NOT_RUN",
                    "max": max(values) if values else "NOT_RUN",
                    "mean": round(statistics.mean(values), 3) if values else "NOT_RUN",
                }
                for field in pool_fields
                for values in [[int(entry[field]) for entry in valid if isinstance(entry.get(field), (int, float))]]
            },
        }

    direct_scored = [
        row for row in raw
        if row.get("retrieval_scoring_status") == "SCORED_KNOWN_SUPPORT_PAGES"
        and isinstance(row.get("gold_page_grades"), dict)
        and any(int(grade) > 0 for grade in row["gold_page_grades"].values())
    ]
    context_scored = [
        row for row in raw
        if row.get("retrieval_scoring_status") == "SCORED_KNOWN_CONTEXT_PAGES_ONLY"
        and isinstance(row.get("gold_page_grades"), dict)
        and any(int(grade) > 0 for grade in row["gold_page_grades"].values())
    ]

    def _score_scope(eligible: list[dict[str, Any]], *, direct_support: bool) -> dict[str, Any]:
        label_scope = Counter(str(row.get("retrieval_scoring_status")) for row in eligible)
        metrics_by_k: dict[str, Any] = {}
        for k in (1, 3, 5, 10):
            hits: list[float] = []
            page_coverages: list[float] = []
            page_hits = page_denominator = 0
            family_hit: dict[str, list[float]] = defaultdict(list)
            family_coverage: dict[str, list[float]] = defaultdict(list)
            for row in eligible:
                relevant = {str(page) for page, grade in row["gold_page_grades"].items() if int(grade) > 0}
                ranked = [str(item.get("page_key") or "") for item in row.get("ranked_evidence") or []]
                matched = relevant.intersection(ranked[:k])
                hit = float(bool(matched))
                coverage = len(matched) / len(relevant) if relevant else 0.0
                hits.append(hit)
                page_coverages.append(coverage)
                page_hits += len(matched)
                page_denominator += len(relevant)
                family_hit[str(row["family_id"])].append(hit)
                family_coverage[str(row["family_id"])].append(coverage)
            metrics_by_k[str(k)] = {
                "labelled_page_hit": {"n": int(sum(hits)), "denominator": len(hits), "rate": round(statistics.mean(hits), 4) if hits else "NOT_RUN"},
                "labelled_page_coverage_mean_per_request": round(statistics.mean(page_coverages), 4) if page_coverages else "NOT_RUN",
                "labelled_page_micro_coverage": {"n": page_hits, "denominator": page_denominator, "rate": round(page_hits / page_denominator, 4) if page_denominator else "NOT_RUN"},
                "non_exhaustive_judged_direct_support_page_coverage": {"n": page_hits, "denominator": page_denominator, "rate": round(page_hits / page_denominator, 4) if page_denominator else "NOT_RUN", "interpretation": "Only the listed judged direct-support pages; not full-corpus Recall@k and not a mathematical lower bound."} if direct_support else "NOT_APPLICABLE_CONTEXT_ONLY",
                "family_cluster_ci_page_hit": _cluster_ci(family_hit),
                "family_cluster_ci_page_coverage": _cluster_ci(family_coverage),
            }
        base = {
            "labelled_requests": len(eligible),
            "unique_families": len({str(row["family_id"]) for row in eligible}),
            "label_scope_counts": dict(label_scope),
        }
        if not direct_support:
            return {
                **base,
                "context_only_page_diagnostic_at_k": metrics_by_k,
                "interpretation": "Context-only pages are not direct answer evidence and are excluded from support-quality headline and paired metrics.",
            }
        by_family_mrr: dict[str, list[float]] = defaultdict(list)
        for row in eligible:
            relevant = {str(page) for page, grade in row["gold_page_grades"].items() if int(grade) > 0}
            ranks = [rank for rank, item in enumerate(row.get("ranked_evidence") or [], start=1) if str(item.get("page_key") or "") in relevant]
            by_family_mrr[str(row["family_id"])].append(1.0 / min(ranks) if ranks else 0.0)
        return {
            **base,
            "direct_support_hit_and_page_coverage_at_k": metrics_by_k,
            "direct_support_MRR": {
                "n_requests": len(eligible),
                "denominator_families": len(by_family_mrr),
                "family_weighted_mean": round(statistics.mean(statistics.mean(values) for values in by_family_mrr.values()), 4) if by_family_mrr else "NOT_RUN",
                "family_cluster_ci": _cluster_ci(by_family_mrr),
                "interpretation": "MRR to explicitly labelled direct-support page only; unjudged pages may also be relevant.",
            },
            "nDCG": "NOT_COMPUTED_INCOMPLETE_RELEVANCE_POOL_UNJUDGED_PAGES_ARE_NOT_IRRELEVANT",
            "canonical_recall_claim": "NOT_SUPPORTED_LABELS_ARE_NON_EXHAUSTIVE",
        }

    quality: dict[str, dict[str, Any]] = {}
    for mode in sorted(modes):
        direct = _score_scope([row for row in direct_scored if row["method"] == mode], direct_support=True)
        context = _score_scope([row for row in context_scored if row["method"] == mode], direct_support=False)
        quality[mode] = {"method_name_zh": method_names[mode], **direct, "context_only_diagnostic": context}

    quality_slices: dict[str, Any] = {}
    for mode in sorted(modes):
        method_rows = [row for row in direct_scored if row["method"] == mode]
        groups: dict[str, dict[str, list[dict[str, Any]]]] = {
            "question_type": defaultdict(list),
            "fact_year": defaultdict(list),
        }
        for row in method_rows:
            groups["question_type"][str(row.get("question_type") or "unclassified")].append(row)
            plan = row.get("query_plan") or {}
            years = sorted({str(year) for year in plan.get("fiscal_years", []) if str(year).isdigit()})
            for year in years or ["NOT_PARSED"]:
                groups["fact_year"][year].append(row)

        def _slice_metrics(slice_rows: list[dict[str, Any]]) -> dict[str, Any]:
            family_hits: dict[str, list[float]] = defaultdict(list)
            hit_count = page_hit_count = page_total = 0
            for row in slice_rows:
                relevant = {str(page) for page, grade in row["gold_page_grades"].items() if int(grade) > 0}
                returned = {str(item.get("page_key") or "") for item in (row.get("ranked_evidence") or [])[:10]}
                hit = float(bool(relevant.intersection(returned)))
                hit_count += int(hit)
                page_hit_count += len(relevant.intersection(returned))
                page_total += len(relevant)
                family_hits[str(row["family_id"])].append(hit)
            return {
                "n_direct_support_requests": len(slice_rows),
                "n_semantic_families": len(family_hits),
                "known_direct_support_hit_at_10": {"n": hit_count, "denominator": len(slice_rows), "rate": round(hit_count / len(slice_rows), 4) if slice_rows else "NOT_RUN"},
                "known_page_coverage": {"n": page_hit_count, "denominator": page_total, "rate": round(page_hit_count / page_total, 4) if page_total else "NOT_RUN"},
                "family_cluster_ci_hit_at_10": _cluster_ci(family_hits),
            }

        quality_slices[mode] = {
            "method_name_zh": method_names[mode],
            "by_question_type": {key: _slice_metrics(value) for key, value in sorted(groups["question_type"].items())},
            "by_fact_year_overlapping_membership": {key: _slice_metrics(value) for key, value in sorted(groups["fact_year"].items())},
            "fact_year_note": "A multi-year question appears in each explicitly parsed fact-year slice; slices overlap and must not be summed.",
        }

    paired_graph_delta: dict[str, Any] = {}
    for graph_mode, base_mode in (
        ("fusion_graph_expansion", "keyword_dense_fusion_rrf"),
        ("fusion_graph_expansion_temporal", "fusion_temporal_only"),
    ):
        graph_values: dict[str, list[float]] = defaultdict(list)
        base_values: dict[str, list[float]] = defaultdict(list)
        for row in direct_scored:
            gold = {str(page) for page, grade in row["gold_page_grades"].items() if int(grade) > 0}
            top = {str(item.get("page_key") or "") for item in (row.get("ranked_evidence") or [])[:10]}
            value = float(bool(gold.intersection(top)))
            target = graph_values if row["method"] == graph_mode else base_values if row["method"] == base_mode else None
            if target is not None:
                target[str(row["family_id"])].append(value)
        paired = sorted(set(graph_values).intersection(base_values))
        deltas = {family: [statistics.mean(graph_values[family]) - statistics.mean(base_values[family])] for family in paired}
        paired_graph_delta[f"{method_names.get(graph_mode, graph_mode)} minus {method_names.get(base_mode, base_mode)}"] = {
            "metric": "family-level direct-support page-hit@10 difference",
            "n_paired_families": len(paired),
            "family_cluster_ci": _cluster_ci(deltas),
            "note": "Diagnostic association only; AI/PDF development labels and incomplete page judgments do not establish superiority. Graph and no-graph time arms have different candidate pools.",
        }

    by_item_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in raw:
        by_item_method[str(row["item_id"])][str(row["method"])] = row

    def _support_ranks(row: dict[str, Any] | None) -> dict[str, int]:
        if row is None or row.get("retrieval_scoring_status") != "SCORED_KNOWN_SUPPORT_PAGES":
            return {}
        gold = {
            str(page) for page, grade in (row.get("gold_page_grades") or {}).items()
            if int(grade) > 0
        }
        return {
            str(hit.get("page_key")): int(hit.get("rank") or rank)
            for rank, hit in enumerate(row.get("ranked_evidence") or [], start=1)
            if str(hit.get("page_key") or "") in gold
        }

    graph_regressions = []
    time_vs_graph_time_regressions = []
    graph_added_requests = 0
    graph_added_pages = 0
    graph_result_page_slots = 0
    for item_id, methods in sorted(by_item_method.items()):
        graph_row = methods.get("fusion_graph_expansion")
        if graph_row:
            graph_result_page_slots += len(graph_row.get("ranked_evidence") or [])
            additions = [
                str(hit.get("page_key")) for hit in graph_row.get("ranked_evidence") or []
                if hit.get("graph_expansion")
            ]
            graph_added_pages += len(additions)
            graph_added_requests += int(bool(additions))
        fused_row = methods.get("keyword_dense_fusion_rrf")
        fused_ranks, graph_ranks = _support_ranks(fused_row), _support_ranks(graph_row)
        lost_known_support = {page: rank for page, rank in fused_ranks.items() if rank <= 10 and page not in graph_ranks}
        if lost_known_support:
            graph_regressions.append({
                "item_id": item_id,
                "family_id": (fused_row or graph_row or {}).get("family_id"),
                "question": (fused_row or graph_row or {}).get("question"),
                "known_support_pages_lost_from_top10": lost_known_support,
                "fusion_top_pages": [hit.get("page_key") for hit in (fused_row or {}).get("ranked_evidence", [])[:10]],
                "graph_top_pages": [
                    {"page_key": hit.get("page_key"), "graph_expansion": bool(hit.get("graph_expansion"))}
                    for hit in (graph_row or {}).get("ranked_evidence", [])[:10]
                ],
                "result_status": (graph_row or {}).get("result_status"),
            })
        time_row = methods.get("fusion_temporal_only")
        graph_time_row = methods.get("fusion_graph_expansion_temporal")
        time_ranks, graph_time_ranks = _support_ranks(time_row), _support_ranks(graph_time_row)
        lost_time_support = {page: rank for page, rank in time_ranks.items() if rank <= 10 and page not in graph_time_ranks}
        if lost_time_support:
            time_vs_graph_time_regressions.append({
                "item_id": item_id,
                "family_id": (time_row or graph_time_row or {}).get("family_id"),
                "question": (time_row or {}).get("question"),
                "known_support_pages_in_temporal_only_top10_lost_after_graph_expansion": lost_time_support,
                "temporal_only_top_pages": [hit.get("page_key") for hit in (time_row or {}).get("ranked_evidence", [])[:10]],
                "graph_plus_temporal_top_pages": [hit.get("page_key") for hit in (graph_time_row or {}).get("ranked_evidence", [])[:10]],
                "result_status": (graph_time_row or {}).get("result_status"),
            })
    graph_regressions.sort(key=lambda row: (str(row["family_id"]), row["item_id"]))
    time_vs_graph_time_regressions.sort(key=lambda row: (str(row["family_id"]), row["item_id"]))
    table_signals = _table_diagnostics(table)
    table_signals["page_locator_exists_count"] = sum(row.get("source_page_visual_status") == "VISUAL_MATCH" for row in table)
    table_signals["candidate_value_appears_on_source_page_count"] = sum(row.get("numeric_value_cell_visual_status") == "VISUAL_MATCH" for row in table)
    table_signals["duplicate_key_rows_preserved"] = True
    scored_item_ids = {str(row.get("item_id")) for row in direct_scored}
    scored_family_ids = {str(row.get("family_id")) for row in direct_scored}
    context_item_ids = {str(row.get("item_id")) for row in context_scored}
    context_family_ids = {str(row.get("family_id")) for row in context_scored}
    no_positive_item_ids = {
        str(row.get("item_id")) for row in raw
        if row.get("retrieval_scoring_status") == "NOT_SCORED_NO_POSITIVE_PAGE_JUDGMENT"
    }
    no_positive_family_ids = {
        str(row.get("family_id")) for row in raw
        if row.get("retrieval_scoring_status") == "NOT_SCORED_NO_POSITIVE_PAGE_JUDGMENT"
    }
    by_type = defaultdict(Counter)
    scheduled_by_type = defaultdict(Counter)
    items_by_type: dict[str, set[str]] = defaultdict(set)
    families_by_type: dict[str, set[str]] = defaultdict(set)
    for row in raw:
        kind = str(row.get("question_type") or "unclassified")
        scheduled_by_type[kind][row["method"]] += 1
        by_type[kind][row["method"]] += int(row.get("execution_status") == "SUCCESS")
        items_by_type[kind].add(str(row.get("item_id")))
        families_by_type[kind].add(str(row.get("family_id")))
    report = {
        "schema": "financial-candidate-summary/v2",
        "raw_request_count": len(raw),
        "reproducibility": {
            "raw_jsonl_sha256": raw_hash,
            "candidate_dataset_sha256": run_manifest.get("candidate_dataset_sha256"),
            "table_ai_diagnostic_sha256": _sha256(table_path),
            "summarizer_sha256": _sha256(Path(__file__).resolve()),
            "run_manifest": run_manifest,
        },
        "dataset_review_tier": "AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW; NOT_GOLD",
        "retrieval_quality_by_method": quality,
        "retrieval_quality_slices_by_method": quality_slices,
        "paired_development_diagnostics": paired_graph_delta,
        "failure_analysis": {
            "graph_expansion": {
                "requests_with_at_least_one_graph_added_page_in_top10": graph_added_requests,
                "requests_with_graph_additions_denominator": len(by_item_method),
                "graph_added_pages_in_final_top10": graph_added_pages,
                "final_top10_page_slots_observed": graph_result_page_slots,
                "direct_support_requests_losing_a_fusion_top10_page_after_graph_expansion": len(graph_regressions),
                "unique_families_with_a_direct_support_regression": len({row["family_id"] for row in graph_regressions}),
                "examples": graph_regressions[:10],
            },
            "temporal_filter": {
                "graph_then_time_no_hit_request_count": sum(
                    row.get("method") == "fusion_graph_expansion_temporal"
                    and row.get("result_status") == "NO_HITS_AFTER_TEMPORAL_FILTER"
                    for row in raw
                ),
                "direct_support_requests_losing_a_time_only_top10_page_after_graph_then_time": len(time_vs_graph_time_regressions),
                "unique_families_with_a_graph_time_regression": len({row["family_id"] for row in time_vs_graph_time_regressions}),
                "examples": time_vs_graph_time_regressions[:10],
                "interpretation": "No graph fact edge after filtering is not evidence that the original filing lacks support.",
            },
        },
        "answer_and_citation_quality": {
            "numeric_answer_correctness": "NOT_RUN_GENERATION_DISABLED",
            "complete_fact_correctness": "NOT_RUN_NO_SYSTEM_ANSWER_PREDICTIONS",
            "citation_support_correctness": "NOT_RUN_NO_SYSTEM_ANSWER_CITATIONS",
            "pdf_locator_correctness": "NOT_RUN_NO_BROWSER_CITATION_OPEN_SUCCESS",
            "correct_abstention": "NOT_RUN_NO_SYSTEM_ANSWER_PREDICTIONS",
            "incorrect_abstention": "NOT_RUN_NO_SYSTEM_ANSWER_PREDICTIONS",
            "nDCG": "NOT_COMPUTED_INCOMPLETE_RELEVANCE_POOL",
        },
        "quality_metric_status": {
            "known_support_page_metrics": "DIRECT_SUPPORT_ONLY_CONTEXT_PAGES_REPORTED_SEPARATELY_NON_EXHAUSTIVE_AI_PDF_DIAGNOSTIC",
            "standard_recall_claim": "NOT_SUPPORTED_BY_NON_EXHAUSTIVE_LABELS",
            "nDCG": "NOT_COMPUTED_UNJUDGED_PAGES_ARE_NOT_IRRELEVANT",
            "answer_correctness": "NOT_RUN_GENERATION_DISABLED",
            "human_quality": "NOT_RUN_NO_HUMAN_REVIEW",
        },
        "retrieval_runtime_by_method": latency,
        "candidate_pool_sizes_by_method": candidate_pool_by_method,
        "execution_success_by_question_type": {
            kind: {
                "n_candidate_questions": len(items_by_type[kind]),
                "n_semantic_families": len(families_by_type[kind]),
                "scheduled_counts_by_method": dict(scheduled_by_type[kind]),
                "successful_counts_by_method": dict(counter),
            }
            for kind, counter in sorted(by_type.items())
        },
        "answer_and_citation_error_distribution": "NOT_RUN_GENERATION_DISABLED",
        "observed_retrieval_execution_errors": dict(Counter(
            str(row.get("error") or "NO_ERROR") for row in raw
        )),
        "table_pdf_signal_counts_not_accuracy": table_signals,
        "label_coverage": {
            "source_labelled_development_queries": len(scored_item_ids),
            "scored_direct_support_matrix_requests_across_methods": len(direct_scored),
            "source_labelled_development_families": len(scored_family_ids),
            "context_only_development_queries_excluded_from_support_metrics": len(context_item_ids),
            "context_only_development_families_excluded_from_support_metrics": len(context_family_ids),
            "context_only_matrix_requests_reported_separately": len(context_scored),
            "no_positive_page_judgment_queries_not_scored": len(no_positive_item_ids),
            "no_positive_page_judgment_families_not_scored": len(no_positive_family_ids),
            "unjudged_retrieved_pages_are_not_negative": True,
            "family_level_bootstrap": "2000 resamples; seed 20260924; intervals are descriptive only",
        },
    }
    return report


def _chart(report: dict[str, Any], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1680, 1570), "#f4f2ed")
    draw = ImageDraw.Draw(image)
    fonts = Path("C:/Windows/Fonts")
    font_regular = fonts / "msyh.ttc"
    font_bold = fonts / "msyhbd.ttc"
    try:
        title_font = ImageFont.truetype(str(font_bold), 30)
        panel_font = ImageFont.truetype(str(font_bold), 20)
        regular = ImageFont.truetype(str(font_regular), 16)
        small = ImageFont.truetype(str(font_regular), 14)
    except OSError:
        title_font = panel_font = regular = small = ImageFont.load_default()

    ink, muted, accent, teal = "#1e3038", "#5e6d72", "#376c87", "#77a9aa"
    run_started = str(report.get("reproducibility", {}).get("run_manifest", {}).get("run_started_at_utc") or "")
    run_date = run_started[:10] if len(run_started) >= 10 else "日期未记录"
    draw.text((52, 30), f"财务证据问答实验候选版 · {run_date}", fill=ink, font=title_font)
    draw.text((54, 75), "运行时延＋AI/PDF 开发支持页诊断；非独立测试，答案生成关闭", fill=muted, font=regular)

    panels = [
        (45, 125, 790, 410), (845, 125, 790, 410),
        (45, 565, 790, 410), (845, 565, 790, 410),
        (45, 1005, 790, 410), (845, 1005, 790, 410),
    ]
    for x, y, w, h in panels:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill="#ffffff", outline="#d8dedc", width=2)

    def panel_title(box, title, subtitle):
        x, y, w, _ = box
        draw.text((x + 24, y + 18), title, fill=ink, font=panel_font)
        draw.text((x + 24, y + 48), subtitle, fill=muted, font=small)

    def bars(box, labels, values, maximum, *, color=accent, suffix=""):
        x, y, w, h = box
        label_w = 185
        bar_x = x + label_w
        plot_w = w - label_w - 170
        row_h = min(45, (h - 108) // max(len(labels), 1))
        for i, (label, value) in enumerate(zip(labels, values)):
            yy = y + 93 + i * row_h
            draw.text((x + 24, yy + 2), str(label)[:25], fill=ink, font=small)
            if isinstance(value, (int, float)):
                width = int(plot_w * max(float(value), 0.0) / max(float(maximum), 1.0))
                draw.rounded_rectangle((bar_x, yy, bar_x + width, yy + 24), radius=5, fill=color)
                draw.text((bar_x + width + 8, yy + 2), f"{value}{suffix}", fill=muted, font=small)
            else:
                draw.text((bar_x, yy + 2), str(value), fill=muted, font=small)

    latency = report["retrieval_runtime_by_method"]
    panel_title(panels[0], "六种检索方法运行时延", "本地 CPU／ONNX 嵌入 · 预热查询编码器 · p50 与 p95")
    method_labels = {
        "keyword_bm25": "关键词检索（BM25）",
        "dense_semantic": "语义向量检索",
        "keyword_dense_fusion_rrf": "关键词与语义融合检索（RRF）",
        "fusion_graph_expansion": "融合检索＋知识图谱扩展",
        "fusion_graph_expansion_temporal": "融合检索＋知识图谱扩展＋时间约束",
        "fusion_temporal_only": "融合检索＋时间约束（无图扩展诊断对照）",
    }
    modes = [mode for mode in method_labels if mode in latency]
    labels = [method_labels.get(mode, mode) for mode in modes]
    maximum = max([float(latency[m][key]) for m in modes for key in ("p50_ms_all_requests", "p95_ms_all_requests") if isinstance(latency[m][key], (int, float))] or [1])
    x, y, w, h = panels[0]
    label_w, plot_w = 335, w - 450
    for i, mode in enumerate(modes):
        yy = y + 82 + i * 50
        draw.text((x + 24, yy + 8), labels[i], fill=ink, font=small)
        for j, key in enumerate(("p50_ms_all_requests", "p95_ms_all_requests")):
            value = latency[mode][key]
            width = int(plot_w * float(value) / maximum) if isinstance(value, (int, float)) else 0
            by = yy + j * 23
            draw.rounded_rectangle((x + label_w, by, x + label_w + width, by + 16), radius=4, fill=accent if j == 0 else teal)
            draw.text((x + label_w + width + 7, by - 2), f"{key[0:3]} {value} ms", fill=muted, font=small)
    signals = report["table_pdf_signal_counts_not_accuracy"]
    candidate_rows = int(signals.get("candidate_rows") or 0)
    panel_title(panels[1], f"{candidate_rows} 条表格候选 · AI 原文视觉诊断", "不是人工正确率；联合状态亦非独立准确率")
    fields = signals["fields"]
    labels = ["数字单元格视觉匹配", "正负号视觉匹配", "单位／尺度视觉匹配", "年度列视觉匹配", "指标语义别名匹配", "物理页定位匹配"]
    keys = ["numeric_cell_visual_match", "sign_visual_match", "unit_scale_visual_match", "year_column_visual_match", "metric_semantics_alias_match", "source_physical_page_visual_match"]
    values = [fields[key]["n"] for key in keys]
    bars(panels[1], labels, values, candidate_rows, color=teal, suffix=f" / {candidate_rows}")
    joint = signals["joint_ai_diagnostic_counts_not_accuracy"]
    no_obvious = joint.get("AI_VISUAL_NO_OBVIOUS_MISMATCH", 0)
    tx, ty, _, _ = panels[1]
    draw.text((tx + 24, ty + 372), f"AI 联合诊断无明显异常：{no_obvious}/{candidate_rows}；重复候选保持原样。", fill=muted, font=small)

    types = report["execution_success_by_question_type"]
    panel_title(panels[2], "候选问题类型与样本量", "按语义家族划分的待审问题；数量不代表质量成绩")
    labels = list(types)
    values = [types[kind]["n_candidate_questions"] for kind in labels]
    type_labels = {
        "single_year_fact": "单年度事实",
        "fact_year_in_later_disclosure": "后续披露中的历史事实",
        "cross_year_comparison": "跨年度比较",
        "unit_conversion_or_calculation": "单位换算或计算",
        "relation_or_conditional_risk": "关系或条件风险",
        "unanswerable_or_ambiguous": "不可回答或歧义",
        "relation": "关系问题",
        "unclassified": "未分类",
    }
    labels = [type_labels.get(kind, kind) for kind in labels]
    bars(panels[2], labels, values, max(values or [1]), color="#8aa2ac", suffix=" 个语义家族")

    panel_title(panels[3], "直接支持页命中率 @10", "AI/PDF 开发诊断 · context-only 与直接支持分开")
    quality = report["retrieval_quality_by_method"]
    quality_modes = [mode for mode in method_labels if mode in quality]
    qlabels = [method_labels.get(mode, mode) for mode in quality_modes]
    qvalues = [quality[mode]["direct_support_hit_and_page_coverage_at_k"]["10"]["labelled_page_hit"]["rate"] for mode in quality_modes]
    x, y, w, h = panels[3]
    label_w, plot_w = 360, w - 465
    for i, (label, value, mode) in enumerate(zip(qlabels, qvalues, quality_modes)):
        yy = y + 88 + i * 43
        draw.text((x + 22, yy + 2), label, fill=ink, font=small)
        denom = quality[mode]["direct_support_hit_and_page_coverage_at_k"]["10"]["labelled_page_hit"]["denominator"]
        width = int(plot_w * float(value) / max(float(max([v for v in qvalues if isinstance(v, (int, float))] or [1])), 1.0)) if isinstance(value, (int, float)) else 0
        draw.rounded_rectangle((x + label_w, yy, x + label_w + width, yy + 19), radius=4, fill=accent)
        draw.text((x + label_w + width + 7, yy), f"{value} ({denom} 题)", fill=muted, font=small)
    draw.text((x + 22, y + 372), f"直接支持分母：{quality[quality_modes[0]]['labelled_requests']} 问／{quality[quality_modes[0]]['unique_families']} 家族；context-only 另列。", fill=muted, font=small)

    errors = report["observed_retrieval_execution_errors"]
    nonzero = {key: value for key, value in errors.items() if key != "NO_ERROR" and value}
    panel_title(panels[4], "执行失败与方法返回状态", "执行失败不等于空结果；过滤后无页不等于原文无证据")
    ex, ey, ew, eh = panels[4]
    if nonzero:
        err_labels = [str(key).split(":", 1)[0][:30] for key in nonzero]
        err_values = list(nonzero.values())
        bars(panels[4], err_labels, err_values, max(err_values or [1]), color="#b75b4b", suffix=" 次")
    else:
        draw.text((ex + 28, ey + 105), f"检索执行异常：0/{report['raw_request_count']} 次排程请求", fill=ink, font=regular)
    statuses: Counter[str] = Counter()
    for method in report["retrieval_runtime_by_method"].values():
        statuses.update({str(key): int(value) for key, value in method["result_statuses"].items()})
    status_text = " · ".join(f"{key} {value}" for key, value in sorted(statuses.items())) or "NOT_RUN"
    draw.text((ex + 28, ey + 160), "方法返回状态（含成功空集）：", fill=muted, font=small)
    draw.multiline_text((ex + 28, ey + 187), status_text, fill=ink, font=small, spacing=8)

    panel_title(panels[5], "各方法实际页级候选池", "均值；原始 JSONL 逐题保留候选块/页数、扩展数与过滤前后数")
    cx, cy, cw, ch = panels[5]
    pool = report["candidate_pool_sizes_by_method"]
    columns = [
        ("bm25_unique_candidate_pages", "关键词页"),
        ("dense_unique_candidate_pages", "向量页"),
        ("fusion_unique_candidate_pages", "融合页"),
        ("graph_added_candidate_pages", "图新增"),
        ("temporal_candidate_pages_before_filter", "过滤前"),
        ("temporal_candidate_pages_after_filter", "过滤后"),
        ("returned_unique_pages", "返回页"),
    ]
    label_x = cx + 22
    table_x = cx + 316
    col_w = 61
    header_y = cy + 89
    draw.text((label_x, header_y), "检索方法", fill=muted, font=small)
    for col_idx, (_, heading) in enumerate(columns):
        draw.text((table_x + col_idx * col_w, header_y), heading, fill=muted, font=small)
    for row_idx, mode in enumerate(modes):
        yy = header_y + 37 + row_idx * 43
        label_font = ImageFont.truetype(str(font_regular), 12) if (fonts / "msyh.ttc").exists() else small
        draw.text((label_x, yy + 2), method_labels[mode], fill=ink, font=label_font)
        values = pool.get(mode, {}).get("counts", {})
        for col_idx, (key, _) in enumerate(columns):
            value = values.get(key, {}).get("mean", "NOT_RUN")
            display = f"{value:.1f}" if isinstance(value, (int, float)) else "—"
            draw.text((table_x + col_idx * col_w, yy + 2), display, fill=ink, font=small)

    draw.text((52, 1515), "* 仅统计显式核对的支持页；未判断页不作负例。AI/PDF 诊断不是人审；答案、引文与拒答准确率未运行。", fill=muted, font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--table-audit", required=True)
    parser.add_argument("--dataset", required=True, help="Exact development JSONL named by the raw manifest")
    parser.add_argument("--output", required=True, help="New summary JSON; never overwritten")
    parser.add_argument("--chart", required=True, help="New PNG; never overwritten")
    args = parser.parse_args()
    raw_path, table_path = Path(args.raw).resolve(), Path(args.table_audit).resolve()
    output, chart = Path(args.output).resolve(), Path(args.chart).resolve()
    if output.exists() or chart.exists():
        raise FileExistsError("Refusing to overwrite existing summary output")
    dataset_path = Path(args.dataset).resolve()
    report = summarize(raw_path, table_path, dataset_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _chart(report, chart)
    print(json.dumps({"summary": str(output), "chart": str(chart), "schema": report["schema"], "retrieval_methods_scored": len(report["retrieval_quality_by_method"]), "answer_metrics": report["answer_and_citation_quality"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
