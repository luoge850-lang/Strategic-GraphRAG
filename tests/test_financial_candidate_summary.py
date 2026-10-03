import json
import hashlib
from pathlib import Path

import pytest

from scripts.summarize_financial_candidate_run import EXPECTED_METHODS, FROZEN_INPUT_KEYS, _chart, _percentile
from scripts.summarize_financial_candidate_run import summarize


def test_p50_uses_median_for_even_sample_count():
    assert _percentile([1.0, 2.0, 100.0, 101.0], 0.5) == 51.0


def test_p95_uses_nearest_rank_and_missing_is_not_run():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert _percentile([], 0.5) == "NOT_RUN"


def test_context_only_pages_never_inflate_direct_support_metrics(tmp_path):
    dataset_rows = []
    raw_rows = []
    method_names = {
        "keyword_bm25": "关键词检索（BM25）",
        "dense_semantic": "语义向量检索",
        "keyword_dense_fusion_rrf": "关键词与语义融合检索（RRF）",
        "fusion_graph_expansion": "融合检索＋知识图谱扩展",
        "fusion_graph_expansion_temporal": "融合检索＋知识图谱扩展＋时间约束",
        "fusion_temporal_only": "融合检索＋时间约束（无图扩展诊断对照）",
    }
    cases = (
        ("direct", "family-direct", "SCORED_KNOWN_SUPPORT_PAGES", "filing.pdf#1"),
        ("context", "family-context", "SCORED_KNOWN_CONTEXT_PAGES_ONLY", "filing.pdf#2"),
        ("unknown", "family-unknown", "NOT_SCORED_NO_POSITIVE_PAGE_JUDGMENT", None),
    )
    for item_id, family, status, relevant_page in cases:
        dataset_rows.append({
            "item_id": item_id,
            "family_id": family,
            "partition": "development",
            "label_status": "AI_SOURCE_REVIEW_NOT_HUMAN",
            "label_tier": "AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW",
            "gold_status": "NOT_GOLD",
            "question": item_id,
            "question_type": "single_year_fact",
            "answerability_label": "ANSWERABLE",
            "reference_answer": f"Reference for {item_id}",
            "gold_page_grades": {relevant_page: 3} if relevant_page else {},
            "retrieval_scoring_status": status,
            "candidate_filing": "filing.pdf",
            "candidate_pages": [1],
            "candidate_variants": [],
        })
        for method in EXPECTED_METHODS:
            raw_rows.append({
                "request_id": f"{item_id}:{method}",
                "item_id": item_id,
                "family_id": family,
                "method": method,
                "method_name_zh": method_names[method],
                "partition": "development",
                "question": item_id,
                "question_type": "single_year_fact",
                "execution_status": "SUCCESS",
                "result_status": "OK",
                "build_id": "build-test",
                "query_plan": {"fiscal_years": [2025]},
                "latency_ms": 1.0,
                "retrieval_scoring_status": status,
                "label_tier": "AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW",
                "answerability_label": "ANSWERABLE",
                "reference_answer": f"Reference for {item_id}",
                "gold_page_grades": {relevant_page: 3} if relevant_page else {},
                "candidate_filing_hint": "filing.pdf",
                "candidate_pages_hint": [1],
                "ranked_evidence": [{"page_key": relevant_page}] if relevant_page else [],
                "candidate_counts": {"returned_unique_pages": int(bool(relevant_page))},
            })

    raw_path = tmp_path / "raw.jsonl"
    table_path = tmp_path / "table.jsonl"
    dataset_path = tmp_path / "dataset.jsonl"
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8")
    dataset_path.write_text("".join(json.dumps(row) + "\n" for row in dataset_rows), encoding="utf-8")
    table_path.write_text("", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    source_paths = {
        "runner": root / "scripts/run_financial_retrieval_matrix.py",
        "query_parser": root / "strategic_graphrag/engine/query_understanding.py",
        "summarizer": root / "scripts/summarize_financial_candidate_run.py",
        "protocol": root / "docs/financial_qa_candidate_protocol_v4.md",
        "dataset": dataset_path,
        "dependency_lock": root / "requirements-lock-2026-09-19.txt",
    }
    frozen_hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in source_paths.items()
    }
    raw_manifest = {
        "schema": "financial-retrieval-raw-run/v1",
        "records": len(raw_rows),
        "raw_jsonl_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "candidate_dataset_sha256": frozen_hashes["dataset"],
        "candidate_question_count": len(dataset_rows),
        "candidate_family_count": len(dataset_rows),
        "selected_family_ids": sorted(row["family_id"] for row in dataset_rows),
        "limited_smoke_run": False,
        "matrix_modes": list(EXPECTED_METHODS),
        "build_id": "build-test",
        "runner_sha256": frozen_hashes["runner"],
        "protocol_sha256": frozen_hashes["protocol"],
        "dependency_lock_sha256": frozen_hashes["dependency_lock"],
        "frozen_input_sha256_before": frozen_hashes,
        "frozen_input_sha256_after": dict(frozen_hashes),
    }
    manifest_path = raw_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

    report = summarize(raw_path, table_path, dataset_path)
    method = report["retrieval_quality_by_method"]["keyword_bm25"]
    assert method["direct_support_hit_and_page_coverage_at_k"]["10"]["labelled_page_hit"] == {
        "n": 1, "denominator": 1, "rate": 1.0
    }
    assert method["context_only_diagnostic"]["context_only_page_diagnostic_at_k"]["10"]["labelled_page_hit"] == {
        "n": 1, "denominator": 1, "rate": 1.0
    }
    assert report["label_coverage"]["source_labelled_development_queries"] == 1
    assert report["label_coverage"]["context_only_development_queries_excluded_from_support_metrics"] == 1
    chart = tmp_path / "summary.png"
    _chart(report, chart)
    assert chart.is_file() and chart.stat().st_size > 0

    missing_freeze = dict(raw_manifest)
    missing_freeze.pop("frozen_input_sha256_before")
    manifest_path.write_text(json.dumps(missing_freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="complete, stable pre/post freeze"):
        summarize(raw_path, table_path, dataset_path)

    manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")
    truncated_raw = "".join(
        json.dumps(row) + "\n" for row in raw_rows if row["item_id"] != "unknown"
    )
    raw_path.write_text(truncated_raw, encoding="utf-8")
    truncated_manifest = dict(raw_manifest)
    truncated_manifest["records"] = len(raw_rows) - len(EXPECTED_METHODS)
    truncated_manifest["raw_jsonl_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(truncated_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Raw schedule differs from frozen dataset"):
        summarize(raw_path, table_path, dataset_path)

    altered_rows = [dict(row) for row in raw_rows]
    for row in altered_rows:
        if row["item_id"] == "direct":
            row["gold_page_grades"] = {"filing.pdf#999": 3}
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in altered_rows), encoding="utf-8")
    altered_manifest = dict(raw_manifest)
    altered_manifest["raw_jsonl_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(altered_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Request labels differ"):
        summarize(raw_path, table_path, dataset_path)
