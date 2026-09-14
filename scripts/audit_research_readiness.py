"""Audit whether the current GraphRAG checkout is ready for research claims.

This is a fail-closed, local-only audit.  It checks that published-looking
artifacts are backed by current data, that the extraction sample is complete,
and that a human-reviewed Golden QA benchmark exists before retrieval metrics
are treated as research results.  An automatic Silver benchmark can satisfy
the engineering regression gate, but it never satisfies the human-gold gate.
It does not contact Neo4j, call an LLM, or modify the corpus and annotation
files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "research_readiness_current.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            rows.append(value)
    return rows


def _latest_file(directory: Path, *patterns: str) -> Path | None:
    """Return the newest matching report without silently using an old baseline."""

    if not directory.exists():
        return None
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in directory.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    *,
    blocking: bool = True,
    note: str = "",
) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else "BLOCKED" if blocking else "WARNING",
            "blocking": blocking,
            "observed": observed,
            "expected": expected,
            "note": note,
        }
    )


def _count_labeled(rows: list[dict[str, Any]]) -> int:
    return sum(row.get("annotation_status") == "LABELED" for row in rows)


def _annotation_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "source_entity_correct",
        "target_entity_correct",
        "relation_correct",
        "evidence_supports_relation",
    )
    total = len(rows)
    rates: dict[str, Any] = {}
    for field in fields:
        correct = sum((row.get("labels") or {}).get(field) is True for row in rows)
        rates[field] = {
            "correct": correct,
            "total": total,
            "rate": round(correct / total, 4) if total else None,
        }
    return rates


def _annotation_label_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = {"true": 0, "false": 0, "uncertain": 0, "null": 0}
    for row in rows:
        value = (row.get("labels") or {}).get(field)
        key = "null" if value is None else str(value).lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("review_status") or "<missing>") for row in rows)
    return dict(sorted(counts.items()))


def _human_gold_invalid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact validation errors for rows claiming to be human gold."""

    invalid: list[dict[str, Any]] = []
    for row in rows:
        missing: list[str] = []
        if row.get("review_status") != "HUMAN_REVIEWED":
            missing.append("review_status")
        if not str(row.get("reviewer") or "").strip():
            missing.append("reviewer")
        if not str(row.get("reference_answer") or "").strip():
            missing.append("reference_answer")
        if not isinstance(row.get("answerable"), bool):
            missing.append("answerable")
        if not isinstance(row.get("requires_abstention"), bool):
            missing.append("requires_abstention")
        evidence_ids = row.get("gold_evidence_ids")
        if not isinstance(evidence_ids, list):
            missing.append("gold_evidence_ids")
        elif row.get("answerable") is True and not evidence_ids:
            missing.append("gold_evidence_ids(nonempty for answerable row)")
        if not isinstance(row.get("relevant_evidence_grades"), dict):
            missing.append("relevant_evidence_grades")
        if row.get("answerable") is False and row.get("requires_abstention") is not True:
            missing.append("requires_abstention(true for unanswerable row)")
        if missing:
            invalid.append({"id": row.get("id"), "missing": missing})
    return invalid


def audit(root: Path = ROOT) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    facts: dict[str, Any] = {}

    reports_dir = root / "reports"
    manifest_path = _latest_file(reports_dir, "*corpus_manifest*.json") or (
        reports_dir / "2026-08-14_corpus_manifest.json"
    )
    snapshot_path = _latest_file(reports_dir, "neo4j_snapshot_post_rebuild_*.json") or (
        reports_dir / "neo4j_snapshot_post_rebuild_2025_v2.json"
    )
    strict_path = _latest_file(
        reports_dir, "strict_chain_audit_2025_post_rebuild_*.json"
    ) or (reports_dir / "strict_chain_audit_2025_post_repair_v2.json")
    annotation_audit_path = _latest_file(
        reports_dir, "extraction_annotation_audit_2025_post_rebuild_*.json"
    ) or (reports_dir / "extraction_annotation_audit_2025_post_repair_v2.json")
    historical_extraction_report_path = root / "reports" / "extraction_quality_2025_post_repair_v2.json"
    sample_path = root / "evaluation" / "annotation" / "extraction_sample_2025_post_repair_v2.jsonl"
    human_annotation_path = root / "evaluation" / "annotation" / "extraction_sample_2025_post_repair_human_v1.jsonl"
    baseline_path = root / "evaluation" / "annotation" / "extraction_sample_v1.jsonl"
    golden_candidate_path = root / "data" / "evaluation" / "golden_qa_v2.jsonl"
    human_golden_path = root / "evaluation" / "golden_qa_human_v1.jsonl"
    silver_benchmark_path = root / "evaluation" / "silver_retrieval_v1.jsonl"
    retrieval_path = _latest_file(reports_dir, "retrieval_baselines_*.json") or (
        reports_dir / "retrieval_baselines_smoke.json"
    )
    silver_retrieval_path = _latest_file(reports_dir, "retrieval_benchmark_silver_*.json")

    manifest = None
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        facts["corpus_id"] = manifest.get("corpus_id")
        facts["manifest_claim_id_version"] = manifest.get("claim_id_version")
    _check(
        checks,
        "corpus_manifest_present",
        manifest is not None,
        str(manifest_path) if manifest is not None else "missing",
        "current corpus manifest",
        note="The corpus hash and model metadata are required for a reproducible run.",
    )

    snapshot = _load_json(snapshot_path) if snapshot_path.exists() else None
    _check(
        checks,
        "post_rebuild_snapshot_present",
        snapshot is not None,
        str(snapshot_path) if snapshot is not None else "missing",
        "post-rebuild machine-readable snapshot",
        note="This is an audit baseline, not a substitute for a binary database backup.",
    )

    strict = _load_json(strict_path) if strict_path.exists() else None
    strict_status = strict.get("status") if strict else None
    strict_edges = ((strict or {}).get("edge_audit") or {}).get("strict_verbatim_edges")
    invalid_edges = ((strict or {}).get("edge_audit") or {}).get("invalid_strict_edges")
    _check(
        checks,
        "strict_provenance_audit",
        strict_status == "PASS" and invalid_edges == 0,
        {"status": strict_status, "strict_verbatim_edges": strict_edges, "invalid_edges": invalid_edges},
        {"status": "PASS", "invalid_edges": 0},
        note="Structural provenance is necessary but does not establish semantic correctness.",
    )

    snapshot_labels = {
        str(item.get("label")): int(item.get("count") or 0)
        for item in ((snapshot or {}).get("graph") or {}).get("nodes_by_label") or []
    }
    active_claim_count = sum(
        int(item.get("claims") or 0)
        for item in ((snapshot or {}).get("graph") or {}).get("active_claims") or []
    )
    facts["derived_model_inventory"] = {
        "active_evidence_claims": active_claim_count or None,
        "temporal_facts": snapshot_labels.get("TemporalFact"),
        "temporal_changes": snapshot_labels.get("TemporalChange"),
        "financial_observations": snapshot_labels.get("FinancialObservation"),
    }
    _check(
        checks,
        "derived_temporal_model_matches_active_claims",
        active_claim_count > 0 and snapshot_labels.get("TemporalFact") == active_claim_count,
        {
            "active_evidence_claims": active_claim_count or None,
            "temporal_facts": snapshot_labels.get("TemporalFact"),
        },
        "one bitemporal fact version per active VERBATIM EvidenceClaim",
        note="Replacing a filing invalidates old temporal materializations; rebuild derived models before using hybrid-temporal results.",
    )

    rows = _load_jsonl(sample_path) if sample_path.exists() else []
    labeled = _count_labeled(rows)
    facts["post_repair_annotation"] = {
        "path": str(sample_path.relative_to(root)) if sample_path.exists() else str(sample_path),
        "rows": len(rows),
        "labeled": labeled,
        "rates": _annotation_rates(rows),
    }
    _check(
        checks,
        "post_repair_annotation_complete",
        len(rows) == 30 and labeled == 30,
        {"rows": len(rows), "labeled": labeled},
        {"rows": 30, "labeled": 30},
        note="These are precision-like estimates from extracted claims, not recall or F1.",
    )

    human_rows = _load_jsonl(human_annotation_path) if human_annotation_path.exists() else []
    human_labeled = _count_labeled(human_rows)
    annotator_counts: dict[str, int] = {}
    for row in human_rows:
        annotator = str(row.get("annotator") or "<missing>").strip() or "<missing>"
        annotator_counts[annotator] = annotator_counts.get(annotator, 0) + 1
    ai_annotators = {
        annotator
        for annotator in annotator_counts
        if annotator != "<missing>" and ("gpt" in annotator.lower() or "ai" in annotator.lower())
    }
    label_fields = (
        "source_entity_correct",
        "target_entity_correct",
        "relation_correct",
        "evidence_supports_relation",
    )
    facts["ai_assisted_annotation"] = {
        "path": str(human_annotation_path.relative_to(root))
        if human_annotation_path.exists()
        else str(human_annotation_path),
        "rows": len(human_rows),
        "labeled": human_labeled,
        "annotator_counts": annotator_counts,
        "label_counts": {
            field: _annotation_label_counts(human_rows, field) for field in label_fields
        },
        "is_independent_human_golden_qa": False,
    }
    _check(
        checks,
        "ai_assisted_annotation_not_human_golden_qa",
        human_annotation_path.exists() and bool(ai_annotators),
        {
            "rows": len(human_rows),
            "labeled": human_labeled,
            "annotator_counts": annotator_counts,
            "explicit_ai_annotators": sorted(ai_annotators),
            "is_independent_human_golden_qa": False,
        },
        {"explicit_ai_annotator": True, "is_independent_human_golden_qa": False},
        blocking=False,
        note="This AI-assisted working set must not satisfy or be reported as an independent human Golden QA benchmark.",
    )

    baseline_rows = _load_jsonl(baseline_path) if baseline_path.exists() else []
    baseline_labeled = _count_labeled(baseline_rows)
    facts["baseline_annotation"] = {"rows": len(baseline_rows), "labeled": baseline_labeled}
    _check(
        checks,
        "baseline_annotation_inventory",
        len(baseline_rows) == 60 and baseline_labeled == 60,
        {"rows": len(baseline_rows), "labeled": baseline_labeled},
        {"rows": 60, "labeled": 60},
        note="The original 60-row baseline remains a separate comparison set.",
    )

    annotation_audit = _load_json(annotation_audit_path) if annotation_audit_path.exists() else None
    report_sample = (annotation_audit or {}).get("sample") or {}
    report_status_counts = report_sample.get("status_counts") or {}
    report_labeled = report_status_counts.get("LABELED", 0)
    facts["extraction_annotation_audit"] = {
        "path": str(annotation_audit_path.relative_to(root)) if annotation_audit_path.exists() else str(annotation_audit_path),
        "rows": report_sample.get("rows"),
        "labeled": report_labeled,
    }
    _check(
        checks,
        "extraction_report_is_current",
        annotation_audit is not None and report_labeled == labeled and report_sample.get("rows") == len(rows),
        {"report_rows": report_sample.get("rows"), "report_labeled": report_labeled, "live_rows": len(rows), "live_labeled": labeled},
        {"rows": len(rows), "labeled": labeled},
        note="A stale report must not be cited as the current annotation result.",
    )
    historical_extraction_report = (
        _load_json(historical_extraction_report_path) if historical_extraction_report_path.exists() else None
    )
    historical_sample = (historical_extraction_report or {}).get("annotation_sample") or {}
    _check(
        checks,
        "historical_extraction_report_is_classified",
        historical_extraction_report is None or historical_sample.get("labeled") == labeled,
        {
            "path": str(historical_extraction_report_path.relative_to(root))
            if historical_extraction_report_path.exists()
            else "missing",
            "report_rows": historical_sample.get("rows"),
            "report_labeled": historical_sample.get("labeled"),
        },
        "historical report is absent or current",
        blocking=False,
        note="The older extraction-quality report is retained as history and must not be cited as current.",
    )

    candidate_golden_rows = (
        _load_jsonl(golden_candidate_path) if golden_candidate_path.exists() else []
    )
    facts["golden_qa_candidate"] = {
        "source": str(golden_candidate_path.relative_to(root))
        if golden_candidate_path.exists()
        else str(golden_candidate_path),
        "rows": len(candidate_golden_rows),
        "status_counts": _status_counts(candidate_golden_rows),
        "dataset_status": "AUTO_GENERATED_REGRESSION_CANDIDATE",
        "eligible_as_human_gold": False,
    }

    human_golden_rows = _load_jsonl(human_golden_path) if human_golden_path.exists() else []
    human_reviewed = sum(
        row.get("review_status") == "HUMAN_REVIEWED" for row in human_golden_rows
    )
    human_answerable = sum(row.get("answerable") is True for row in human_golden_rows)
    human_invalid_rows = _human_gold_invalid_rows(human_golden_rows)
    human_source = (
        str(human_golden_path.relative_to(root))
        if human_golden_path.exists()
        else str(human_golden_path)
    )
    human_gold_facts = {
        "source": human_source,
        "rows": len(human_golden_rows),
        "status_counts": _status_counts(human_golden_rows),
        "human_reviewed": human_reviewed,
        "answerable": human_answerable,
        "invalid_rows": len(human_invalid_rows),
        "invalid_examples": human_invalid_rows[:5],
    }
    facts["human_gold"] = human_gold_facts
    facts["golden_qa"] = human_gold_facts
    _check(
        checks,
        "human_golden_qa_available",
        human_golden_path.exists()
        and len(human_golden_rows) >= 30
        and human_reviewed == len(human_golden_rows)
        and not human_invalid_rows,
        {
            "source": human_source,
            "rows": len(human_golden_rows),
            "status_counts": _status_counts(human_golden_rows),
            "invalid_rows": len(human_invalid_rows),
            "invalid_examples": human_invalid_rows[:5],
        },
        {
            "source": "evaluation/golden_qa_human_v1.jsonl",
            "minimum_rows": 30,
            "all_rows": "HUMAN_REVIEWED",
            "all_core_fields": "valid",
        },
        note="Only the separate human file can satisfy this check; the candidate file is never human gold.",
    )

    retrieval = _load_json(retrieval_path) if retrieval_path.exists() else None
    question_count = (retrieval or {}).get("question_count", 0)
    modes = set((retrieval or {}).get("modes") or [])
    records = (retrieval or {}).get("records") or []
    facts["retrieval_smoke"] = {"question_count": question_count, "run_count": len(records), "modes": sorted(modes)}
    _check(
        checks,
        "retrieval_baseline_modes",
        modes == {"vector", "graph", "hybrid", "hybrid_temporal"},
        sorted(modes),
        ["graph", "hybrid", "hybrid_temporal", "vector"],
        note="All four modes must be present before a comparison is meaningful.",
    )
    silver_rows = _load_jsonl(silver_benchmark_path) if silver_benchmark_path.exists() else []
    silver_report = _load_json(silver_retrieval_path) if silver_retrieval_path and silver_retrieval_path.exists() else None
    silver_question_count = (silver_report or {}).get("evaluated_questions") or (silver_report or {}).get("dataset_size") or len(silver_rows)
    silver_status = (silver_report or {}).get("dataset_status") or ("AUTO_GENERATED_SILVER_NOT_HUMAN_GOLD" if silver_rows else None)
    silver_modes = set((silver_report or {}).get("modes") or [])
    facts["retrieval_benchmark"] = {
        "dataset": str(silver_benchmark_path.relative_to(root)) if silver_benchmark_path.exists() else str(silver_benchmark_path),
        "report": str(silver_retrieval_path.relative_to(root)) if silver_retrieval_path and silver_retrieval_path.exists() else None,
        "question_count": int(silver_question_count or 0),
        "evaluated_questions": (silver_report or {}).get("evaluated_questions"),
        "dataset_status": silver_status,
        "modes": sorted(silver_modes),
        "human_reviewed": False,
        "eligible_as_human_gold": False,
    }
    has_automatic_benchmark = int(silver_question_count or 0) >= 30 and (
        not silver_report or silver_modes == {"vector", "graph", "hybrid", "hybrid_temporal"}
    )
    _check(
        checks,
        "retrieval_benchmark_has_multiple_questions",
        question_count >= 30 or has_automatic_benchmark,
        {
            "smoke_question_count": question_count,
            "silver_question_count": int(silver_question_count or 0),
            "silver_status": silver_status,
            "silver_modes": sorted(silver_modes),
        },
        ">= 30 questions in a human-reviewed benchmark or automatic Silver regression benchmark",
        note="Silver is sufficient for engineering regression, but human-reviewed Golden QA is still required for paper-level semantic claims.",
    )

    # Only compare runs that record the frozen extraction temperature.  The
    # older repair reports predate that field and must not be used to claim
    # non-repeatability of the current protocol.
    extraction_runs: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("rebuild_2025_*.json")):
        try:
            report = _load_json(path)
        except (OSError, ValueError):
            continue
        file_report = ((report.get("files") or [{}])[0])
        llm_report = file_report.get("llm") or {}
        if file_report.get("status") != "completed" or llm_report.get("extraction_temperature") is None:
            continue
        dry_run = bool(file_report.get("dry_run", False))
        output_count = (
            file_report.get("triples_extracted")
            if dry_run
            else file_report.get("triples_ingested")
        )
        extraction_runs.append(
            {
                "path": str(path.relative_to(root)),
                "triples_ingested": file_report.get("triples_ingested"),
                "triples_extracted": file_report.get("triples_extracted"),
                "dry_run": dry_run,
                "repeatability_output_count": output_count,
                "document_sha256": file_report.get("document_sha256"),
                "llm_model": file_report.get("llm_model"),
                "prompt_version": file_report.get("prompt_version"),
                "extraction_temperature": llm_report.get("extraction_temperature"),
            }
        )
    run_fingerprints = {
        (
            item["document_sha256"],
            item["llm_model"],
            item["prompt_version"],
            item["extraction_temperature"],
            item["repeatability_output_count"],
        )
        for item in extraction_runs
    }
    facts["extraction_runs"] = {
        "successful_frozen_runs": len(extraction_runs),
        "runs": extraction_runs,
    }
    _check(
        checks,
        "extraction_run_is_repeatable",
        len(extraction_runs) >= 2 and len(run_fingerprints) == 1,
        {"successful_frozen_runs": len(extraction_runs), "runs": extraction_runs},
        "same output under the same frozen configuration",
        note="At least two successful runs with the same corpus, model, prompt, temperature, and output count are required before claiming repeatability.",
    )

    has_legacy_src = (root / "src").exists()
    _check(
        checks,
        "legacy_source_tree_is_classified",
        not has_legacy_src,
        "src/ present" if has_legacy_src else "not present",
        "no unclassified parallel implementation",
        blocking=False,
        note="Classify or archive src/ before claiming a clean canonical codebase; do not delete it blindly.",
    )

    report_files = list((root / "reports").glob("*")) if (root / "reports").exists() else []
    log_count = sum(path.suffix == ".log" for path in report_files)
    _check(
        checks,
        "report_directory_is_hygienic",
        len(report_files) <= 40 and log_count <= 5,
        {"files": len(report_files), "logs": log_count},
        {"files": "<= 40", "logs": "<= 5"},
        blocking=False,
        note="Historical reports should be moved behind an explicit archive manifest.",
    )

    blocking_failures = [item["name"] for item in checks if item["status"] == "BLOCKED" and item["blocking"]]
    warnings = [item["name"] for item in checks if item["status"] == "WARNING"]
    if snapshot:
        active_claims = ((snapshot.get("graph") or {}).get("active_claims") or [])
        facts["graph_inventory"] = {
            "total_claims": sum(int(item.get("claims", 0)) for item in active_claims),
            "by_filing": active_claims,
        }

    return {
        "schema": "strategic-graphrag-research-readiness/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_RESEARCH" if not blocking_failures else "NOT_READY",
        "blocking_failures": blocking_failures,
        "warnings": warnings,
        "facts": facts,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit research readiness without touching the corpus")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "READY_FOR_RESEARCH" else 2)


if __name__ == "__main__":
    main()
