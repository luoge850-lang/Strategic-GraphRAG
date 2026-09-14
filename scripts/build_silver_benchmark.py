"""Build a deterministic, evidence-linked Silver retrieval benchmark.

This builder reads the current strict Neo4j graph and never edits human QA
files.  The resulting data is deliberately marked as ``AUTO_GENERATED_SILVER``
because the questions and expected evidence are derived from the same graph
that the system retrieves from.  It is suitable for engineering regression,
ablation smoke tests, and coverage diagnostics, but it is not independent
human Golden QA.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "evaluation" / "silver_retrieval_v1.jsonl"
FILINGS = (
    ("2023-10-K", "2023-10-K.pdf", 2023),
    ("2024-10-K", "2024-10-K.pdf", 2024),
    ("2025-10-K", "2025-10-K.pdf", 2025),
)


def _query_records(session: Any, query: str, **params: Any) -> list[dict[str, Any]]:
    return [record.data() for record in session.run(query, **params)]


def _page_key(filing: str, page: int) -> str:
    return f"{str(filing).removesuffix('.pdf')}#{int(page)}"


def _entity_token(value: Any) -> str:
    return str(value or "UNKNOWN").strip().upper().replace(" ", "_")


def _relation_phrase(relation: str) -> str:
    return str(relation or "RELATION").strip().upper()


def _unique_rows(rows: Iterable[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _balanced_claims(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select deterministic claims while covering relation types first."""

    if limit <= 0:
        return []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("relation") or "UNKNOWN")].append(row)
    selected: list[dict[str, Any]] = []
    for relation in sorted(groups):
        selected.append(groups[relation][0])
        if len(selected) >= limit:
            return selected[:limit]
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _direct_rows(session: Any, doc_id: str, limit: int) -> list[dict[str, Any]]:
    rows = _query_records(
        session,
        """
        MATCH (claim:EvidenceClaim {doc_id:$doc_id})-[:ABOUT_SOURCE]->(src)
        MATCH (claim)-[:ABOUT_TARGET]->(tgt)
        WHERE claim.verification_status='VERBATIM'
        RETURN claim.id AS claim_id, claim.text AS claim_text,
               claim.page AS page, claim.year AS year,
               src.id AS source_id, tgt.id AS target_id,
               claim.relation_type AS relation
        ORDER BY claim.page, claim.id
        """,
        doc_id=doc_id,
    )
    return _balanced_claims(rows, limit)


def _multi_hop_rows(session: Any, filename: str, limit: int) -> list[dict[str, Any]]:
    rows = _query_records(
        session,
        """
        MATCH (src)-[r1]->(mid)-[r2]->(tgt)
        WHERE coalesce(r1.source_filing, r1.filing, '')=$filename
          AND coalesce(r2.source_filing, r2.filing, '')=$filename
          AND r1.evidence_id IS NOT NULL AND r2.evidence_id IS NOT NULL
          AND r1.evidence_id <> r2.evidence_id
        MATCH (c1:EvidenceClaim {id:r1.evidence_id})
        MATCH (c2:EvidenceClaim {id:r2.evidence_id})
        WHERE c1.verification_status='VERBATIM'
          AND c2.verification_status='VERBATIM'
        RETURN src.id AS source_id, mid.id AS middle_id, tgt.id AS target_id,
               type(r1) AS relation1, type(r2) AS relation2,
               c1.id AS claim1_id, c1.text AS claim1_text,
               c1.page AS page1, c1.year AS year1,
               c2.id AS claim2_id, c2.text AS claim2_text,
               c2.page AS page2, c2.year AS year2
        ORDER BY page1, page2, source_id, middle_id, target_id, claim1_id, claim2_id
        """,
        filename=filename,
    )
    return _unique_rows(rows, ("source_id", "middle_id", "target_id", "claim1_id", "claim2_id"))[:limit]


def _metric_rows(session: Any) -> dict[str, list[dict[str, Any]]]:
    rows = _query_records(
        session,
        """
        MATCH (claim:EvidenceClaim {relation_type:'REPORTS_METRIC'})-[:ABOUT_SOURCE]->(src)
        MATCH (claim)-[:ABOUT_TARGET]->(metric)
        WHERE claim.verification_status='VERBATIM'
          AND src.id='nvidia_corporation'
        RETURN metric.id AS metric_id, claim.id AS claim_id,
               claim.text AS claim_text, claim.page AS page,
               claim.year AS year, claim.doc_id AS doc_id
        ORDER BY metric_id, year, claim.page, claim.id
        """,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # The active rebuilt graph stores filing year on the relationship and
        # document node; older claim records may leave claim.year null. The
        # filing ID is still frozen and unambiguous, so derive the disclosure
        # year rather than silently dropping all temporal questions.
        if row.get("year") is None:
            doc_id = str(row.get("doc_id") or "")
            if doc_id[:4].isdigit():
                row["year"] = int(doc_id[:4])
        grouped[str(row.get("metric_id"))].append(row)
    return {
        metric: _unique_rows(metric_rows, ("claim_id",))
        for metric, metric_rows in grouped.items()
        if {int(row.get("year")) for row in metric_rows if row.get("year") is not None} >= {2023, 2024, 2025}
    }


def _direct_item(row: dict[str, Any], filename: str, year: int, index: int) -> dict[str, Any]:
    source = _entity_token(row.get("source_id"))
    target = _entity_token(row.get("target_id"))
    relation = _relation_phrase(row.get("relation"))
    claim_id = str(row["claim_id"])
    page = int(row["page"])
    return {
        "id": f"SILVER-{index:03d}",
        "question_type": "direct_relation",
        "difficulty": "basic",
        "question": (
            f"In the fiscal {year} NVIDIA 10-K, what evidence-backed {relation} "
            f"relationship connects {source} to {target}?"
        ),
        "reference_answer": str(row.get("claim_text") or ""),
        "answerable": True,
        "requires_abstention": False,
        "source_filing": filename,
        "cross_filing": False,
        "expected_evidence_ids": [claim_id],
        "expected_pages": [page],
        "expected_page_keys": [_page_key(filename, page)],
        "relevant_evidence_grades": {claim_id: 2},
        "supporting_triples": [{"source": row.get("source_id"), "relation": relation, "target": row.get("target_id")}],
        "benchmark_status": "AUTO_GENERATED_SILVER",
        "generation": "strict_verbatim_evidence_claim",
    }


def _multi_item(row: dict[str, Any], filename: str, year: int, index: int) -> dict[str, Any]:
    source = _entity_token(row.get("source_id"))
    middle = _entity_token(row.get("middle_id"))
    target = _entity_token(row.get("target_id"))
    claim_ids = [str(row["claim1_id"]), str(row["claim2_id"])]
    pages = [int(row["page1"]), int(row["page2"])]
    relation1 = _relation_phrase(row.get("relation1"))
    relation2 = _relation_phrase(row.get("relation2"))
    return {
        "id": f"SILVER-{index:03d}",
        "question_type": "multi_hop",
        "difficulty": "multi_hop",
        "question": (
            f"What two-step evidence chain connects {source} to {target} through "
            f"{middle} in fiscal {year}? It uses {relation1} followed by {relation2}."
        ),
        "reference_answer": f"{row.get('claim1_text', '')} {row.get('claim2_text', '')}".strip(),
        "answerable": True,
        "requires_abstention": False,
        "source_filing": filename,
        "cross_filing": False,
        "expected_evidence_ids": claim_ids,
        "expected_pages": pages,
        "expected_page_keys": [_page_key(filename, page) for page in pages],
        "relevant_evidence_grades": {claim_id: 2 for claim_id in claim_ids},
        "supporting_triples": [
            {"source": row.get("source_id"), "relation": row.get("relation1"), "target": row.get("middle_id")},
            {"source": row.get("middle_id"), "relation": row.get("relation2"), "target": row.get("target_id")},
        ],
        "benchmark_status": "AUTO_GENERATED_SILVER",
        "generation": "strict_same_filing_two_hop_chain",
    }


def _temporal_item(metric: str, rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    ordered = sorted(
        (row for row in rows if int(row["year"]) in {2023, 2024, 2025}),
        key=lambda row: (int(row["year"]), int(row["page"]), str(row["claim_id"])),
    )
    claim_ids = [str(row["claim_id"]) for row in ordered]
    pages = [int(row["page"]) for row in ordered]
    metric_token = _entity_token(metric)
    return {
        "id": f"SILVER-{index:03d}",
        "question_type": "temporal_metric",
        "difficulty": "temporal",
        "question": (
            f"How did NVIDIA's reported {metric_token} change across fiscal 2023, "
            "2024, and 2025? Use only the three 10-K filings."
        ),
        "reference_answer": " ".join(str(row.get("claim_text") or "") for row in ordered),
        "answerable": True,
        "requires_abstention": False,
        "source_filing": None,
        "cross_filing": True,
        "expected_evidence_ids": claim_ids,
        "expected_pages": pages,
        "expected_page_keys": [_page_key(row["doc_id"], int(row["page"])) for row in ordered],
        "relevant_evidence_grades": {claim_id: 2 for claim_id in claim_ids},
        "supporting_triples": [
            {"source": "nvidia_corporation", "relation": "REPORTS_METRIC", "target": metric}
        ],
        "benchmark_status": "AUTO_GENERATED_SILVER",
        "generation": "same_metric_across_three_frozen_filings",
    }


def _unsupported_items(start_index: int) -> list[dict[str, Any]]:
    questions = [
        "What evidence-backed REPORTS_METRIC relationship connects NVIDIA_CORPORATION to DIVIDEND_PAYOUT_RATIO in fiscal 2019?",
        "What evidence-backed PRODUCES relationship connects NVIDIA_CORPORATION to QUANTUM_WIDGET_X in fiscal 2030?",
        "What evidence-backed DECREASES relationship connects EUROPE_MARKET to REVENUE in fiscal 2026?",
        "What was NVIDIA's operating margin in fiscal 2026 according to the frozen 10-K corpus?",
        "Which NVIDIA product acquired in fiscal 2022 is recorded in these three 10-K filings?",
    ]
    return [
        {
            "id": f"SILVER-{start_index + offset:03d}",
            "question_type": "unsupported",
            "difficulty": "abstention",
            "question": question,
            "reference_answer": "INSUFFICIENT_EVIDENCE",
            "answerable": False,
            "requires_abstention": True,
            "source_filing": "2025-10-K.pdf",
            "cross_filing": False,
            "expected_evidence_ids": [],
            "expected_pages": [],
            "expected_page_keys": [],
            "relevant_evidence_grades": {},
            "supporting_triples": [],
            "benchmark_status": "AUTO_GENERATED_SILVER",
            "generation": "fixed_unsupported_question_template",
        }
        for offset, question in enumerate(questions)
    ]


def build_dataset(output: Path, direct_per_filing: int = 8, multi_per_filing: int = 2, temporal_count: int = 2, seed: int = 20260909) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing Neo4j settings: {', '.join(missing)}")

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    rows: list[dict[str, Any]] = []
    try:
        with driver.session() as session:
            for doc_id, filename, year in FILINGS:
                for row in _direct_rows(session, doc_id, direct_per_filing):
                    rows.append(_direct_item(row, filename, year, len(rows) + 1))
                for row in _multi_hop_rows(session, filename, multi_per_filing):
                    rows.append(_multi_item(row, filename, year, len(rows) + 1))
            metrics = _metric_rows(session)
    finally:
        driver.close()

    for metric in sorted(metrics)[: max(temporal_count, 0)]:
        rows.append(_temporal_item(metric, metrics[metric], len(rows) + 1))
    rows.extend(_unsupported_items(len(rows) + 1))

    # The seed is recorded and used only for deterministic ordering of rows
    # within equally valid generated categories; it never changes evidence.
    random.Random(seed).shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["id"] = f"SILVER-{index:03d}"

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "schema": "strategic-graphrag-silver-benchmark/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_status": "AUTO_GENERATED_SILVER",
        "output": str(output),
        "seed": seed,
        "question_count": len(rows),
        "question_type_counts": {
            question_type: sum(row["question_type"] == question_type for row in rows)
            for question_type in sorted({row["question_type"] for row in rows})
        },
        "human_reviewed": False,
        "source": "strict VERBATIM EvidenceClaim graph and deterministic unsupported templates",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the automatic Silver retrieval benchmark")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--direct-per-filing", type=int, default=8)
    parser.add_argument("--multi-per-filing", type=int, default=2)
    parser.add_argument("--temporal-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output, args.direct_per_filing, args.multi_per_filing, args.temporal_count, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
