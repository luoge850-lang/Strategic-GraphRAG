"""Create an honest, reproducible extraction-quality baseline.

Automated checks measure provenance and schema conformance. Semantic entity /
relation precision and missing-relation recall are intentionally reported as
NOT_MEASURED until the generated stratified sample receives independent labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
PDFS = {
    "2023-10-K": ROOT / "data/pdfs_other/2023-10-K.pdf",
    "2024-10-K": ROOT / "data/pdfs_other/2024-10-K.pdf",
    "2025-10-K": ROOT / "data/pdfs/2025-10-K.pdf",
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("\u00ad", "").strip()).casefold()


def page_contains(pdf: pymupdf.Document, page: int, evidence: str) -> bool:
    if page < 1 or page > len(pdf) or not evidence:
        return False
    page_text = normalize(pdf[page - 1].get_text("text"))
    quote = normalize(evidence)
    if quote in page_text:
        return True
    # PDF line wrapping/hyphenation can differ from stored normalized evidence.
    compact_page = re.sub(r"[^a-z0-9$%.-]+", "", page_text)
    compact_quote = re.sub(r"[^a-z0-9$%.-]+", "", quote)
    return len(compact_quote) >= 20 and compact_quote in compact_page


def stratified_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("doc_id"), row.get("section") or "UNSPECIFIED", row.get("extraction_method") or "UNSPECIFIED")].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = [bucket.pop() for bucket in buckets.values() if bucket]
    remaining = [item for bucket in buckets.values() for item in bucket]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, size - len(selected))])
    return selected[:size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            rows = [dict(row) for row in session.run(
                """
                MATCH (c:EvidenceClaim)
                WHERE c.doc_id IN $doc_ids AND c.verification_status='VERBATIM'
                OPTIONAL MATCH (c)-[:ABOUT_SOURCE]->(source)
                OPTIONAL MATCH (c)-[:ABOUT_TARGET]->(target)
                OPTIONAL MATCH (source)-[r]->(target) WHERE r.evidence_id=c.id
                RETURN c.id AS id, c.doc_id AS doc_id, c.page AS page,
                       c.section AS section, c.text AS text,
                       c.source_id AS source_id, c.target_id AS target_id,
                       c.relation_type AS relation_type,
                       c.extraction_method AS extraction_method,
                       c.evidence_char_start AS evidence_char_start,
                       c.evidence_char_end AS evidence_char_end,
                       c.chunk_id AS chunk_id, c.document_sha256 AS document_sha256,
                       c.filing_fiscal_year AS filing_fiscal_year,
                       c.evidence_referenced_period AS evidence_referenced_period,
                       c.metric_values_json AS metric_values_json,
                       count(r) AS linked_edges
                ORDER BY c.doc_id, c.page, c.id
                """,
                doc_ids=list(PDFS),
            )]
    finally:
        driver.close()

    documents = {doc_id: pymupdf.open(path) for doc_id, path in PDFS.items()}
    try:
        exact = []
        for row in rows:
            row["verbatim_on_declared_page"] = page_contains(documents[row["doc_id"]], int(row.get("page") or 0), row.get("text") or "")
            exact.append(row["verbatim_on_declared_page"])
    finally:
        for document in documents.values():
            document.close()

    duplicates = Counter((r["doc_id"], r["page"], r["source_id"], r["relation_type"], r["target_id"], normalize(r["text"])) for r in rows)
    required = ("id", "doc_id", "page", "text", "source_id", "target_id", "relation_type", "document_sha256", "filing_fiscal_year")
    complete = [all(row.get(field) not in (None, "") for field in required) for row in rows]
    linked = [int(row.get("linked_edges") or 0) == 1 for row in rows]
    sample = stratified_sample(rows, args.sample_size, args.seed)
    annotation_rows = []
    for row in sample:
        annotation_rows.append({
            "claim_id": row["id"], "doc_id": row["doc_id"], "page": row["page"],
            "section": row.get("section"), "extraction_method": row.get("extraction_method"),
            "source_id": row["source_id"], "relation_type": row["relation_type"], "target_id": row["target_id"],
            "evidence": row["text"], "verbatim_on_declared_page": row["verbatim_on_declared_page"],
            "labels": {"source_entity_correct": None, "target_entity_correct": None, "relation_correct": None, "evidence_supports_relation": None, "missing_gold_relations": None},
            "annotation_status": "UNLABELED", "annotator": None, "notes": "",
        })
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    args.sample.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in annotation_rows) + "\n", encoding="utf-8")

    report = {
        "schema": "strategic-graphrag-extraction-quality/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": list(PDFS), "claim_count": len(rows),
        "automated_metrics": {
            "verbatim_page_match_rate": sum(exact) / len(exact) if exact else 0,
            "required_provenance_completeness": sum(complete) / len(complete) if complete else 0,
            "exactly_one_linked_business_edge_rate": sum(linked) / len(linked) if linked else 0,
            "duplicate_excess_claims": sum(count - 1 for count in duplicates.values() if count > 1),
        },
        "distribution": {
            "by_filing": dict(Counter(row["doc_id"] for row in rows)),
            "by_section": dict(Counter(row.get("section") or "UNSPECIFIED" for row in rows)),
            "by_extraction_method": dict(Counter(row.get("extraction_method") or "UNSPECIFIED" for row in rows)),
        },
        "semantic_metrics": {
            "entity_precision": "NOT_MEASURED", "relation_precision": "NOT_MEASURED",
            "evidence_support_precision": "NOT_MEASURED", "relation_recall": "NOT_MEASURED",
            "reason": "Independent labels are required; automated self-scoring would inflate quality.",
        },
        "annotation_sample": {"path": str(args.sample), "rows": len(annotation_rows), "labeled": 0, "seed": args.seed},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
