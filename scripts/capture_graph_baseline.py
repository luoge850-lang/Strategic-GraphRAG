"""Capture a machine-readable, read-only graph/vector baseline."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_FILINGS = ["2023-10-K.pdf", "2024-10-K.pdf", "2025-10-K.pdf"]


def _rows(session, cypher: str, **params):
    return [record.data() for record in session.run(cypher, **params)]


def capture() -> dict:
    load_dotenv(ROOT / ".env")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        graph = {
            "nodes_by_label": _rows(
                session,
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label",
            ),
            "relationships_by_type": _rows(
                session,
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type",
            ),
            "documents": _rows(
                session,
                "MATCH (d:Document) RETURN d.doc_id AS doc_id, d.filename AS filename, "
                "d.fiscal_year AS fiscal_year, d.total_pages AS total_pages, "
                "d.pdf_sha256 AS pdf_sha256 ORDER BY fiscal_year, filename",
            ),
            "active_claims": _rows(
                session,
                "MATCH (c:EvidenceClaim) WHERE c.doc_id IN $doc_ids "
                "RETURN c.doc_id AS doc_id, count(*) AS claims, "
                "count(DISTINCT c.page) AS distinct_pages, "
                "sum(CASE WHEN c.chunk_id IS NULL OR trim(c.chunk_id) = '' THEN 1 ELSE 0 END) AS missing_chunk_id, "
                "sum(CASE WHEN c.verification_status = 'VERBATIM' THEN 1 ELSE 0 END) AS verbatim "
                "ORDER BY doc_id",
                doc_ids=[value.replace(".pdf", "") for value in ACTIVE_FILINGS],
            ),
            "active_claim_sections": _rows(
                session,
                "MATCH (c:EvidenceClaim) WHERE c.doc_id IN $doc_ids "
                "RETURN c.doc_id AS doc_id, coalesce(c.section, 'UNSPECIFIED') AS section, "
                "count(*) AS claims, count(DISTINCT c.page) AS distinct_pages "
                "ORDER BY doc_id, section",
                doc_ids=[value.replace(".pdf", "") for value in ACTIVE_FILINGS],
            ),
            "active_extraction_methods": _rows(
                session,
                "MATCH (c:EvidenceClaim) WHERE c.doc_id IN $doc_ids "
                "RETURN coalesce(c.extraction_method, 'UNSPECIFIED') AS extraction_method, count(*) AS claims "
                "ORDER BY extraction_method",
                doc_ids=[value.replace(".pdf", "") for value in ACTIVE_FILINGS],
            ),
            "active_entity_nodes": _rows(
                session,
                "MATCH (s)-[r]->(t) WHERE coalesce(r.source_filing, r.filing, '') IN $filings "
                "AND r.evidence_id IS NOT NULL AND EXISTS { "
                "MATCH (c:EvidenceClaim {id: r.evidence_id}) WHERE c.verification_status = 'VERBATIM' } "
                "WITH collect(DISTINCT s) + collect(DISTINCT t) AS nodes UNWIND nodes AS n "
                "WITH DISTINCT n UNWIND labels(n) AS label "
                "RETURN label, count(*) AS nodes ORDER BY label",
                filings=ACTIVE_FILINGS,
            ),
            "active_data_quality": _rows(
                session,
                "MATCH (c:EvidenceClaim) WHERE c.doc_id IN $doc_ids "
                "OPTIONAL MATCH ()-[r]->() WHERE r.evidence_id = c.id "
                "WITH c, count(r) AS linked_edges "
                "RETURN count(*) AS claims, "
                "sum(CASE WHEN linked_edges = 0 THEN 1 ELSE 0 END) AS orphan_claims, "
                "sum(CASE WHEN c.section IS NULL OR trim(c.section) = '' THEN 1 ELSE 0 END) AS missing_section, "
                "sum(CASE WHEN c.fiscal_year IS NULL THEN 1 ELSE 0 END) AS missing_fiscal_year, "
                "sum(CASE WHEN c.evidence_referenced_period IS NULL OR trim(c.evidence_referenced_period) = '' THEN 1 ELSE 0 END) AS missing_referenced_period, "
                "sum(CASE WHEN c.temporal_model_version IS NULL OR trim(c.temporal_model_version) = '' THEN 1 ELSE 0 END) AS missing_temporal_model_version",
                doc_ids=[value.replace(".pdf", "") for value in ACTIVE_FILINGS],
            ),
            "active_duplicate_claim_groups": _rows(
                session,
                "MATCH (c:EvidenceClaim) WHERE c.doc_id IN $doc_ids "
                "WITH c.doc_id AS doc_id, c.source_id AS source_id, c.target_id AS target_id, "
                "c.relation_type AS relation_type, c.page AS page, c.text AS text, count(*) AS copies "
                "WHERE copies > 1 RETURN count(*) AS duplicate_groups, sum(copies - 1) AS excess_claims",
                doc_ids=[value.replace(".pdf", "") for value in ACTIVE_FILINGS],
            ),
            "active_edges": _rows(
                session,
                "MATCH ()-[r]->() WHERE coalesce(r.source_filing, r.filing, '') IN $filings "
                "RETURN coalesce(r.source_filing, r.filing, '') AS filing, type(r) AS type, "
                "count(*) AS count, sum(CASE WHEN r.evidence_id IS NULL THEN 1 ELSE 0 END) AS missing_evidence_id "
                "ORDER BY filing, type",
                filings=ACTIVE_FILINGS,
            ),
            "legacy_or_unlinked_business_edges": _rows(
                session,
                "MATCH ()-[r]->() WHERE type(r) IN $business_types "
                "AND (r.evidence_id IS NULL OR NOT coalesce(r.source_filing, r.filing, '') IN $filings) "
                "RETURN coalesce(r.source_filing, r.filing, 'UNSCOPED') AS filing, "
                "type(r) AS type, count(*) AS count ORDER BY filing, type",
                filings=ACTIVE_FILINGS,
                business_types=[
                    "CAUSES", "TRIGGERS", "AMPLIFIES", "INCREASES", "DECREASES",
                    "MITIGATES", "CONSTRAINS", "EXPOSED_TO", "IMPLEMENTS",
                    "OPERATES_IN", "PRODUCES", "COMPETES_WITH", "DEPENDS_ON",
                    "REGULATED_BY", "SUPPLIES_TO", "REPORTS_METRIC",
                ],
            ),
        }
    driver.close()

    client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma_db"))
    collections = []
    for collection in client.list_collections():
        collections.append({"name": collection.name, "count": collection.count()})
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_filings": ACTIVE_FILINGS,
        "active_filing_env": os.getenv("GRAPH_ACTIVE_FILING"),
        "vector_collection_env": os.getenv("GRAPH_VECTOR_COLLECTION"),
        "graph": graph,
        "vector_collections": sorted(collections, key=lambda item: item["name"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "documents": len(report["graph"]["documents"]), "vector_collections": report["vector_collections"]}, indent=2))


if __name__ == "__main__":
    main()
