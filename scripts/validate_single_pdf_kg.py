"""Validate one filing against the evidence-grounded Neo4j contract.

This is a read-only diagnostic. It does not reset, delete, or mutate Neo4j.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategic_graphrag.ontology.relation_inference import validate_triple


def run_validation(doc_id: str, filename: str) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    with driver.session() as session:
        claims = [
            record.data()
            for record in session.run(
                """
                MATCH (c:EvidenceClaim {doc_id: $doc_id})
                MATCH (c)-[:ABOUT_SOURCE]->(src)
                MATCH (c)-[:ABOUT_TARGET]->(tgt)
                RETURN c.id AS claim_id,
                       c.relation_id AS relation_id,
                       c.relation_type AS relation,
                       c.page AS page,
                       c.text AS text,
                       labels(src)[0] AS source_category,
                       src.id AS source,
                       labels(tgt)[0] AS target_category,
                       tgt.id AS target
                """,
                doc_id=doc_id,
            )
        ]

        claim_links = session.run(
            """
            MATCH (c:EvidenceClaim {doc_id: $doc_id})
            OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:Sentence)
            OPTIONAL MATCH (c)-[:ABOUT_SOURCE]->(src)
            OPTIONAL MATCH (c)-[:ABOUT_TARGET]->(tgt)
            RETURN count(c) AS claims,
                   count(s) AS sentences,
                   count(src) AS sources,
                   count(tgt) AS targets,
                   sum(CASE WHEN s IS NULL THEN 1 ELSE 0 END) AS missing_sentence,
                   sum(CASE WHEN src IS NULL THEN 1 ELSE 0 END) AS missing_source,
                   sum(CASE WHEN tgt IS NULL THEN 1 ELSE 0 END) AS missing_target,
                   sum(CASE WHEN s IS NOT NULL AND c.page <> s.page THEN 1 ELSE 0 END) AS page_mismatch,
                   sum(CASE WHEN s IS NOT NULL AND c.text <> s.text THEN 1 ELSE 0 END) AS text_mismatch
            """,
            doc_id=doc_id,
        ).single().data()

        edge_join = session.run(
            """
            MATCH (c:EvidenceClaim {doc_id: $doc_id})
            OPTIONAL MATCH ()-[r]->()
            WHERE r.id = c.relation_id
            RETURN count(c) AS claims,
                   sum(CASE WHEN r IS NULL THEN 1 ELSE 0 END) AS claims_without_edge,
                   sum(CASE WHEN r.year IS NULL THEN 1 ELSE 0 END) AS missing_year,
                   sum(CASE WHEN r.source_page IS NULL THEN 1 ELSE 0 END) AS missing_page,
                   sum(CASE WHEN r.evidence_id <> c.id THEN 1 ELSE 0 END) AS evidence_id_mismatch
            """,
            doc_id=doc_id,
        ).single().data()

        document_year = session.run(
            """
            MATCH (d:Document {doc_id: $doc_id})-[:REPORTS]->(y:Year)
            RETURN count(d) AS documents, count(y) AS years,
                   collect(DISTINCT y.year) AS linked_years
            """,
            doc_id=doc_id,
        ).single().data()

        legacy_edges = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.source_filing = $filename
            OPTIONAL MATCH (c:EvidenceClaim {id: r.evidence_id})
            RETURN count(r) AS filing_edges,
                   sum(CASE WHEN c IS NULL THEN 1 ELSE 0 END) AS legacy_unlinked_edges
            """,
            filename=filename,
        ).single().data()

    driver.close()

    invalid = []
    labels = Counter()
    relations = Counter()
    for claim in claims:
        labels[claim["source_category"]] += 1
        labels[claim["target_category"]] += 1
        relations[claim["relation"]] += 1
        valid, reason = validate_triple(
            claim["source_category"],
            claim["target_category"],
            claim["relation"],
            claim["source"],
            claim["target"],
        )
        if not valid:
            invalid.append({"claim_id": claim["claim_id"], "reason": reason})

    checks = {
        "has_claims": claim_links["claims"] > 0,
        "all_claims_have_sentence": claim_links["missing_sentence"] == 0,
        "all_claims_have_source": claim_links["missing_source"] == 0,
        "all_claims_have_target": claim_links["missing_target"] == 0,
        "page_alignment": claim_links["page_mismatch"] == 0,
        "text_alignment": claim_links["text_mismatch"] == 0,
        "all_claims_have_edge": edge_join["claims_without_edge"] == 0,
        "edge_years_present": edge_join["missing_year"] == 0,
        "edge_pages_present": edge_join["missing_page"] == 0,
        "evidence_ids_aligned": edge_join["evidence_id_mismatch"] == 0,
        "document_reports_year": document_year["documents"] == 1 and document_year["years"] == 1,
        "all_triples_validate": not invalid,
    }

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "doc_id": doc_id,
        "filename": filename,
        "checks": checks,
        "claim_links": dict(claim_links),
        "edge_join": dict(edge_join),
        "document_year": dict(document_year),
        "entity_label_mentions": dict(labels),
        "relation_counts": dict(relations),
        "invalid_triples": invalid[:20],
        "legacy_warning": dict(legacy_edges),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only single-filing KG validation")
    parser.add_argument("--doc_id", required=True, help="Document node ID, e.g. 2025-10-K")
    parser.add_argument("--filename", required=True, help="Filing filename, e.g. 2025-10-K.pdf")
    args = parser.parse_args()
    print(json.dumps(run_validation(args.doc_id, args.filename), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
