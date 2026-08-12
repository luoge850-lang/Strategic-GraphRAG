"""Validate one filing against the evidence-grounded Neo4j contract.

This is a read-only diagnostic. It does not reset, delete, or mutate Neo4j.
"""

import argparse
import json
import os
import sys
from collections import Counter
import re
from pathlib import Path

import fitz
from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategic_graphrag.ontology.relation_inference import validate_triple


def _build_page_coverage(pdf_path: Path, claims: list[dict]) -> dict:
    """Build a compact per-page coverage report without storing page text."""
    claim_by_page = Counter(
        int(claim["page"])
        for claim in claims
        if claim.get("page") is not None and str(claim["page"]).isdigit()
    )
    if not pdf_path.exists():
        return {
            "status": "MISSING_PDF",
            "pdf": str(pdf_path),
            "pages": 0,
            "pages_with_text": 0,
            "pages_with_claim": 0,
            "claim_coverage_percent": 0.0,
            "page_records": [],
        }

    page_records = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text_chars = len((page.get_text("text") or "").strip())
            page_records.append(
                {
                    "page": page_number,
                    "text_chars": text_chars,
                    "claim_count": claim_by_page.get(page_number, 0),
                    "claim_pages": claim_by_page.get(page_number, 0) > 0,
                }
            )

    pages = len(page_records)
    pages_with_text = sum(record["text_chars"] > 0 for record in page_records)
    pages_with_claim = sum(record["claim_pages"] for record in page_records)
    return {
        "status": "PASS",
        "pdf": str(pdf_path),
        "pages": pages,
        "pages_with_text": pages_with_text,
        "pages_without_text": pages - pages_with_text,
        "pages_with_claim": pages_with_claim,
        "pages_without_claim": pages - pages_with_claim,
        "claim_coverage_percent": round((pages_with_claim / pages) * 100, 2) if pages else 0.0,
        "claim_page_numbers": sorted(claim_by_page),
        "page_records": page_records,
    }


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _audit_claim_pdf_alignment(pdf_path: Path, claims: list[dict]) -> dict:
    """Verify each stored quote against the original PDF page automatically."""
    if not pdf_path.exists():
        return {"status": "MISSING_PDF", "checked": 0, "quote_mismatches": 0}
    page_text = {}
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text[page_number] = _normalize_text(page.get_text("text"))

    checked = 0
    mismatches = []
    missing_spans = []
    for claim in claims:
        page = claim.get("page")
        evidence = _normalize_text(claim.get("text"))
        if not isinstance(page, int) or not evidence:
            mismatches.append({"claim_id": claim.get("claim_id"), "reason": "MISSING_PAGE_OR_TEXT"})
            continue
        checked += 1
        if evidence not in page_text.get(page, ""):
            mismatches.append({"claim_id": claim.get("claim_id"), "reason": "QUOTE_NOT_ON_PAGE", "page": page})
        if claim.get("evidence_char_start") is None or claim.get("evidence_char_end") is None:
            missing_spans.append(claim.get("claim_id"))

    return {
        "status": "PASS" if not mismatches else "FAIL",
        "checked": checked,
        "quote_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "claims_without_char_span": len(missing_spans),
        "missing_span_claim_ids": missing_spans[:20],
    }


EVIDENCE_CHAIN_RELATIONS = {
    "OBSERVED_IN", "REPORTS", "SUPPORTS", "BELONGS_TO", "SUPPORTED_BY",
    "ABOUT_SOURCE", "ABOUT_TARGET",
}


def run_validation(doc_id: str, filename: str, pdf_path: Path | None = None) -> dict:
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
                       c.verification_status AS verification_status,
                       c.fiscal_year AS fiscal_year,
                       c.evidence_char_start AS evidence_char_start,
                       c.evidence_char_end AS evidence_char_end,
                       c.relation_polarity AS relation_polarity,
                       c.modality AS modality,
                       c.temporal_scope AS temporal_scope,
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

        scoped_edges = [
            record.data()
            for record in session.run(
                """
                MATCH (src)-[r]->(tgt)
                WHERE coalesce(r.source_filing, r.filing, '') = $filename
                OPTIONAL MATCH (c:EvidenceClaim {id: r.evidence_id})
                RETURN src.id AS source,
                       tgt.id AS target,
                       r.id AS relation_id,
                       type(r) AS relation,
                       r.evidence_id AS evidence_id,
                       r.source_page AS source_page,
                       r.evidence_sentence AS evidence_sentence,
                       r.extraction_method AS extraction_method,
                       r.confidence AS confidence,
                       c.id AS claim_id,
                       c.doc_id AS claim_doc_id,
                       c.verification_status AS verification_status
                """,
                filename=filename,
            )
        ]

    driver.close()

    invalid = []
    labels = Counter()
    relations = Counter()
    relation_categories = Counter()
    verification_statuses = Counter()
    claim_pages = []
    for claim in claims:
        labels[claim["source_category"]] += 1
        labels[claim["target_category"]] += 1
        relations[claim["relation"]] += 1
        relation_categories[
            f"{claim['source_category']} -> {claim['relation']} -> {claim['target_category']}"
        ] += 1
        verification_statuses[claim.get("verification_status")] += 1
        if claim.get("page") is not None:
            claim_pages.append(int(claim["page"]))
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
        "all_claims_verbatim": (
            claim_links["claims"] > 0
            and verification_statuses.get("VERBATIM", 0) == claim_links["claims"]
        ),
        "page_alignment": claim_links["page_mismatch"] == 0,
        "text_alignment": claim_links["text_mismatch"] == 0,
        "all_claims_have_edge": edge_join["claims_without_edge"] == 0,
        "edge_years_present": edge_join["missing_year"] == 0,
        "edge_pages_present": edge_join["missing_page"] == 0,
        "evidence_ids_aligned": edge_join["evidence_id_mismatch"] == 0,
        "document_reports_year": document_year["documents"] == 1 and document_year["years"] == 1,
        "all_triples_validate": not invalid,
    }

    unique_scoped_edges = {
        (edge.get("source"), edge.get("target"), edge.get("relation"))
        for edge in scoped_edges
    }
    scoped_edge_claims = [edge for edge in scoped_edges if edge.get("claim_id")]
    scoped_edge_verbatim = [
        edge
        for edge in scoped_edges
        if edge.get("verification_status") == "VERBATIM"
    ]
    scoped_relation_counts = Counter(edge.get("relation") for edge in scoped_edges)
    claim_doc_ids = Counter(
        edge.get("claim_doc_id")
        for edge in scoped_edges
        if edge.get("claim_doc_id")
    )
    unverified_edges = [
        edge
        for edge in scoped_edges
        if edge.get("verification_status") != "VERBATIM"
    ]
    business_edges = [
        edge for edge in scoped_edges
        if edge.get("relation") not in EVIDENCE_CHAIN_RELATIONS
    ]
    strict_business_edges = [
        edge for edge in business_edges
        if edge.get("verification_status") == "VERBATIM"
    ]
    legacy_business_edges = [
        edge for edge in business_edges
        if edge.get("verification_status") != "VERBATIM"
    ]
    pdf_path = pdf_path or PROJECT_ROOT / "data" / "pdfs" / filename
    page_coverage = _build_page_coverage(pdf_path, claims)
    pdf_alignment = _audit_claim_pdf_alignment(pdf_path, claims)

    temporal_missing = [
        claim.get("claim_id")
        for claim in claims
        if claim.get("fiscal_year") is None or not claim.get("temporal_scope")
    ]
    checks["pdf_quote_alignment"] = pdf_alignment["status"] == "PASS"
    checks["claim_years_present"] = not temporal_missing
    checks["claim_temporal_scope_present"] = not temporal_missing

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
        "relation_category_counts": dict(relation_categories),
        "verification_status_counts": dict(verification_statuses),
        "evidence_page_coverage": {
            "distinct_claim_pages": len(set(claim_pages)),
            "first_claim_page": min(claim_pages) if claim_pages else None,
            "last_claim_page": max(claim_pages) if claim_pages else None,
            "claim_pages": sorted(set(claim_pages)),
        },
        "pdf_page_coverage": page_coverage,
        "pdf_alignment": pdf_alignment,
        "temporal_audit": {
            "claims_without_temporal_metadata": len(temporal_missing),
            "claim_ids": temporal_missing[:20],
        },
        "scoped_edge_audit": {
            "scope": "coalesce(source_filing, filing) = filename",
            "edge_instances": len(scoped_edges),
            "unique_source_target_relation_edges": len(unique_scoped_edges),
            "edges_with_evidence_id": len(scoped_edge_claims),
            "edges_with_verbatim_claim": len(scoped_edge_verbatim),
            "edges_without_evidence_id": sum(
                1 for edge in scoped_edges if not edge.get("evidence_id")
            ),
            "edges_without_claim_node": sum(
                1 for edge in scoped_edges if not edge.get("claim_id")
            ),
            "relation_counts": dict(scoped_relation_counts),
            "claim_doc_ids": dict(claim_doc_ids),
            "unverified_edges": unverified_edges,
            "business_edge_instances": len(business_edges),
            "strict_business_edges": len(strict_business_edges),
            "legacy_unverified_business_edges": len(legacy_business_edges),
            "evidence_chain_edge_instances": len(scoped_edges) - len(business_edges),
        },
        "invalid_triples": invalid[:20],
        "legacy_warning": dict(legacy_edges),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only single-filing KG validation")
    parser.add_argument("--doc_id", required=True, help="Document node ID, e.g. 2025-10-K")
    parser.add_argument("--filename", required=True, help="Filing filename, e.g. 2025-10-K.pdf")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional PDF path; defaults to data/pdfs/<filename>",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=None,
        help="Optional path for the JSON coverage report",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="Optional path for the complete KG audit report",
    )
    args = parser.parse_args()
    result = run_validation(args.doc_id, args.filename, pdf_path=args.pdf)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.coverage_output:
        args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_output.write_text(
            json.dumps(result["pdf_page_coverage"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
