"""Build evidence-preserving cross-year disclosure-order links.

NEXT_DISCLOSURE means only that the same normalized triple was disclosed in a
later filing. It does not claim that the risk intensified, declined, or caused
an outcome. Those labels require a separately evaluated temporal classifier.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_FILINGS = ["2023-10-K.pdf", "2024-10-K.pdf", "2025-10-K.pdf"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    query = """
    MATCH (c:EvidenceClaim)
    WHERE c.doc_id + '.pdf' IN $filings AND c.verification_status='VERBATIM'
    WITH c.source_id AS source_id, c.relation_type AS relation_type,
         c.target_id AS target_id, c
    ORDER BY c.filing_fiscal_year, c.page, c.id
    WITH source_id, relation_type, target_id, collect(c) AS observations
    WHERE size(observations) > 1
    UNWIND range(0, size(observations)-2) AS i
    WITH observations[i] AS earlier, observations[i+1] AS later,
         source_id, relation_type, target_id
    WHERE earlier.filing_fiscal_year < later.filing_fiscal_year
    RETURN earlier.id AS earlier_claim_id, later.id AS later_claim_id,
           earlier.filing_fiscal_year AS earlier_year,
           later.filing_fiscal_year AS later_year,
           source_id, relation_type, target_id
    ORDER BY source_id, relation_type, target_id, earlier_year
    """
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            links = [dict(row) for row in session.run(query, filings=ACTIVE_FILINGS)]
            if args.apply:
                session.run("MATCH ()-[r:NEXT_DISCLOSURE]->() DELETE r").consume()
                result = session.run(
                    """
                    UNWIND $links AS item
                    MATCH (earlier:EvidenceClaim {id:item.earlier_claim_id})
                    MATCH (later:EvidenceClaim {id:item.later_claim_id})
                    MERGE (earlier)-[r:NEXT_DISCLOSURE]->(later)
                    SET r.derived=true,
                        r.semantics='same normalized triple disclosed in a later filing',
                        r.year_gap=item.later_year-item.earlier_year,
                        r.temporal_model_version='disclosure_order_v1'
                    RETURN count(r) AS applied
                    """,
                    links=links,
                ).single()
                applied = int(result["applied"] if result else 0)
            else:
                applied = 0
    finally:
        driver.close()
    report = {
        "schema": "strategic-graphrag-temporal-disclosure-index/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "semantics": "ordering of repeated disclosures, not causal evolution",
        "candidate_links": len(links),
        "applied": applied,
        "links": links,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "links"}, indent=2))


if __name__ == "__main__":
    main()
