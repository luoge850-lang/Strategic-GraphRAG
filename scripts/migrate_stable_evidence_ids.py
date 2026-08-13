"""Dry-run or migrate active EvidenceClaim IDs to deterministic v2 IDs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategic_graphrag.provenance import evidence_identity

ACTIVE_FILINGS = ["2023-10-K.pdf", "2024-10-K.pdf", "2025-10-K.pdf"]


def build_plan(session) -> list[dict]:
    rows = session.run(
        """
        MATCH (c:EvidenceClaim)
        WHERE c.doc_id + '.pdf' IN $filings AND c.verification_status = 'VERBATIM'
        OPTIONAL MATCH (s:Sentence)<-[:SUPPORTED_BY]-(c)
        RETURN c.id AS old_claim_id, c.relation_id AS old_relation_id,
               s.id AS old_sentence_id, c.document_sha256 AS document_sha256,
               c.doc_id + '.pdf' AS filename, c.page AS page, c.text AS text,
               c.source_id AS source_id, c.relation_type AS relation_type,
               c.target_id AS target_id
        ORDER BY filename, page, old_claim_id
        """,
        filings=ACTIVE_FILINGS,
    )
    plan = []
    for row in rows:
        item = dict(row)
        identity = evidence_identity(
            document_sha256=item["document_sha256"],
            filename=item["filename"],
            page=item["page"],
            evidence_text=item["text"],
            source_id=item["source_id"],
            relation_type=item["relation_type"],
            target_id=item["target_id"],
        )
        item.update(
            new_claim_id=identity.claim_id,
            new_relation_id=identity.relation_id,
            new_sentence_id=identity.sentence_id,
        )
        plan.append(item)
    return plan


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
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            plan = build_plan(session)
            new_ids = [item["new_claim_id"] for item in plan]
            collisions = sorted({value for value in new_ids if new_ids.count(value) > 1})
            report = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "apply" if args.apply else "dry-run",
                "active_filings": ACTIVE_FILINGS,
                "claims_planned": len(plan),
                "claims_already_stable": sum(
                    item["old_claim_id"] == item["new_claim_id"] for item in plan
                ),
                "new_id_collisions": collisions,
                "changes": plan,
            }
            if args.apply:
                if collisions:
                    raise RuntimeError(f"Stable-ID collisions found: {collisions[:5]}")
                pending = [
                    item for item in plan
                    if item["old_claim_id"] != item["new_claim_id"]
                ]
                result = session.run(
                    """
                    UNWIND $changes AS item
                    MATCH (c:EvidenceClaim {id:item.old_claim_id})
                    OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:Sentence)
                    OPTIONAL MATCH (c)-[:ABOUT_SOURCE]->(source)
                    OPTIONAL MATCH (c)-[:ABOUT_TARGET]->(target)
                    OPTIONAL MATCH (source)-[r]->(target)
                    WHERE r.evidence_id = item.old_claim_id
                    SET c.id=item.new_claim_id, c.relation_id=item.new_relation_id,
                        c.claim_id_version='v2',
                        s.id=item.new_sentence_id,
                        r.id=item.new_relation_id, r.evidence_id=item.new_claim_id
                    RETURN count(DISTINCT c) AS applied
                    """,
                    changes=pending,
                ).single()
                report["applied"] = int(result["applied"] if result else 0)
                report["skipped_already_stable"] = len(plan) - len(pending)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in report.items() if k != "changes"}, ensure_ascii=False, indent=2))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
