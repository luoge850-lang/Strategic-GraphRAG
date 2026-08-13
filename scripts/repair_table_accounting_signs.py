"""Repair structured signs for active table claims without rebuilding PDFs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from strategic_graphrag.pipeline.financial_table_extractor import _numeric_values


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_DOC_IDS = ["2023-10-K", "2024-10-K", "2025-10-K"]


def repair(*, apply: bool) -> dict:
    load_dotenv(ROOT / ".env", override=True)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    changes = []
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        rows = [
            record.data()
            for record in session.run(
                "MATCH (c:EvidenceClaim) "
                "WHERE c.doc_id IN $doc_ids AND c.extraction_method = 'TABLE_EXTRACTION' "
                "RETURN c.id AS id, c.doc_id AS doc_id, c.page AS page, c.text AS text, "
                "c.metric_values_json AS metric_values_json ORDER BY doc_id, page, id",
                doc_ids=ACTIVE_DOC_IDS,
            )
        ]
        for row in rows:
            try:
                old_values = json.loads(row.get("metric_values_json") or "[]")
            except json.JSONDecodeError:
                continue
            parsed = _numeric_values(row.get("text") or "")
            if not old_values or len(parsed) < len(old_values):
                continue
            new_values = [
                {"period": str(item.get("period", "")), "value": parsed[index]}
                for index, item in enumerate(old_values)
            ]
            if new_values == old_values:
                continue
            change = {
                "claim_id": row["id"],
                "doc_id": row["doc_id"],
                "page": row["page"],
                "old_values": old_values,
                "new_values": new_values,
            }
            if apply:
                result = session.run(
                    "MATCH (c:EvidenceClaim {id: $id}) "
                    "SET c.metric_values_json = $values, c.metric_value = $first "
                    "WITH c OPTIONAL MATCH ()-[r]->() WHERE r.evidence_id = c.id "
                    "SET r.metric_values_json = $values, r.metric_value = $first "
                    "RETURN count(r) AS edges_updated",
                    id=row["id"],
                    values=json.dumps(new_values, ensure_ascii=False),
                    first=new_values[0]["value"],
                ).single()
                change["edges_updated"] = int(result["edges_updated"] if result else 0)
            changes.append(change)
    driver.close()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "active_doc_ids": ACTIVE_DOC_IDS,
        "claims_changed": len(changes),
        "edges_changed": sum(item.get("edges_updated", 0) for item in changes),
        "changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = repair(apply=args.apply)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("mode", "claims_changed", "edges_changed")}, indent=2))


if __name__ == "__main__":
    main()
