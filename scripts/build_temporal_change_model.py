"""Build evidence-preserving observed changes across filing years.

The model distinguishes quantitative metric changes from repeated narrative
disclosures. It never infers that a risk was resolved merely because a later
filing omits it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
MODEL_VERSION = "observed_change_v1"


def parse_number(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value or "").strip().replace(",", "").replace("$", "").replace("%", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group())
    return -number if negative else number


def metric_for_year(raw: str, year: int):
    try:
        values = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return None
    for item in values if isinstance(values, list) else []:
        if str(year) in str(item.get("period", "")):
            return parse_number(item.get("value"))
    return None


def classify(row: dict) -> dict:
    earlier_year, later_year = int(row["earlier_year"]), int(row["later_year"])
    result = {
        "change_type": "CONTINUED_DISCLOSURE" if later_year - earlier_year == 1 else "RECURRED_DISCLOSURE",
        "quantitative": False, "from_value": None, "to_value": None,
        "absolute_delta": None, "percent_delta": None,
        "semantics": "same normalized claim disclosed in a later filing; no direction inferred",
    }
    if row.get("relation_type") != "REPORTS_METRIC":
        return result
    earlier_unit = str(row.get("earlier_unit") or "").strip().casefold()
    later_unit = str(row.get("later_unit") or "").strip().casefold()
    earlier_text = str(row.get("earlier_text") or "")
    later_text = str(row.get("later_text") or "")
    currency_signature_matches = ("$" in earlier_text) == ("$" in later_text)
    percent_signature_matches = ("%" in earlier_text) == ("%" in later_text)
    if not earlier_unit or earlier_unit != later_unit or not currency_signature_matches or not percent_signature_matches:
        result["change_type"] = "METRIC_NOT_COMPARABLE"
        result["semantics"] = "metric labels repeat, but unit or evidence measurement signatures differ"
        return result
    earlier = metric_for_year(row.get("earlier_metric_values_json"), earlier_year)
    later = metric_for_year(row.get("later_metric_values_json"), later_year)
    if earlier is None or later is None:
        result["change_type"] = "METRIC_CHANGE_UNAVAILABLE"
        result["semantics"] = "metric repeated but comparable fiscal-year values were unavailable"
        return result
    delta = later - earlier
    tolerance = max(abs(earlier), abs(later), 1.0) * 1e-9
    result.update({
        "change_type": "METRIC_INCREASED" if delta > tolerance else "METRIC_DECREASED" if delta < -tolerance else "METRIC_UNCHANGED",
        "quantitative": True, "from_value": earlier, "to_value": later,
        "absolute_delta": delta,
        "percent_delta": (delta / abs(earlier) * 100.0) if earlier else None,
        "semantics": "observed fiscal-year metric change computed from two verbatim table claims",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            rows = [dict(row) for row in session.run(
                """
                MATCH (earlier:EvidenceClaim)-[link:NEXT_DISCLOSURE]->(later:EvidenceClaim)
                WHERE earlier.verification_status='VERBATIM' AND later.verification_status='VERBATIM'
                RETURN earlier.id AS earlier_claim_id, later.id AS later_claim_id,
                       earlier.filing_fiscal_year AS earlier_year, later.filing_fiscal_year AS later_year,
                       earlier.source_id AS source_id, earlier.relation_type AS relation_type,
                       earlier.target_id AS target_id,
                       earlier.evidence_referenced_period AS earlier_valid_period,
                       later.evidence_referenced_period AS later_valid_period,
                       earlier.metric_values_json AS earlier_metric_values_json,
                       later.metric_values_json AS later_metric_values_json,
                       earlier.text AS earlier_text, later.text AS later_text,
                       earlier.metric_unit AS earlier_unit, later.metric_unit AS later_unit
                ORDER BY source_id, relation_type, target_id, earlier_year
                """
            )]
            changes = []
            for row in rows:
                change = {**row, **classify(row)}
                change["id"] = "TC_" + hashlib.sha256(f'{row["earlier_claim_id"]}|{row["later_claim_id"]}|{MODEL_VERSION}'.encode()).hexdigest()[:24].upper()
                change["model_version"] = MODEL_VERSION
                changes.append(change)
            applied = 0
            if args.apply:
                session.run("MATCH (change:TemporalChange {model_version:$version}) DETACH DELETE change", version=MODEL_VERSION).consume()
                record = session.run(
                    """
                    UNWIND $changes AS item
                    MATCH (earlier:EvidenceClaim {id:item.earlier_claim_id})
                    MATCH (later:EvidenceClaim {id:item.later_claim_id})
                    MERGE (change:TemporalChange {id:item.id})
                    SET change += item,
                        change.transaction_time_from=item.earlier_year,
                        change.transaction_time_to=item.later_year,
                        change.valid_time_from=item.earlier_valid_period,
                        change.valid_time_to=item.later_valid_period,
                        change.created_at=datetime()
                    MERGE (earlier)-[:HAS_TEMPORAL_CHANGE]->(change)
                    MERGE (change)-[:CHANGES_TO]->(later)
                    RETURN count(change) AS applied
                    """,
                    changes=changes,
                ).single()
                applied = int(record["applied"] if record else 0)
    finally:
        driver.close()
    report = {
        "schema": "strategic-graphrag-temporal-change/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION, "mode": "apply" if args.apply else "dry-run",
        "input_disclosure_links": len(rows), "change_nodes": len(changes), "applied": applied,
        "by_change_type": dict(Counter(item["change_type"] for item in changes)),
        "guardrails": ["no resolution from absence", "no narrative direction without explicit comparable values", "every change links two VERBATIM EvidenceClaims"],
    }
    if args.include_records:
        report["changes"] = changes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    main()
