"""Build bitemporal, evidence-preserving fact versions and observed changes.

Valid time describes the fiscal period referenced by the filing. Recorded time
describes when this knowledge base stored a version. A later disclosure closes
the record version but never asserts that the earlier statement became false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
MODEL_VERSION = "bitemporal_fact_v2"


def parse_number(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value or "").strip().replace(",", "").replace("$", "").replace("%", "")
    negative = text.startswith("(") and text.endswith(")")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.strip("()"))
    if not match:
        return None
    number = float(match.group())
    return -abs(number) if negative else number


def metric_for_year(raw: Any, year: int):
    try:
        values = json.loads(raw or "[]") if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return None
    for item in values if isinstance(values, list) else []:
        if str(year) in str(item.get("period", item.get("fiscal_period", ""))):
            return parse_number(item.get("value"))
    return None


def _observation_value(row: Dict[str, Any], side: str, year: int) -> Optional[float]:
    observations = row.get(f"{side}_observations") or []
    if observations:
        target_id = str(row.get("target_id") or "").casefold()
        compatible = [
            item for item in observations
            if str(item.get("metric_id") or target_id).casefold() == target_id
        ]
        return metric_for_year(compatible, year)
    return metric_for_year(row.get(f"{side}_metric_values_json"), year)


def classify(row: dict) -> dict:
    earlier_year, later_year = int(row["earlier_year"]), int(row["later_year"])
    result = {
        "change_type": "CONTINUED_DISCLOSURE" if later_year - earlier_year == 1 else "RECURRED_DISCLOSURE",
        "quantitative": False,
        "from_value": None,
        "to_value": None,
        "absolute_delta": None,
        "percent_delta": None,
        "semantics": "same normalized claim disclosed in a later filing; no direction inferred",
    }
    if row.get("relation_type") != "REPORTS_METRIC":
        return result
    earlier_unit = str(row.get("earlier_unit") or "").strip().casefold()
    later_unit = str(row.get("later_unit") or "").strip().casefold()
    earlier_text = str(row.get("earlier_text") or "")
    later_text = str(row.get("later_text") or "")
    signatures_match = (
        ("$" in earlier_text) == ("$" in later_text)
        and ("%" in earlier_text) == ("%" in later_text)
    )
    if not earlier_unit or earlier_unit != later_unit or not signatures_match:
        result.update(
            change_type="METRIC_NOT_COMPARABLE",
            semantics="metric labels repeat, but unit or evidence measurement signatures differ",
        )
        return result
    earlier = _observation_value(row, "earlier", earlier_year)
    later = _observation_value(row, "later", later_year)
    if earlier is None or later is None:
        result.update(
            change_type="METRIC_CHANGE_UNAVAILABLE",
            semantics="metric repeated but comparable fiscal-year observations were unavailable",
        )
        return result
    delta = later - earlier
    tolerance = max(abs(earlier), abs(later), 1.0) * 1e-9
    result.update({
        "change_type": "METRIC_INCREASED" if delta > tolerance else "METRIC_DECREASED" if delta < -tolerance else "METRIC_UNCHANGED",
        "quantitative": True,
        "from_value": earlier,
        "to_value": later,
        "absolute_delta": delta,
        "percent_delta": (delta / abs(earlier) * 100.0) if earlier else None,
        "semantics": "observed fiscal-year metric change computed from two verbatim table claims",
    })
    return result


def _fact_key(row: Dict[str, Any]) -> str:
    payload = "|".join(
        str(row.get(key) or "") for key in ("source_id", "relation_type", "target_id")
    )
    return "FK_" + hashlib.sha256(payload.encode()).hexdigest()[:20].upper()


def build_fact_versions(
    claims: Iterable[Dict[str, Any]],
    *,
    migration_recorded_at: Optional[str] = None,
):
    """Create one immutable TemporalFact version per verbatim claim."""
    fallback = migration_recorded_at or datetime.now(timezone.utc).isoformat()
    grouped = defaultdict(list)
    for source in claims:
        row = dict(source)
        row["fact_key"] = _fact_key(row)
        grouped[row["fact_key"]].append(row)

    facts = []
    for fact_key, versions in grouped.items():
        versions.sort(key=lambda item: (
            int(item.get("filing_fiscal_year") or item.get("fiscal_year") or 0),
            int(item.get("page") or 0),
            str(item.get("claim_id") or ""),
        ))
        for index, row in enumerate(versions):
            next_row = versions[index + 1] if index + 1 < len(versions) else None
            recorded_from = str(row.get("recorded_from") or fallback)
            valid_period = str(
                row.get("metric_period")
                or row.get("evidence_referenced_period")
                or f"FY{row.get('filing_fiscal_year') or row.get('fiscal_year')}"
            )
            claim_id = str(row["claim_id"])
            facts.append({
                "id": "TF_" + hashlib.sha256(f"{claim_id}|{MODEL_VERSION}".encode()).hexdigest()[:24].upper(),
                "fact_key": fact_key,
                "claim_id": claim_id,
                "source_id": row.get("source_id"),
                "relation_type": row.get("relation_type"),
                "target_id": row.get("target_id"),
                "source_filing": row.get("source_filing") or row.get("doc_id"),
                "page": int(row.get("page") or 0),
                "disclosure_order": int(row.get("filing_fiscal_year") or row.get("fiscal_year") or 0),
                "valid_from": valid_period,
                "valid_to": valid_period if row.get("relation_type") == "REPORTS_METRIC" else None,
                "valid_time_precision": "FISCAL_PERIOD",
                "recorded_from": recorded_from,
                "recorded_to": (
                    str(next_row.get("recorded_from"))
                    if next_row and next_row.get("recorded_from")
                    and str(next_row.get("recorded_from")) != recorded_from else None
                ),
                "recorded_time_precision": "EXACT_INGESTION_TIME" if row.get("recorded_from") else "MIGRATION_TIME",
                "is_current_record": next_row is None,
                "invalidation_status": "ACTIVE_CURRENT" if next_row is None else "SUPERSEDED_DISCLOSURE",
                "invalidation_reason": None if next_row is None else "later disclosure version available",
                "next_claim_id": str(next_row.get("claim_id")) if next_row else None,
                "truth_status": "DISCLOSED_FACT",
                "model_version": MODEL_VERSION,
            })
    return facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            claims = [dict(row) for row in session.run(
                """
                MATCH (claim:EvidenceClaim)
                WHERE claim.verification_status='VERBATIM'
                RETURN claim.id AS claim_id, claim.source_id AS source_id,
                       claim.relation_type AS relation_type, claim.target_id AS target_id,
                       claim.doc_id AS doc_id, claim.page AS page,
                       claim.fiscal_year AS fiscal_year,
                       claim.filing_fiscal_year AS filing_fiscal_year,
                       claim.evidence_referenced_period AS evidence_referenced_period,
                       claim.metric_period AS metric_period,
                       toString(claim.recorded_from) AS recorded_from
                """
            )]
            facts = build_fact_versions(claims, migration_recorded_at=generated_at)
            rows = [dict(row) for row in session.run(
                """
                MATCH (earlier:EvidenceClaim)-[:NEXT_DISCLOSURE]->(later:EvidenceClaim)
                WHERE earlier.verification_status='VERBATIM' AND later.verification_status='VERBATIM'
                OPTIONAL MATCH (eo:FinancialObservation)-[:SUPPORTED_BY_CLAIM]->(earlier)
                WITH earlier, later, collect(eo{.metric_id, .fiscal_period, .value}) AS earlier_observations
                OPTIONAL MATCH (lo:FinancialObservation)-[:SUPPORTED_BY_CLAIM]->(later)
                RETURN earlier.id AS earlier_claim_id, later.id AS later_claim_id,
                       earlier.filing_fiscal_year AS earlier_year, later.filing_fiscal_year AS later_year,
                       earlier.source_id AS source_id, earlier.relation_type AS relation_type,
                       earlier.target_id AS target_id,
                       earlier.evidence_referenced_period AS earlier_valid_period,
                       later.evidence_referenced_period AS later_valid_period,
                       earlier.metric_values_json AS earlier_metric_values_json,
                       later.metric_values_json AS later_metric_values_json,
                       earlier.text AS earlier_text, later.text AS later_text,
                       earlier.metric_unit AS earlier_unit, later.metric_unit AS later_unit,
                       earlier_observations,
                       collect(lo{.metric_id, .fiscal_period, .value}) AS later_observations
                ORDER BY source_id, relation_type, target_id, earlier_year
                """
            )]
            changes = []
            for row in rows:
                change = {**row, **classify(row)}
                # Observation maps are calculation inputs, not Neo4j scalar
                # properties. Their source nodes are already linked through
                # the two EvidenceClaims.
                change.pop("earlier_observations", None)
                change.pop("later_observations", None)
                change["id"] = "TC_" + hashlib.sha256(
                    f'{row["earlier_claim_id"]}|{row["later_claim_id"]}|{MODEL_VERSION}'.encode()
                ).hexdigest()[:24].upper()
                change["model_version"] = MODEL_VERSION
                changes.append(change)

            applied_facts = applied_changes = 0
            if args.apply:
                session.run(
                    "MATCH (fact:TemporalFact {model_version:$version}) DETACH DELETE fact",
                    version=MODEL_VERSION,
                ).consume()
                record = session.run(
                    """
                    UNWIND $facts AS item
                    MATCH (claim:EvidenceClaim {id:item.claim_id})
                    MERGE (fact:TemporalFact {id:item.id})
                    SET fact += item,
                        fact.recorded_from=datetime(item.recorded_from),
                        fact.recorded_to=CASE WHEN item.recorded_to IS NULL THEN null ELSE datetime(item.recorded_to) END
                    MERGE (fact)-[:SUPPORTED_BY_CLAIM]->(claim)
                    WITH fact, item
                    OPTIONAL MATCH (document:Document {doc_id:replace(item.source_filing, '.pdf', '')})
                    FOREACH (_ IN CASE WHEN document IS NULL THEN [] ELSE [1] END |
                        MERGE (fact)-[:DISCLOSED_IN]->(document))
                    RETURN count(fact) AS applied
                    """,
                    facts=facts,
                ).single()
                applied_facts = int(record["applied"] if record else 0)
                session.run(
                    """
                    UNWIND $facts AS item
                    WITH item WHERE item.next_claim_id IS NOT NULL
                    MATCH (earlier:TemporalFact {id:item.id})
                    MATCH (later:TemporalFact {claim_id:item.next_claim_id, model_version:item.model_version})
                    MERGE (earlier)-[:NEXT_VERSION]->(later)
                    MERGE (earlier)-[:INVALIDATED_BY]->(later)
                    """,
                    facts=facts,
                ).consume()
                session.run(
                    "MATCH (change:TemporalChange {model_version:$version}) DETACH DELETE change",
                    version=MODEL_VERSION,
                ).consume()
                record = session.run(
                    """
                    UNWIND $changes AS item
                    MATCH (earlier:EvidenceClaim {id:item.earlier_claim_id})
                    MATCH (later:EvidenceClaim {id:item.later_claim_id})
                    MATCH (earlierFact:TemporalFact {claim_id:item.earlier_claim_id, model_version:item.model_version})
                    MATCH (laterFact:TemporalFact {claim_id:item.later_claim_id, model_version:item.model_version})
                    MERGE (change:TemporalChange {id:item.id})
                    SET change += item,
                        change.valid_time_from=item.earlier_valid_period,
                        change.valid_time_to=item.later_valid_period,
                        change.recorded_from=earlierFact.recorded_from,
                        change.recorded_to=laterFact.recorded_from,
                        change.created_at=datetime()
                    MERGE (earlier)-[:HAS_TEMPORAL_CHANGE]->(change)
                    MERGE (change)-[:CHANGES_TO]->(later)
                    MERGE (earlierFact)-[:HAS_TEMPORAL_CHANGE]->(change)
                    MERGE (change)-[:CHANGES_TO_FACT]->(laterFact)
                    RETURN count(change) AS applied
                    """,
                    changes=changes,
                ).single()
                applied_changes = int(record["applied"] if record else 0)
    finally:
        driver.close()

    report = {
        "schema": "strategic-graphrag-bitemporal-facts/v2",
        "generated_at_utc": generated_at,
        "model_version": MODEL_VERSION,
        "mode": "apply" if args.apply else "dry-run",
        "evidence_claims": len(claims),
        "temporal_facts": len(facts),
        "applied_facts": applied_facts,
        "input_disclosure_links": len(rows),
        "change_nodes": len(changes),
        "applied_changes": applied_changes,
        "by_change_type": dict(Counter(item["change_type"] for item in changes)),
        "by_invalidation_status": dict(Counter(item["invalidation_status"] for item in facts)),
        "guardrails": [
            "valid time and recorded time are separate",
            "migration timestamps are labeled instead of fabricated",
            "no resolution from absence",
            "superseded disclosure does not mean real-world falsity",
            "every fact and change links to verbatim EvidenceClaims",
        ],
    }
    if args.include_records:
        report["facts"] = facts
        report["changes"] = changes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"facts", "changes"}}, indent=2))


if __name__ == "__main__":
    main()
