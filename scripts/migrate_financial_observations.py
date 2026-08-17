"""Materialize FinancialObservation nodes from existing strict metric claims."""

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
sys.path.insert(0, str(ROOT))

from strategic_graphrag.schema.financial_observation import build_financial_observations
from strategic_graphrag.schema.manager import SchemaManager


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
    observations = []
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            claims = [dict(row) for row in session.run(
                """
                MATCH (claim:EvidenceClaim)
                WHERE claim.verification_status='VERBATIM'
                  AND claim.relation_type='REPORTS_METRIC'
                RETURN claim.id AS claim_id, claim.source_id AS source_id,
                       claim.target_id AS target_id, claim.doc_id AS doc_id,
                       claim.page AS page, claim.filing_fiscal_year AS filing_year,
                       claim.section AS section, claim.metric_value AS metric_value,
                       claim.metric_unit AS metric_unit, claim.metric_period AS metric_period,
                       claim.metric_values_json AS metric_values_json,
                       claim.text AS evidence_text
                ORDER BY doc_id, page, claim_id
                """
            )]
            for claim in claims:
                triple = {
                    "relation": "REPORTS_METRIC",
                    "target": claim.get("target_id"),
                    "metric_value": claim.get("metric_value"),
                    "metric_unit": claim.get("metric_unit"),
                    "metric_period": claim.get("metric_period"),
                    "metric_values_json": claim.get("metric_values_json"),
                    "row_label": claim.get("target_id"),
                    "table_name": "MIGRATED_FROM_EVIDENCE_CLAIM",
                    "statement_type": claim.get("section") or "UNKNOWN",
                    "comparability_status": "UNASSESSED",
                    "evidence_sentence": claim.get("evidence_text") or "",
                }
                observations.extend(build_financial_observations(
                    triple,
                    claim_id=claim["claim_id"],
                    company_id=claim["source_id"],
                    metric_id=claim["target_id"],
                    source_filing=f'{claim["doc_id"]}.pdf',
                    page=int(claim.get("page") or 0),
                    filing_year=int(claim.get("filing_year") or 0),
                    section=claim.get("section") or "UNKNOWN",
                ))
            applied = 0
            if args.apply and observations:
                session.run(
                    "MATCH (observation:FinancialObservation {model_version:'financial_observation_v1'}) DETACH DELETE observation"
                ).consume()
                result = session.run(
                    """
                    UNWIND $observations AS item
                    MATCH (company {id:item.company_id})
                    MERGE (metric:FinancialMetric {id:item.metric_id})
                    ON CREATE SET metric.name=replace(item.metric_id, '_', ' '),
                                  metric.metric_type='RATIO'
                    MATCH (claim:EvidenceClaim {id:item.claim_id})
                    MATCH (document:Document {doc_id:replace(item.source_filing, '.pdf', '')})
                    MERGE (observation:FinancialObservation {id:item.id})
                    SET observation += item,
                        observation.recorded_from=coalesce(observation.recorded_from, datetime()),
                        observation.recorded_to=null,
                        observation.is_current_record=true,
                        observation.model_version='financial_observation_v1'
                    MERGE (period:Year {year:item.fiscal_year})
                    MERGE (company)-[:HAS_FINANCIAL_OBSERVATION]->(observation)
                    MERGE (observation)-[:OBSERVES_METRIC]->(metric)
                    MERGE (observation)-[:SUPPORTED_BY_CLAIM]->(claim)
                    MERGE (observation)-[:DISCLOSED_IN]->(document)
                    MERGE (observation)-[:VALID_DURING]->(period)
                    SET claim.recorded_from=coalesce(claim.recorded_from, datetime()),
                        document.ingested_at=coalesce(document.ingested_at, datetime())
                    RETURN count(observation) AS applied
                    """,
                    observations=observations,
                ).single()
                applied = int(result["applied"] if result else 0)
    finally:
        driver.close()

    if args.apply:
        manager = SchemaManager()
        if manager.connect():
            try:
                manager.init_constraints()
                manager.init_indexes()
            finally:
                manager.close()

    report = {
        "schema": "strategic-graphrag-financial-observation/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "metric_claims": len(claims),
        "financial_observations": len(observations),
        "applied": applied,
        "guardrails": [
            "one observation per claim, metric, period, value, and unit",
            "every observation links to a verbatim EvidenceClaim",
            "fiscal-year granularity is preserved without fabricated calendar dates",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
