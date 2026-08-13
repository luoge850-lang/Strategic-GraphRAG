"""Read-only audit of strict, evidence-backed graph chains.

The audit does not infer new causal relationships. It only reports chains
already present as native relationships whose two edges both point to a
VERBATIM EvidenceClaim in the same filing. This is the minimum useful check
before exposing multi-hop reasoning to users.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from strategic_graphrag.ontology.relation_inference import validate_triple


EVIDENCE_RELATIONS = {
    "OBSERVED_IN", "REPORTS", "SUPPORTS", "BELONGS_TO", "SUPPORTED_BY",
    "ABOUT_SOURCE", "ABOUT_TARGET",
}


def run_audit(filing: str | None = None) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    edge_query = """
    MATCH (src)-[r]->(tgt)
    WHERE ($filing IS NULL OR coalesce(r.source_filing, r.filing, '') = $filing)
      AND NOT type(r) IN $evidence_relations
    OPTIONAL MATCH (claim:EvidenceClaim {id:r.evidence_id})
    RETURN labels(src)[0] AS source_category, src.id AS source,
           type(r) AS relation, labels(tgt)[0] AS target_category, tgt.id AS target,
           coalesce(r.source_filing, r.filing, '') AS filing,
           r.year AS year, r.source_page AS page, r.chunk_id AS chunk_id,
           r.temporal_scope AS temporal_scope, r.evidence_id AS evidence_id,
           claim.verification_status AS verification_status,
           claim.doc_id AS claim_doc_id
    """
    chain_query = """
    MATCH (a)-[r1]->(b)-[r2]->(c)
    WHERE ($filing IS NULL OR coalesce(r1.source_filing, r1.filing, '') = $filing)
      AND coalesce(r1.source_filing, r1.filing, '') = coalesce(r2.source_filing, r2.filing, '')
      AND NOT type(r1) IN $evidence_relations
      AND NOT type(r2) IN $evidence_relations
    MATCH (c1:EvidenceClaim {id:r1.evidence_id})
    MATCH (c2:EvidenceClaim {id:r2.evidence_id})
    WHERE c1.verification_status = 'VERBATIM'
      AND c2.verification_status = 'VERBATIM'
    RETURN labels(a)[0] AS source_category, a.id AS source,
           type(r1) AS relation_1, labels(b)[0] AS bridge_category, b.id AS bridge,
           type(r2) AS relation_2, labels(c)[0] AS target_category, c.id AS target,
           coalesce(r1.source_filing, r1.filing, '') AS filing,
           r1.year AS year_1, r1.source_page AS page_1,
           r2.year AS year_2, r2.source_page AS page_2,
           r1.temporal_scope AS temporal_scope_1,
           r2.temporal_scope AS temporal_scope_2,
           r1.chunk_id AS chunk_id_1, r2.chunk_id AS chunk_id_2,
           c1.id AS claim_1, c2.id AS claim_2
    LIMIT 5000
    """
    params = {"filing": filing, "evidence_relations": sorted(EVIDENCE_RELATIONS)}
    try:
        with driver.session() as session:
            edges = [record.data() for record in session.run(edge_query, **params)]
            chains = [record.data() for record in session.run(chain_query, **params)]
    finally:
        driver.close()

    strict_edges = [
        edge for edge in edges
        if edge.get("verification_status") == "VERBATIM"
    ]
    invalid_edges = []
    missing_provenance = []
    for edge in strict_edges:
        valid, reason = validate_triple(
            edge.get("source_category"), edge.get("target_category"),
            edge.get("relation"), edge.get("source"), edge.get("target"),
        )
        if not valid:
            invalid_edges.append({"edge": edge, "reason": reason})
        if not edge.get("chunk_id"):
            missing_provenance.append(edge)

    invalid_chains = []
    for chain in chains:
        first_valid, first_reason = validate_triple(
            chain.get("source_category"), chain.get("bridge_category"),
            chain.get("relation_1"), chain.get("source"), chain.get("bridge"),
        )
        second_valid, second_reason = validate_triple(
            chain.get("bridge_category"), chain.get("target_category"),
            chain.get("relation_2"), chain.get("bridge"), chain.get("target"),
        )
        if not first_valid or not second_valid:
            invalid_chains.append({
                "chain": chain,
                "reason": first_reason if not first_valid else second_reason,
            })

    return {
        "status": "PASS" if not invalid_edges and not invalid_chains else "FAIL",
        "scope": filing or "ALL_FILINGS",
        "contract": "two native business edges, same filing, both VERBATIM claims",
        "edge_audit": {
            "all_business_edge_instances": len(edges),
            "strict_verbatim_edges": len(strict_edges),
            "invalid_strict_edges": len(invalid_edges),
            "strict_edges_missing_chunk_id": len(missing_provenance),
            "relation_counts": dict(Counter(edge.get("relation") for edge in strict_edges)),
            "year_counts": dict(Counter(str(edge.get("year")) for edge in strict_edges)),
            "invalid_examples": invalid_edges[:20],
        },
        "chain_audit": {
            "strict_two_hop_chains": len(chains),
            "invalid_chains": len(invalid_chains),
            "same_year_chains": sum(
                chain.get("year_1") == chain.get("year_2") for chain in chains
            ),
            "cross_year_chains": sum(
                chain.get("year_1") != chain.get("year_2") for chain in chains
            ),
            "chains_missing_chunk_id": sum(
                not chain.get("chunk_id_1") or not chain.get("chunk_id_2")
                for chain in chains
            ),
            "invalid_examples": invalid_chains[:20],
            "examples": chains[:50],
        },
        "interpretation": [
            "A chain is reported, not promoted to a new causal claim.",
            "Cross-year chains require explicit temporal reasoning before production use.",
            "Missing chunk_id is expected for the pre-migration graph and must be zero after the next rebuild.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only strict chain audit")
    parser.add_argument("--filing", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_audit(args.filing)
    rendered = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
