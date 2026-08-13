"""Archive and optionally remove graph/vector data outside the active corpus."""

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
BUSINESS_RELATIONS = [
    "CAUSES", "COMPETES_WITH", "CONSTRAINS", "DECREASES", "EXPOSED_TO",
    "INCREASES", "MITIGATES", "OPERATES_IN", "PRODUCES", "REPORTS_METRIC",
    "TRIGGERS",
]


def serial(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [serial(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return serial(value.tolist())
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            legacy_edges = [serial(dict(row)) for row in session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE (r.confidence IS NOT NULL OR type(r) IN $business_relations) AND NOT (
                  coalesce(r.source_filing,r.filing,'') IN $filings
                  AND r.evidence_id IS NOT NULL
                  AND EXISTS { MATCH (c:EvidenceClaim {id:r.evidence_id})
                               WHERE c.verification_status='VERBATIM' })
                RETURN labels(a) AS source_labels, properties(a) AS source,
                       type(r) AS relation_type, properties(r) AS relation,
                       labels(b) AS target_labels, properties(b) AS target
                ORDER BY coalesce(r.source_filing,r.filing,''), type(r), a.id, b.id
                """, filings=ACTIVE_FILINGS, business_relations=BUSINESS_RELATIONS
            )]
            legacy_nodes = [serial(dict(row)) for row in session.run(
                """
                MATCH (n) WHERE (n:Document OR n:Sentence OR n:EvidenceClaim)
                AND coalesce(n.filename, n.doc_id + '.pdf', '') <> ''
                AND NOT coalesce(n.filename, n.doc_id + '.pdf', '') IN $filings
                WITH n, labels(n) AS node_labels
                RETURN node_labels AS labels, properties(n) AS properties
                ORDER BY node_labels[0], coalesce(n.id,n.doc_id,'')
                """, filings=ACTIVE_FILINGS
            )]
            before = {"legacy_edges": len(legacy_edges), "legacy_nodes": len(legacy_nodes)}
            if args.apply:
                session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE (r.confidence IS NOT NULL OR type(r) IN $business_relations) AND NOT (
                      coalesce(r.source_filing,r.filing,'') IN $filings
                      AND r.evidence_id IS NOT NULL
                      AND EXISTS { MATCH (c:EvidenceClaim {id:r.evidence_id})
                                   WHERE c.verification_status='VERBATIM' })
                    DELETE r
                    """, filings=ACTIVE_FILINGS, business_relations=BUSINESS_RELATIONS
                ).consume()
                session.run(
                    """
                    MATCH (n) WHERE (n:Document OR n:Sentence OR n:EvidenceClaim)
                    AND coalesce(n.filename, n.doc_id + '.pdf', '') <> ''
                    AND NOT coalesce(n.filename, n.doc_id + '.pdf', '') IN $filings
                    DETACH DELETE n
                    """, filings=ACTIVE_FILINGS
                ).consume()
    finally:
        driver.close()

    chroma_path = ROOT / "data" / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_path))
    configured = os.getenv("GRAPH_VECTOR_COLLECTION", "nvidia_sec_filings_active")
    legacy_collections = []
    for listed_collection in client.list_collections():
        collection_name = (
            listed_collection
            if isinstance(listed_collection, str)
            else listed_collection.name
        )
        if collection_name == configured:
            continue
        current = client.get_collection(collection_name)
        data = current.get(include=["documents", "metadatas", "embeddings"])
        legacy_collections.append({
            "name": collection_name,
            "ids": data.get("ids", []),
            "documents": data.get("documents", []),
            "metadatas": data.get("metadatas", []),
            "embeddings": serial(data.get("embeddings", [])),
        })
        if args.apply:
            client.delete_collection(collection_name)

    report = {
        "schema": "strategic-graphrag-legacy-archive/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "active_filings": ACTIVE_FILINGS,
        "graph": {**before, "edges": legacy_edges, "nodes": legacy_nodes},
        "legacy_vector_collections": legacy_collections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "mode": report["mode"],
        "legacy_edges": len(legacy_edges),
        "legacy_nodes": len(legacy_nodes),
        "legacy_vector_collections": [
            {"name": item["name"], "records": len(item["ids"])} for item in legacy_collections
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
