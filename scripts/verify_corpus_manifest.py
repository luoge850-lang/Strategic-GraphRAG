"""Verify that a published corpus manifest still matches code, files and stores."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from neo4j import GraphDatabase

import create_corpus_manifest as manifest_lib


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports/2026-08-14_corpus_manifest.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    checks = {}
    checks["schema_v2"] = saved.get("schema") == "strategic-graphrag-corpus-manifest/v2"
    checks["source_tree"] = saved.get("pipeline", {}).get("source_tree_sha256") == manifest_lib.source_tree_sha256()
    expected = {item["filename"]: item["sha256"] for item in saved.get("active_filings", [])}
    checks["pdf_hashes"] = all(manifest_lib.sha256(path) == expected.get(path.name) for path in manifest_lib.ACTIVE_PDFS)
    checks["ontology_hash"] = saved.get("ontology_sha256") == manifest_lib.sha256(ROOT / "strategic_graphrag/ontology/financial_ontology.json")
    load_dotenv(ROOT / ".env", override=True)
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            graph_inventory = [dict(row) for row in session.run(
                """
                MATCH (c:EvidenceClaim)
                WHERE c.doc_id + '.pdf' IN $filings AND c.verification_status='VERBATIM'
                RETURN c.doc_id + '.pdf' AS filing, count(c) AS claims,
                       count(DISTINCT c.page) AS evidence_pages,
                       count(DISTINCT c.chunk_id) AS evidence_chunks ORDER BY filing
                """,
                filings=[path.name for path in manifest_lib.ACTIVE_PDFS],
            )]
            temporal = dict(session.run(
                """
                OPTIONAL MATCH ()-[next:NEXT_DISCLOSURE]->()
                WITH count(next) AS disclosure_links
                OPTIONAL MATCH (change:TemporalChange {model_version:'observed_change_v1'})
                RETURN disclosure_links, count(change) AS temporal_changes,
                       sum(CASE WHEN change.quantitative=true THEN 1 ELSE 0 END) AS quantitative_changes
                """
            ).single() or {})
    finally:
        driver.close()
    checks["graph_inventory"] = graph_inventory == saved.get("graph_inventory")
    checks["temporal_inventory"] = temporal == saved.get("temporal_inventory")
    embedding = saved.get("embedding", {})
    client = chromadb.PersistentClient(path=str(ROOT / "data/chroma_db"))
    try:
        vector_count = client.get_collection(embedding.get("collection")).count()
    except Exception:
        vector_count = -1
    checks["vector_inventory"] = vector_count == embedding.get("chunk_count")
    source_commit = saved.get("git", {}).get("release_source_commit", "")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT, check=False)
    checks["release_commit_is_ancestor"] = ancestor.returncode == 0
    result = {"manifest": str(path), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
