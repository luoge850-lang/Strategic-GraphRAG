"""Create a reproducibility manifest for the active three-filing corpus."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
import chromadb


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_PDFS = [
    ROOT / "data" / "pdfs_other" / "2023-10-K.pdf",
    ROOT / "data" / "pdfs_other" / "2024-10-K.pdf",
    ROOT / "data" / "pdfs" / "2025-10-K.pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def source_tree_sha256() -> str:
    digest = hashlib.sha256()
    tracked = git_value("ls-files").splitlines()
    for relative in sorted(tracked):
        if not (
            relative.startswith(("strategic_graphrag/", "frontend/src/", "scripts/", "tests/"))
            or relative.startswith("requirements")
            or relative in {".env.example", "README.md"}
        ):
            continue
        path = ROOT / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("neo4j", "chromadb", "onnxruntime", "fastapi", "sentence-transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "corpus_manifest.json"
    load_dotenv(ROOT / ".env", override=True)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            rows = [dict(row) for row in session.run(
                """
                MATCH (c:EvidenceClaim)
                WHERE c.doc_id + '.pdf' IN $filings AND c.verification_status='VERBATIM'
                RETURN c.doc_id + '.pdf' AS filing, count(c) AS claims,
                       count(DISTINCT c.page) AS evidence_pages,
                       count(DISTINCT c.chunk_id) AS evidence_chunks
                ORDER BY filing
                """,
                filings=[path.name for path in ACTIVE_PDFS],
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
    collection_name = os.getenv("GRAPH_VECTOR_COLLECTION", "nvidia_sec_filings_active")
    client = chromadb.PersistentClient(path=str(ROOT / "data/chroma_db"))
    try:
        vector_count = client.get_collection(collection_name).count()
    except Exception:
        vector_count = 0
    manifest = {
        "schema": "strategic-graphrag-corpus-manifest/v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_id": "nvidia-10k-2023-2025-v3",
        "active_filings": [
            {"filename": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in ACTIVE_PDFS
        ],
        "graph_inventory": rows,
        "temporal_inventory": temporal,
        "claim_id_version": "v2",
        "ontology_sha256": sha256(ROOT / "strategic_graphrag" / "ontology" / "financial_ontology.json"),
        "provider": os.getenv("LLM_PROVIDER"),
        "extraction_model": os.getenv("LLM_EXTRACTION_MODEL") or os.getenv("LLM_MODEL"),
        "query_model": os.getenv("LLM_QUERY_MODEL") or os.getenv("LLM_MODEL"),
        "report_model": os.getenv("LLM_REPORT_MODEL") or os.getenv("LLM_MODEL"),
        "embedding": {
            "backend": os.getenv("GRAPH_EMBEDDING_BACKEND", "sentence_transformers"),
            "model": os.getenv("GRAPH_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "collection": collection_name,
            "chunk_count": vector_count,
        },
        "pipeline": {
            "prompt_version": os.getenv("GRAPHRAG_PROMPT_VERSION", "v2-evidence-claim-1"),
            "source_tree_sha256": source_tree_sha256(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "packages": package_versions(),
        },
        "git": {
            "branch": git_value("branch", "--show-current"),
            "release_source_commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "evaluation": {
            "extraction_baseline": "reports/2026-08-14_extraction_quality_baseline.json",
            "human_annotation_status": "UNLABELED_STRATIFIED_SAMPLE",
            "golden_qa_status": "NO_CURRENT_HUMAN_GOLDEN_QA",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
