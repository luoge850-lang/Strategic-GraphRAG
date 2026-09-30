"""Fail-closed deployment checks; read data, never migrate or bless legacy stores."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configuration_errors(env: dict, production: bool) -> list[str]:
    errors = []
    if not re.fullmatch(r"build_[0-9a-f]{16}", env.get("GRAPHRAG_BUILD_ID", "")):
        errors.append("canonical GRAPHRAG_BUILD_ID required")
    if env.get("GRAPH_EMBEDDING_BACKEND") != "chroma_onnx":
        errors.append("candidate requires chroma_onnx embedding backend")
    if not env.get("GRAPH_VECTOR_COLLECTION"):
        errors.append("explicit GRAPH_VECTOR_COLLECTION required")
    if production:
        if env.get("API_AUTH_ENABLED", "").lower() not in {"1", "true", "yes", "on"}:
            errors.append("API authentication must be enabled")
        key = env.get("API_KEY", "")
        if len(key) < 32 or any(word in key.lower() for word in ("replace", "example", "your_", "changeme")):
            errors.append("API_KEY must be a non-placeholder secret of at least 32 characters")
        origins = [value.strip() for value in env.get("CORS_ORIGINS", "").split(",") if value.strip()]
        if not origins or any(urlsplit(value).scheme != "https" or not urlsplit(value).netloc
                              or "*" in value or urlsplit(value).path not in ("", "/") for value in origins):
            errors.append("production CORS requires explicit HTTPS origins")
        if env.get("QUERY_CACHE_TTL_SECONDS", "0") != "0":
            errors.append("disable query cache until release isolation is accepted")
    return errors


def metadata_errors(rows: list[dict], build_id: str, pdfs: dict, expected_count: int) -> dict:
    def wrong_hash(row):
        expected = pdfs.get(row.get("source_filing"), {}).get("sha256")
        hashes = [row[key] for key in ("document_sha256", "pdf_sha256") if key in row]
        return not expected or not hashes or any(value != expected for value in hashes)
    return {
        "count": len(rows), "expected_count": expected_count,
        "foreign_or_unbound_build": sum(row.get("build_id") != build_id for row in rows),
        "source_hash_mismatch": sum(wrong_hash(row) for row in rows),
        "missing_page": sum(not isinstance(row.get("page"), int) or row.get("page", 0) < 1 for row in rows),
    }


def run(candidate: Path, data_root: Path, production: bool) -> dict:
    from strategic_graphrag.build_identity import sha256_file, source_fingerprint
    from scripts.run_isolated_staging import verify_package
    started = time.perf_counter()
    checks = []

    def check(name: str, passed: bool, detail: dict):
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    env = dict(os.environ)
    build_id = env.get("GRAPHRAG_BUILD_ID", "")
    errors = configuration_errors(env, production)
    check("configuration", not errors, {"errors": errors, "production": production})
    verified = verify_package(candidate, build_id or None)
    check("immutable_candidate", verified.get("status") == "PASS", verified)
    if verified.get("status") != "PASS":
        return finish(checks, started, build_id)
    identity = json.loads((candidate / "build_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((candidate / "vector_index_manifest.json").read_text(encoding="utf-8"))
    graph = json.loads((candidate / "graph.json").read_text(encoding="utf-8"))
    source_hash = source_fingerprint(ROOT)
    check("source_identity", source_hash == identity.get("source_tree_sha256")
          and build_id == identity.get("build_id"),
          {"actual_source_sha256": source_hash, "candidate_source_sha256": identity.get("source_tree_sha256"),
           "selected_build_matches": build_id == identity.get("build_id")})
    pdf_results = []
    for filing, expected in identity["pdfs"].items():
        location = data_root / ("pdfs" if filing == "2025-10-K.pdf" else "pdfs_other") / filing
        actual = sha256_file(location) if location.is_file() else None
        pdf_results.append({"filing": filing, "hash_matches": actual == expected["sha256"]})
    check("source_pdfs", all(item["hash_matches"] for item in pdf_results), {"files": pdf_results})

    # The existing application also has global path/statistics queries. Require a
    # dedicated database, not just a same-database build_id filter on metric facts.
    try:
        from neo4j import GraphDatabase
        with GraphDatabase.driver(env.get("NEO4J_URI", "bolt://localhost:7687"),
                                  auth=(env.get("NEO4J_USERNAME", "neo4j"), env.get("NEO4J_PASSWORD", "")),
                                  connection_timeout=5, connection_acquisition_timeout=5,
                                  max_transaction_retry_time=0, notifications_min_severity="OFF") as driver:
            driver.verify_connectivity()
            with driver.session(database=env.get("NEO4J_DATABASE", "neo4j")) as session:
                claims = [dict(row["claim"]) for row in session.run("MATCH (c:EvidenceClaim) RETURN properties(c) AS claim")]
                observations = [dict(row["observation"]) for row in session.run(
                    "MATCH (o:FinancialObservation) RETURN properties(o) AS observation")]
                edges = [dict(row) for row in session.run(
                    "MATCH (s)-[r]->(t) WHERE r.evidence_id IS NOT NULL "
                    "RETURN s.id AS source, t.id AS target, type(r) AS relation, "
                    "r.evidence_id AS claim_id, r.build_id AS build_id, r.source_filing AS source_filing, "
                    "r.page AS page, r.year AS fact_year")]
        expected_claims = {edge["claim_id"] for edge in graph["edges"]}
        def claim_filing(row):
            name = row.get("source_filing") or row.get("doc_id") or ""
            return name if name.endswith(".pdf") else name + ".pdf"
        claim_checks = metadata_errors([
            {"build_id": row.get("build_id"), "source_filing": claim_filing(row),
             "document_sha256": row.get("document_sha256"), "page": row.get("page")}
            for row in claims], build_id, identity["pdfs"], len(expected_claims))
        claim_checks["claim_id_set_matches"] = {row.get("id") for row in claims} == expected_claims
        claim_checks["unsupported_claims"] = sum(row.get("verification_status") != "VERBATIM" for row in claims)
        check("neo4j_claims", len(claims) == len(expected_claims)
              and claim_checks["claim_id_set_matches"]
              and not any(claim_checks[key] for key in ("foreign_or_unbound_build", "source_hash_mismatch", "missing_page", "unsupported_claims")), claim_checks)
        required = ("company_id", "metric_id", "claim_id", "source_filing", "fiscal_period", "unit", "statement_type", "table_name")
        incomplete = sum(any(not row.get(field) for field in required)
                         or not isinstance(row.get("value"), (float, int))
                         or not math.isfinite(row.get("value", float("nan")))
                         or not isinstance(row.get("fiscal_year"), int) for row in observations)
        unbound = sum(row.get("build_id") != build_id for row in observations)
        check("neo4j_observations", bool(observations) and not incomplete and not unbound,
              {"count": len(observations), "incomplete": incomplete, "foreign_or_unbound_build": unbound})
        signature = lambda row: tuple(row.get(key) for key in
            ("source", "target", "relation", "claim_id", "source_filing", "page", "fact_year"))
        actual_edges, expected_edges = Counter(map(signature, edges)), Counter(map(signature, graph["edges"]))
        foreign_edges = sum(row.get("build_id") != build_id for row in edges)
        check("neo4j_graph_identity", actual_edges == expected_edges and not foreign_edges,
              {"count": len(edges), "expected_count": len(graph["edges"]), "foreign_or_unbound_build": foreign_edges,
               "missing_edges": sum((expected_edges - actual_edges).values()),
               "extra_edges": sum((actual_edges - expected_edges).values())})
        # Validate the actual production projection, not a local JSON adapter.
        from strategic_graphrag.engine.graph_rag_engine import CausalPathFinder
        with GraphDatabase.driver(env.get("NEO4J_URI", "bolt://localhost:7687"),
                auth=(env.get("NEO4J_USERNAME", "neo4j"), env.get("NEO4J_PASSWORD", "")),
                connection_timeout=5, connection_acquisition_timeout=5, max_transaction_retry_time=0) as driver:
            paths = CausalPathFinder(driver).find_metric_disclosures("REVENUE", year_start=2025, year_end=2025,
                source_filing="2025-10-K.pdf", build_id=build_id)
        valid_paths = [path for path in paths if path.metric_values and path.metric_units
                       and path.financial_observations and all(value == build_id for value in path.evidence_build_ids)]
        check("real_metric_projection", bool(paths) and len(valid_paths) == len(paths),
              {"paths": len(paths), "complete_paths": len(valid_paths)})
    except Exception as exc:
        check("neo4j_access", False, {"error_type": type(exc).__name__})

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(data_root / "chroma_db"))
        collection = client.get_collection(env.get("GRAPH_VECTOR_COLLECTION", ""), embedding_function=None)
        count = collection.count()
        rows = []
        for offset in range(0, count, 500):
            rows.extend(collection.get(offset=offset, limit=500, include=["metadatas"])["metadatas"])
        vector_checks = metadata_errors([row or {} for row in rows], build_id, identity["pdfs"], int(manifest["chunks_indexed"]))
        vector_checks["collection_matches_candidate"] = collection.name == manifest["collection"]
        check("chroma_identity", count == manifest["chunks_indexed"] and vector_checks["collection_matches_candidate"]
              and not any(vector_checks[key] for key in ("foreign_or_unbound_build", "source_hash_mismatch", "missing_page")), vector_checks)
    except Exception as exc:
        check("chroma_access", False, {"error_type": type(exc).__name__})
    after = verify_package(candidate, identity["build_id"])
    check("immutable_candidate_after", after.get("status") == "PASS", {"status": after.get("status")})
    return finish(checks, started, build_id)


def finish(checks: list, started: float, build_id: str) -> dict:
    return {"schema": "deployment-preflight/v1", "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "BLOCKED",
            "recorded_at": datetime.now(timezone.utc).isoformat(), "build_id": build_id or None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "checks": checks,
            "business_data_mutations": 0, "llm_calls": 0,
            "limitations": "Metadata reads on a writable Chroma runtime can perform storage housekeeping; immutable candidate is never opened by Chroma."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=os.getenv("CANDIDATE_BUILD_DIR"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; select a new path")
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    candidate = args.candidate_dir or Path(os.getenv("CANDIDATE_BUILD_DIR", ""))
    if not candidate.is_dir() or not (candidate / "build_identity.json").is_file():
        parser.error("an explicit immutable CANDIDATE_BUILD_DIR is required")
    report = run(candidate.resolve(), args.data_root.resolve(), args.production)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                      "failed": [row["name"] for row in report["checks"] if row["status"] != "PASS"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
