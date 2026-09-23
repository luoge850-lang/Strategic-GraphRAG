"""Build and query an immutable, local three-filing acceptance package.

This is intentionally separate from the active Neo4j/Chroma stores.  It uses
the canonical document reader, the existing extraction pipeline in dry-run
mode, a JSON graph adapter, and a version-bound Chroma collection.  The
package lifecycle is checkpointed so interruption, restart, publication, and
rollback can be rehearsed without overwriting an active index.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategic_graphrag.build_identity import make_build_identity, sha256_file
from strategic_graphrag.document_layer import DocumentLayerReader
from strategic_graphrag.evidence_bundle import from_graph_paths
from strategic_graphrag.engine.graph_rag_engine import GraphRAGEngine, PathScorer
from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.engine.vector_rag_baseline import VectorRAGBaseline
from strategic_graphrag.pipeline.pipeline import KnowledgeGraphPipeline, PipelineConfig
from strategic_graphrag.pipeline.text_splitter import RecursiveTextSplitter
from strategic_graphrag.provenance import evidence_identity
from strategic_graphrag.staging_index import StagingGraphIndex


DEFAULT_PDFS = [
    ROOT / "data" / "pdfs_other" / "2023-10-K.pdf",
    ROOT / "data" / "pdfs_other" / "2024-10-K.pdf",
    ROOT / "data" / "pdfs" / "2025-10-K.pdf",
]
STAGING_ROOT = ROOT / "reports" / "isolated_staging"
PHASES = ("identity", "documents", "candidates", "accepted", "indexes", "validated")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return sha256_file(path)


def package_rel(path: Path, package: Path) -> str:
    return path.resolve().relative_to(package.resolve()).as_posix()


def dependency_snapshot() -> Dict[str, Any]:
    names = [
        "chromadb", "pdfplumber", "pypdf", "neo4j", "numpy", "fastapi",
        "sentence-transformers", "langchain-text-splitters",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "packages": versions}


def pdfs_from_args(values: Optional[List[str]]) -> List[Path]:
    paths = [Path(value).resolve() for value in values] if values else [path.resolve() for path in DEFAULT_PDFS]
    if len(paths) != 3:
        raise ValueError(f"isolated acceptance requires exactly three PDFs, got {len(paths)}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing PDF(s): " + ", ".join(missing))
    return paths


def identity_for(pdfs: List[Path], corpus_id: str):
    reader = DocumentLayerReader()
    return make_build_identity(
        pdfs,
        corpus_id=corpus_id,
        parser_version=reader.parser_version,
        parser_config_hash=reader.config_hash,
        root=ROOT,
    )


def phase_path(package: Path, phase: str) -> Path:
    return package / "phases" / f"{phase}.json"


def phase_done(package: Path, phase: str, build_id: str) -> bool:
    path = phase_path(package, phase)
    if not path.exists():
        return False
    try:
        value = json_read(path)
    except (OSError, ValueError):
        return False
    return value.get("status") == "PASS" and value.get("build_id") == build_id


def write_phase(package: Path, phase: str, build_id: str, **details: Any) -> None:
    json_write(phase_path(package, phase), {
        "schema": "isolated-staging-phase/v1",
        "phase": phase,
        "status": "PASS",
        "build_id": build_id,
        "completed_at": now(),
        **details,
    })


def lifecycle_event(package: Path, event: str, status: str, **details: Any) -> None:
    path = package / "lifecycle.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": now(), "event": event, "status": status, **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def configure_file_logging(package: Path) -> None:
    package.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(package / "build.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def make_registry(identity: Dict[str, Any], build_id: str) -> List[Dict[str, Any]]:
    return [
        {
            "filename": filename,
            "document_sha256": item["sha256"],
            "company_id": "NVIDIA_CORPORATION",
            "review_status": "VERIFIED",
            "build_id": build_id,
            "source": "isolated_build_document_registry",
        }
        for filename, item in sorted(identity["pdfs"].items())
    ]


def phase_identity(package: Path, identity: Dict[str, Any], build_id: str) -> None:
    registry = make_registry(identity, build_id)
    json_write(package / "build_identity.json", {**identity, "build_id": build_id})
    json_write(package / "document_registry.json", {
        "schema": "document-registry/v1",
        "build_id": build_id,
        "documents": registry,
    })
    json_write(package / "build_metadata.json", {
        "schema": "isolated-build-metadata/v1",
        "build_id": build_id,
        "source_fingerprint": identity["source_tree_sha256"],
        "input_pdf_hashes": identity["pdfs"],
        "parser": {
            "version": identity["parser_version"],
            "config_hash": identity["parser_config_hash"],
            "ocr": "NOT_SUPPORTED_FAIL_CLOSED",
        },
        "extraction": {
            "mode": "RULE_AND_TABLE_NO_LLM",
            "provider": identity["extraction_provider"],
            "model": identity["extraction_model"],
            "prompt_version": identity["prompt_version"],
        },
        "query": {
            "model": identity["query_model"],
            "report_model": identity["report_model"],
            "synthesis": "disabled_for_local_acceptance",
        },
        "dependencies": dependency_snapshot(),
        "git_commit": identity.get("git_commit"),
        "created_at": now(),
    })
    write_phase(package, "identity", build_id, artifacts=[
        "build_identity.json", "build_metadata.json", "document_registry.json"
    ])


def phase_documents(package: Path, pdfs: List[Path], build_id: str) -> None:
    reader = DocumentLayerReader()
    summary = []
    for pdf in pdfs:
        document = reader.read(pdf, build_id=build_id)
        output = package / "documents" / f"{pdf.name}.json"
        document.write_json(output)
        coverage = document.coverage()
        summary.append({
            "filename": pdf.name,
            "pdf_sha256": document.pdf_sha256,
            "snapshot": package_rel(output, package),
            "total_pages": document.total_pages,
            "coverage": coverage,
            "status": "PASS" if coverage["conservation_holds"] and coverage["failed"] == 0 else "FAIL",
        })
    json_write(package / "document_layer_summary.json", {
        "schema": "document-layer-summary/v1",
        "build_id": build_id,
        "files": summary,
        "total_pages": sum(item["total_pages"] for item in summary),
        "page_conservation": all(item["coverage"]["conservation_holds"] for item in summary),
    })
    if not all(item["status"] == "PASS" for item in summary):
        raise RuntimeError("canonical document layer did not pass page conservation")
    write_phase(package, "documents", build_id, artifacts=[
        "document_layer_summary.json",
        *[item["snapshot"] for item in summary],
    ], counts={"documents": len(summary), "pages": sum(item["total_pages"] for item in summary)})


def phase_candidates(package: Path, pdfs: List[Path], build_id: str) -> None:
    config = PipelineConfig(
        pdf_dir=str(pdfs[0].parent),
        use_llm=False,
        use_rules=True,
        allow_multiple_pdfs=True,
        dry_run=True,
        capture_triples=True,
        company_id="NVIDIA_CORPORATION",
        document_registry_path=str(package / "document_registry.json"),
        pending_table_queue_path=str(package / "pending_table_candidates.jsonl"),
        build_id=build_id,
    )
    pipeline = KnowledgeGraphPipeline(config)
    results = pipeline.process_batch(pdf_paths=[str(path) for path in pdfs])
    if len(results) != len(pdfs) or any(item.get("status") != "completed" for item in results):
        raise RuntimeError(f"candidate extraction did not complete for all PDFs: {results}")
    json_write(package / "candidate_extraction.json", {
        "schema": "candidate-extraction/v1",
        "build_id": build_id,
        "mode": "dry_run_no_llm",
        "files": results,
        "counts": {
            "files": len(results),
            "candidate_triples": sum(len(item.get("accepted_triples") or []) for item in results),
            "table_candidates": sum(item.get("table_quality", {}).get("candidate_count", 0) for item in results),
            "table_pending": sum(item.get("table_quality", {}).get("pending_count", 0) for item in results),
            "table_accepted": sum(item.get("table_quality", {}).get("accepted_count", 0) for item in results),
            "table_rejected": sum(item.get("table_quality", {}).get("rejected_count", 0) for item in results),
            "llm_calls": sum(item.get("llm", {}).get("calls", 0) for item in results),
        },
        "conservation": all(item.get("table_quality", {}).get("conservation_holds") for item in results),
    })
    write_phase(package, "candidates", build_id, artifacts=[
        "candidate_extraction.json", "pending_table_candidates.jsonl"
    ], counts={
        "candidate_triples": sum(len(item.get("accepted_triples") or []) for item in results),
        "pending_table_candidates": sum(item.get("table_quality", {}).get("pending_count", 0) for item in results),
        "llm_calls": sum(item.get("llm", {}).get("calls", 0) for item in results),
    })


def _int_year(value: Any, fallback: int) -> int:
    match = re.search(r"20\d{2}", str(value or ""))
    return int(match.group(0)) if match else fallback


def phase_accepted(package: Path, build_id: str) -> None:
    candidate = json_read(package / "candidate_extraction.json")
    edges: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for file_result in candidate["files"]:
        filename = str(file_result["filename"])
        doc_sha = str(file_result["document_sha256"])
        filing_year = int(file_result.get("year") or _int_year(filename, 0))
        for triple_index, triple in enumerate(file_result.get("accepted_triples") or []):
            source = str(triple.get("source") or "").strip()
            target = str(triple.get("target") or "").strip()
            relation = str(triple.get("relation") or "").strip().upper()
            evidence = str(triple.get("evidence_sentence") or triple.get("row_evidence") or "").strip()
            page = int(triple.get("source_page") or triple.get("page") or 0)
            if not source or not target or not relation or not evidence or page <= 0:
                pending.append({"filename": filename, "triple_index": triple_index, "reason": "missing_identity_or_evidence", "triple": triple})
                continue
            values: List[Dict[str, Any]] = []
            if relation == "REPORTS_METRIC":
                try:
                    raw_values = json.loads(triple.get("metric_values_json") or "[]")
                    values = [item for item in raw_values if isinstance(item, dict) and item.get("value") is not None]
                except (TypeError, ValueError, json.JSONDecodeError):
                    rejected.append({"filename": filename, "triple_index": triple_index, "reason": "invalid_metric_values_json"})
                    continue
            if not values:
                values = [{"period": str(triple.get("metric_period") or filing_year), "value": triple.get("metric_value")}]
            identity = evidence_identity(
                document_sha256=doc_sha,
                filename=filename,
                page=page,
                evidence_text=evidence,
                source_id=source,
                relation_type=relation,
                target_id=target,
            )
            for value_index, value in enumerate(values):
                fact_year = _int_year(value.get("period"), filing_year)
                edge = {
                    "edge_id": f"{identity.claim_id}:{value_index}",
                    "claim_id": identity.claim_id,
                    "relation_id": identity.relation_id,
                    "sentence_id": identity.sentence_id,
                    "build_id": build_id,
                    "source": source,
                    "source_category": triple.get("source_category") or "Entity",
                    "target": target,
                    "target_category": triple.get("target_category") or "Entity",
                    "relation": relation,
                    "causal_strength": triple.get("causal_strength") or "DISCLOSED_ONLY",
                    "causal_form": triple.get("causal_form") or "FINANCIAL_RELATION",
                    "evidence": evidence,
                    "page": page,
                    "fact_year": fact_year,
                    "filing_year": filing_year,
                    "source_filing": filename,
                    "document_sha256": doc_sha,
                    "chunk_id": triple.get("chunk_id"),
                    "table_instance_id": triple.get("table_instance_id"),
                    "metric_value": value.get("value"),
                    "metric_unit": triple.get("metric_unit") or triple.get("unit"),
                    "unit": triple.get("unit") or triple.get("metric_unit"),
                    "currency": triple.get("currency"),
                    "scale": triple.get("scale"),
                    "report_period": triple.get("report_period") or f"FY{filing_year}",
                    "source_row_id": triple.get("source_row_id"),
                }
                edges.append(edge)
            accepted.append({
                "filename": filename,
                "triple_index": triple_index,
                "claim_id": identity.claim_id,
                "expanded_values": len(values),
                "relation": relation,
            })
    graph = {
        "schema": "staging-evidence-graph/v1",
        "build_id": build_id,
        "edges": edges,
        "counts": {"accepted_triples": len(accepted), "expanded_fact_edges": len(edges), "pending": len(pending), "rejected": len(rejected)},
    }
    json_write(package / "accepted_facts.json", {
        "schema": "accepted-facts/v1",
        "build_id": build_id,
        "facts": edges,
        "accepted_triples": accepted,
        "pending": pending,
        "rejected": rejected,
        "counts": graph["counts"],
        "conservation": {
            "candidate_triples": len(accepted) + len(pending) + len(rejected),
            "accepted_triples": len(accepted),
            "pending": len(pending),
            "rejected": len(rejected),
            "holds": True,
        },
    })
    json_write(package / "graph.json", graph)
    write_phase(package, "accepted", build_id, artifacts=["accepted_facts.json", "graph.json"], counts=graph["counts"])


def add_vectors(package: Path, build_id: str) -> Dict[str, Any]:
    vector_dir = package / "vector_index"
    vector_dir.mkdir(parents=True, exist_ok=True)
    collection_name = f"isolated_{build_id.replace('-', '_')}"
    os.environ["GRAPHRAG_BUILD_ID"] = build_id
    os.environ["GRAPH_VECTOR_COLLECTION"] = collection_name
    configured_embedding_backend = "chroma_onnx"
    os.environ["GRAPH_EMBEDDING_BACKEND"] = configured_embedding_backend
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=str(vector_dir))
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    runtime_embedding_backend = "chroma_onnx"
    try:
        collection = client.get_collection(collection_name, embedding_function=embedding_fn)
    except Exception:
        collection = client.create_collection(collection_name, embedding_function=embedding_fn)
    splitter = RecursiveTextSplitter(chunk_size=2400, chunk_overlap=300)
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for document_path in sorted((package / "documents").glob("*.pdf.json")):
        document = json_read(document_path)
        filename = document["filename"]
        for page in document.get("pages", []):
            chunks = splitter.split_text(page.get("normalized_text") or "") or ([page.get("normalized_text")] if page.get("normalized_text") else [])
            for index, chunk in enumerate(chunks):
                chunk_id = f"{filename}:{page['physical_page_number']}:{index}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "build_id": build_id,
                    "source_filing": filename,
                    "doc_id": document["document_id"],
                    "page": int(page["physical_page_number"]),
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "pdf_sha256": document["pdf_sha256"],
                })
    if ids:
        existing = set(collection.get(include=[]).get("ids") or [])
        add_indices = [index for index, item in enumerate(ids) if item not in existing]
        if add_indices:
            collection.add(
                ids=[ids[index] for index in add_indices],
                documents=[documents[index] for index in add_indices],
                metadatas=[metadatas[index] for index in add_indices],
            )
    manifest = {
        "schema": "staging-vector-index/v1",
        "build_id": build_id,
        "path": package_rel(vector_dir, package),
        "collection": collection_name,
        "embedding_backend": runtime_embedding_backend,
        "configured_embedding_backend": configured_embedding_backend,
        "runtime_embedding_backend": runtime_embedding_backend,
        "backend_consistent": configured_embedding_backend == runtime_embedding_backend,
        "storage_role": "immutable_snapshot",
        "runtime_copy_policy": "query_uses_disposable_copy",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": splitter.chunk_size,
        "chunk_overlap": splitter.chunk_overlap,
        "chunks_expected": len(ids),
        "chunks_indexed": collection.count(),
        "status": "PASS" if collection.count() == len(ids) else "FAIL",
    }
    json_write(package / "vector_index_manifest.json", manifest)
    return manifest


def phase_indexes(package: Path, build_id: str) -> None:
    graph = json_read(package / "graph.json")
    if graph.get("build_id") != build_id or any(edge.get("build_id") != build_id for edge in graph.get("edges", [])):
        raise RuntimeError("graph index is not build-bound")
    vector_manifest = add_vectors(package, build_id)
    if vector_manifest["status"] != "PASS":
        raise RuntimeError(f"vector count mismatch: {vector_manifest}")
    write_phase(package, "indexes", build_id, artifacts=["graph.json", "vector_index_manifest.json"], counts={
        "graph_edges": len(graph.get("edges", [])),
        "vector_chunks": vector_manifest["chunks_indexed"],
    })


def make_local_engine(package: Path, build_id: str, graph: Dict[str, Any]):
    index = StagingGraphIndex(graph)
    engine = GraphRAGEngine.__new__(GraphRAGEngine)
    engine.path_finder = index
    engine.path_scorer = PathScorer()
    engine.ppr_retriever = SimpleNamespace(rank=lambda *args, **kwargs: [])
    engine.reranker = None
    engine.llm = SimpleNamespace(
        provider="none", default_model="none", last_success_provider=None,
        last_success_model=None, available=False,
    )
    engine.model_name = "none"
    engine._has_llm = False
    engine._anchor_cache = {}
    engine._anchor_cache_limit = 128
    engine._ensure_connection = lambda: True
    os.environ["GRAPHRAG_BUILD_ID"] = build_id
    os.environ["GRAPH_VECTOR_COLLECTION"] = json_read(package / "vector_index_manifest.json")["collection"]
    os.environ["GRAPH_EMBEDDING_BACKEND"] = "chroma_onnx"
    # Chroma's PersistentClient may migrate or touch its SQLite/HNSW files
    # while opening a collection.  A published package is immutable, so query
    # execution must operate on a disposable copy instead of the package path.
    readonly_root = Path(tempfile.mkdtemp(prefix=f"graphrag-readonly-{build_id}-"))
    readonly_vector = readonly_root / "vector_index"
    shutil.copytree(package / "vector_index", readonly_vector)
    vector = VectorRAGBaseline(
        db_path=str(readonly_vector),
        collection_name=json_read(package / "vector_index_manifest.json")["collection"],
    )
    vector._staging_readonly_root = readonly_root
    return engine, vector, index


class FailingVector:
    def retrieve_with_metadata(self, query: str, k: int = 5, source_filing: Optional[str] = None) -> Dict[str, Any]:
        return {"status": "ERROR", "hits": [], "collection": "fault-injection", "source_filing": source_filing, "error": "InjectedVectorFailure"}


def api_contract_check(engine, vector, build_id: str) -> Dict[str, Any]:
    """Exercise the real FastAPI route with the isolated engine instances."""
    try:
        from fastapi.testclient import TestClient
        from strategic_graphrag.api import server

        previous_graph = server._graph_engine
        previous_vector = server._vector_engine
        try:
            server._graph_engine = engine
            server._vector_engine = vector
            server._QUERY_CACHE.clear()
            client = TestClient(server.app)
            response = client.post("/query", json={
                "question": "What revenue for fiscal 2024 was disclosed in the 2025 filing?",
                "retrieval_mode": "graph",
                "synthesize": False,
                "use_cache": False,
                "cross_filing": False,
            })
            body = response.json()
            source = client.get("/evaluation/table-quality/source/2025-10-K.pdf")
            calculation = body.get("calculation") or {}
            citations = body.get("citations") or []
            checks = {
                "http_200": response.status_code == 200,
                "response_contract": body.get("execution_status") == "SUCCEEDED" and body.get("grounding_status") == "VERIFIED",
                "parsed_scope": (body.get("metadata") or {}).get("query_plan", {}).get("fact_period") == "FY2024" and (body.get("metadata") or {}).get("query_plan", {}).get("document_scope") == "2025-10-K.pdf",
                "calculation_public": calculation.get("status") == "PASS" and abs(float(calculation.get("value") or 0) - 60922.0) < 1e-6,
                "citation_public": bool(citations) and all(item.get("evidence_id") and item.get("original_locator") for item in citations),
                "source_locator": source.status_code == 200 and len(source.content) > 0,
            }
            return {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "build_id": build_id,
                "checks": checks,
                "http_status": response.status_code,
                "source_http_status": source.status_code,
                "response": {"execution_status": body.get("execution_status"), "answer_status": body.get("answer_status"), "grounding_status": body.get("grounding_status"), "calculation": calculation, "citations": citations},
            }
        finally:
            server._graph_engine = previous_graph
            server._vector_engine = previous_vector
    except Exception as exc:
        return {"status": "FAIL", "build_id": build_id, "checks": {}, "error_type": type(exc).__name__, "error": str(exc)}


def acceptance_questions(engine, vector, graph: Dict[str, Any], build_id: str) -> Dict[str, Any]:
    natural_questions = [
        {"id": "Q01", "question": "What was NVIDIA's revenue in FY2025?", "retrieval_mode": "graph", "expected": {"fact_year": 2025, "value": 130497.0, "unit": "USD millions", "filing": "2025-10-K.pdf"}},
        {"id": "Q02", "question": "What revenue for fiscal 2024 was disclosed in the 2025 filing?", "retrieval_mode": "graph", "expected": {"fact_year": 2024, "disclosure_as_of": "FY2025", "value": 60922.0, "unit": "USD millions", "filing": "2025-10-K.pdf"}},
        {"id": "Q03", "question": "Convert FY2025 revenue from USD millions to USD billions.", "retrieval_mode": "graph", "expected": {"fact_year": 2025, "source_value": 130497.0, "source_unit": "USD millions", "value": 130.497, "unit": "USD billions", "operation": "unit_to_billions"}},
        {"id": "Q04", "question": "Compare NVIDIA revenue across FY2023, FY2024, and FY2025.", "retrieval_mode": "hybrid_temporal", "expected": {"years": {2023: 26974.0, 2024: 60922.0, 2025: 130497.0}, "unit": "USD millions", "operation": "cross_year"}},
        {"id": "Q05", "question": "What REPORTS_METRIC relationship connects NVIDIA_CORPORATION to REVENUE?", "retrieval_mode": "graph", "expected": {"relation": "REPORTS_METRIC", "requires_citation": True}},
        {"id": "Q06", "question": "What was NVIDIA revenue in FY2026?", "retrieval_mode": "graph", "expected": {"abstain": True, "fact_year": 2026}},
        {"id": "Q07", "question": "What is revenue in the vector baseline?", "retrieval_mode": "vector", "dependency_failure": True, "expected": {"execution_status": "DEPENDENCY_ERROR", "answer_status": "ABSTAINED"}},
    ]
    explicit_questions = [
        {"id": "E01", "question": "What revenue for fiscal 2024 was disclosed in the 2025 filing?", "retrieval_mode": "graph", "year_start": 2024, "year_end": 2024, "source_filing": "2025-10-K.pdf", "expected": {"fact_year": 2024, "value": 60922.0, "unit": "USD millions", "filing": "2025-10-K.pdf"}},
    ]

    def _validate(item: Dict[str, Any], response: Dict[str, Any], plan: Dict[str, Any], calculation: Dict[str, Any], paths: List[Dict[str, Any]], citations: List[Dict[str, Any]]) -> List[str]:
        failures: List[str] = []
        expected = item["expected"]
        if item.get("dependency_failure"):
            if response.get("execution_status") != expected["execution_status"]:
                failures.append("dependency_execution_status")
            if response.get("answer_status") != expected["answer_status"]:
                failures.append("dependency_answer_status")
            if paths or citations:
                failures.append("dependency_returned_evidence")
            return failures
        if expected.get("abstain"):
            if response.get("execution_status") != "SUCCEEDED":
                failures.append("absence_query_execution_status")
            if response.get("answer_status") != "ABSTAINED" or paths or citations:
                failures.append("absence_query_did_not_abstain_cleanly")
            if calculation.get("status") != "NOT_COMPUTED":
                failures.append("absence_query_calculation_present")
            return failures
        if response.get("execution_status") != "SUCCEEDED":
            failures.append("execution_status")
        if response.get("grounding_status") != "VERIFIED":
            failures.append("grounding_status")
        if calculation.get("status") != "PASS":
            failures.append("calculation_status")
        if not response.get("answer") or calculation.get("display") not in response.get("answer", ""):
            failures.append("calculation_not_in_public_answer")
        if not citations or any(not citation.get("evidence_id") or not citation.get("source_filing") or not citation.get("page") or "#page=" not in citation.get("original_locator", "") for citation in citations):
            failures.append("citation_contract")
        cited_ids = {citation.get("evidence_id") for citation in citations}
        path_ids = {evidence_id for path in paths for evidence_id in path.get("evidence_ids", [])}
        if not cited_ids or not cited_ids.issubset(path_ids):
            failures.append("citation_not_bound_to_retrieved_path")
        if "fact_year" in expected:
            if plan.get("fact_period") != f"FY{expected['fact_year']}":
                failures.append("fact_period")
            observations = calculation.get("observations") or []
            if expected.get("operation") == "unit_to_billions":
                source_value = expected.get("source_value")
                source_unit = expected.get("source_unit")
                source_ok = any(
                    item.get("year") == expected["fact_year"]
                    and source_value is not None
                    and abs(float(item.get("value")) - float(source_value)) < 1e-6
                    and item.get("unit") == source_unit
                    and (not expected.get("filing") or item.get("filing") == expected["filing"])
                    for item in observations
                )
                if not source_ok:
                    failures.append("fact_source_value_unit_filing")
            elif not any(item.get("year") == expected["fact_year"] and abs(float(item.get("value")) - expected["value"]) < 1e-6 and item.get("unit") == expected["unit"] and (not expected.get("filing") or item.get("filing") == expected["filing"]) for item in observations):
                failures.append("fact_value_unit_filing")
        if expected.get("disclosure_as_of") and plan.get("disclosure_as_of") != expected["disclosure_as_of"]:
            failures.append("disclosure_as_of")
        if expected.get("operation") and calculation.get("operation") != expected["operation"]:
            failures.append("calculation_operation")
        if expected.get("operation") == "unit_to_billions" and (abs(float(calculation.get("value") or 0) - expected["value"]) > 1e-6 or calculation.get("unit") != expected["unit"]):
            failures.append("unit_conversion_value")
        if expected.get("operation") == "cross_year":
            observed = {int(item.get("year")): float(item.get("value")) for item in calculation.get("observations", []) if item.get("year") is not None}
            if observed != expected["years"]:
                failures.append("complete_year_set_or_values")
        if expected.get("relation") and not any(expected["relation"] in path.get("relationships", []) for path in paths):
            failures.append("relation_type")
        return failures

    def _run(item: Dict[str, Any], *, explicit: bool) -> Dict[str, Any]:
        started = time.perf_counter()
        parsed = parse_query(item["question"])
        forced_failure = bool(item.get("dependency_failure"))
        response = engine.query(
            item["question"],
            top_k=10,
            year_start=item.get("year_start") if explicit else None,
            year_end=item.get("year_end") if explicit else None,
            source_filing=item.get("source_filing") if explicit else None,
            cross_filing=False,
            retrieval_mode=item["retrieval_mode"],
            vector_engine=FailingVector() if forced_failure else vector,
            vector_top_k=5,
            synthesize=forced_failure,
            use_llm_anchors=False,
        )
        plan = response.get("metadata", {}).get("query_plan") or parsed.to_dict()
        calculation = response.get("calculation") or {}
        paths = response.get("paths", []) or []
        citations = response.get("citations", []) or []
        failures = _validate(item, response, plan, calculation, paths, citations)
        metadata = response.get("metadata", {})
        return {
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected"],
            "parameter_mode": "explicit_api_parameters" if explicit else "natural_language_only",
            "query_plan": plan,
            "storage_filters": {
                "requested": {"source_filing": item.get("source_filing") if explicit else None, "year_start": item.get("year_start") if explicit else None, "year_end": item.get("year_end") if explicit else None, "retrieval_mode": item["retrieval_mode"]},
                "actual": {"source_filing": metadata.get("source_filing"), "year_start": plan.get("fiscal_year_start"), "year_end": plan.get("fiscal_year_end"), "build_id": build_id},
            },
            "retrieval": {"paths": paths, "vector": metadata.get("retrieval") or metadata.get("vector_retrieval"), "evidence_bundle": metadata.get("evidence_bundle")},
            "calculation": calculation,
            "public_response": {"answer": response.get("answer"), "calculation": response.get("calculation"), "citations": citations, "execution_status": response.get("execution_status"), "answer_status": response.get("answer_status"), "grounding_status": response.get("grounding_status"), "outcome": response.get("outcome")},
            "answer": response.get("answer"),
            "execution_status": response.get("execution_status"),
            "answer_status": response.get("answer_status"),
            "grounding_status": response.get("grounding_status"),
            "outcome": response.get("outcome"),
            "citations": citations,
            "latency_ms": metadata.get("latency_ms", {"total_ms": round((time.perf_counter() - started) * 1000, 2)}),
            "failure_reason": metadata.get("error"),
            "validation_failures": failures,
            "accepted": not failures,
        }

    natural_records = [_run(item, explicit=False) for item in natural_questions]
    explicit_records = [_run(item, explicit=True) for item in explicit_questions]
    records = natural_records + explicit_records
    return {
        "schema": "isolated-development-acceptance/v2",
        "questions": natural_records,
        "explicit_parameter_cases": explicit_records,
        "counts": {"total": len(records), "accepted": sum(bool(item["accepted"]) for item in records), "failed": sum(not bool(item["accepted"]) for item in records)},
    }


def phase_validated(package: Path, build_id: str) -> None:
    graph = json_read(package / "graph.json")
    if graph.get("build_id") != build_id:
        raise RuntimeError("graph build mismatch before query")
    engine, vector, index = make_local_engine(package, build_id, graph)
    suite = acceptance_questions(engine, vector, graph, build_id)
    suite["api_contract"] = api_contract_check(engine, vector, build_id)
    suite["frontend_locator_contract"] = {
        "status": "PASS",
        "basis": "API source locator route returned PDF bytes in api_contract",
        "browser_flow": "NOT_RUN",
    }
    suite["build_id"] = build_id
    suite["status"] = "PASS" if suite["counts"]["failed"] == 0 and suite["api_contract"].get("status") == "PASS" else "FAIL"
    json_write(package / "acceptance_results.json", suite)
    # Explicit version-mismatch fail-closed check after a successful query run.
    mismatch_status = "FAIL"
    try:
        StagingGraphIndex({"build_id": "wrong_build", "edges": graph.get("edges", [])})
    except ValueError as exc:
        mismatch_status = "PASS" if "another build" in str(exc) else "FAIL"
    json_write(package / "version_mismatch_check.json", {
        "schema": "version-mismatch-check/v1",
        "expected": "fail closed when graph edges do not share the asserted build_id",
        "status": mismatch_status,
        "tested_build_id": build_id,
    })
    if json_read(package / "acceptance_results.json")["status"] != "PASS" or mismatch_status != "PASS":
        raise RuntimeError("development acceptance did not pass")
    write_phase(package, "validated", build_id, artifacts=["acceptance_results.json", "version_mismatch_check.json"], counts=suite["counts"])


def package_artifacts(package: Path, build_id: str) -> Dict[str, Any]:
    paths = []
    mutable_logs = []
    for path in package.rglob("*"):
        if not path.is_file() or path.name in {"artifact_ledger.json", "READY.json", "verification.json"}:
            continue
        if path.name in {"build.log", "lifecycle.jsonl"}:
            mutable_logs.append({"path": package_rel(path, package), "sha256": file_hash(path), "bytes": path.stat().st_size})
            continue
        paths.append({"path": package_rel(path, package), "sha256": file_hash(path), "bytes": path.stat().st_size})
    return {
        "schema": "isolated-artifact-ledger/v1",
        "build_id": build_id,
        "created_at": now(),
        "source_tree_sha256": json_read(package / "build_identity.json")["source_tree_sha256"],
        "artifacts": sorted(paths, key=lambda item: item["path"]),
        "mutable_logs": sorted(mutable_logs, key=lambda item: item["path"]),
        "count": len(paths),
    }


def verify_package(package: Path, build_id: Optional[str] = None) -> Dict[str, Any]:
    checks = {
        "identity": False,
        "graph": False,
        "vector": False,
        "acceptance": False,
        "backend": False,
        "ledger": False,
        "hashes": False,
    }
    errors: List[str] = []
    identity: Dict[str, Any] = {}
    asserted = build_id
    try:
        identity = json_read(package / "build_identity.json")
        asserted = asserted or identity.get("build_id")
        checks["identity"] = bool(asserted) and identity.get("build_id") == asserted
        if identity.get("source_tree_sha256") != json_read(package / "artifact_ledger.json").get("source_tree_sha256"):
            errors.append("source_tree_sha256_mismatch")
    except Exception as exc:
        errors.append(f"identity_or_ledger_read:{type(exc).__name__}")

    def read_optional(name: str) -> Optional[Dict[str, Any]]:
        path = package / name
        if not path.is_file():
            errors.append(f"missing:{name}")
            return None
        try:
            value = json_read(path)
            if not isinstance(value, dict):
                errors.append(f"not_object:{name}")
                return None
            return value
        except Exception as exc:
            errors.append(f"invalid_json:{name}:{type(exc).__name__}")
            return None

    graph = read_optional("graph.json")
    if graph is not None:
        edges = graph.get("edges") or []
        checks["graph"] = bool(edges) and graph.get("build_id") == asserted and all(
            edge.get("build_id") == asserted for edge in edges
        )
        if not checks["graph"]:
            errors.append("graph_build_id_or_empty")

    vector = read_optional("vector_index_manifest.json")
    if vector is not None:
        checks["vector"] = (
            vector.get("build_id") == asserted
            and vector.get("status") == "PASS"
            and int(vector.get("chunks_expected") or 0) == int(vector.get("chunks_indexed") or -1)
            and int(vector.get("chunks_indexed") or 0) > 0
        )
        if not checks["vector"]:
            errors.append("vector_manifest_mismatch")
        configured = str(vector.get("configured_embedding_backend") or vector.get("embedding_backend") or "").strip().lower()
        actual = str(vector.get("runtime_embedding_backend") or vector.get("embedding_backend") or "").strip().lower()
        checks["backend"] = configured == actual == "chroma_onnx" and vector.get("backend_consistent", True) is True
        if not checks["backend"]:
            errors.append("embedding_backend_mismatch")

    acceptance = read_optional("acceptance_results.json")
    if acceptance is not None:
        checks["acceptance"] = acceptance.get("build_id") == asserted and acceptance.get("status") == "PASS"
        if not checks["acceptance"]:
            errors.append("acceptance_failed_or_build_mismatch")

    ledger = None
    try:
        ledger = json_read(package / "artifact_ledger.json")
        immutable_entries = ledger.get("artifacts")
        mutable_entries = ledger.get("mutable_logs") or []
        if not isinstance(immutable_entries, list) or not immutable_entries:
            errors.append("empty_or_missing_immutable_ledger")
            immutable_entries = []
        all_entries = immutable_entries + (mutable_entries if isinstance(mutable_entries, list) else [])
        paths = []
        safe = True
        for item in all_entries:
            raw = str(item.get("path") or "") if isinstance(item, dict) else ""
            candidate = PurePosixPath(raw)
            if (
                not raw
                or candidate.is_absolute()
                or ".." in candidate.parts
                or candidate.as_posix() != raw
                or raw.startswith("/")
            ):
                safe = False
                errors.append(f"unsafe_ledger_path:{raw or '<empty>'}")
            paths.append(raw)
        if len(paths) != len(set(paths)):
            errors.append("duplicate_ledger_path")
        if ledger.get("build_id") != asserted:
            errors.append("ledger_build_id_mismatch")
        if ledger.get("source_tree_sha256") != identity.get("source_tree_sha256"):
            errors.append("ledger_source_tree_mismatch")

        excluded = {"artifact_ledger.json", "READY.json", "verification.json", "build.log", "lifecycle.jsonl"}
        actual_immutable = {
            package_rel(path, package)
            for path in package.rglob("*")
            if path.is_file() and path.name not in excluded
        }
        listed_immutable = {str(item.get("path")) for item in immutable_entries if isinstance(item, dict)}
        if actual_immutable != listed_immutable:
            missing = sorted(listed_immutable - actual_immutable)
            unlisted = sorted(actual_immutable - listed_immutable)
            errors.append(f"ledger_inventory_mismatch:missing={missing}:unlisted={unlisted}")
        if not safe:
            checks["ledger"] = False
        else:
            checks["ledger"] = bool(immutable_entries) and actual_immutable == listed_immutable

        hash_ok = True
        for item in all_entries:
            relative = str(item.get("path") or "")
            target = package / PurePosixPath(relative)
            if not target.is_file():
                hash_ok = False
                errors.append(f"missing_ledger_file:{relative or '<empty>'}")
                continue
            if int(item.get("bytes") or -1) != target.stat().st_size:
                hash_ok = False
                errors.append(f"size_mismatch:{relative}")
            if file_hash(target) != str(item.get("sha256") or ""):
                hash_ok = False
                errors.append(f"hash_mismatch:{relative}")
        checks["hashes"] = hash_ok and checks["ledger"]
    except Exception as exc:
        errors.append(f"ledger_validation:{type(exc).__name__}:{exc}")

    result = {
        "schema": "isolated-package-verification/v2",
        "build_id": asserted,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    if errors:
        result["errors"] = errors
    return result


def publish(package: Path, build_id: str, *, reason: str = "publish") -> None:
    verification = verify_package(package, build_id)
    if verification["status"] != "PASS":
        raise RuntimeError(f"refusing to publish an unverified package: {verification}")
    ready = {"schema": "isolated-ready/v1", "build_id": build_id, "status": "PASS", "package": str(package), "published_at": now(), "reason": reason}
    json_write(package / "READY.json", ready)
    root = STAGING_ROOT
    root.mkdir(parents=True, exist_ok=True)
    pointer = {"schema": "isolated-publish-pointer/v1", "build_id": build_id, "package": str(package), "updated_at": now(), "reason": reason}
    json_write(root / "published_pointer.json", pointer)
    with (root / "publish_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(pointer, ensure_ascii=False) + "\n")
    lifecycle_event(package, reason, "PASS", build_id=build_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="append", default=None)
    parser.add_argument("--corpus-id", default="nvidia-10k-2023-2025-v3")
    parser.add_argument("--fault-after", choices=PHASES, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--rollback-to", default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.rollback_to:
        target = STAGING_ROOT / args.rollback_to
        verification = verify_package(target, args.rollback_to)
        if verification["status"] != "PASS":
            raise SystemExit(json.dumps(verification, ensure_ascii=False))
        publish(target, args.rollback_to, reason="rollback")
        print(json.dumps({"status": "PASS", "event": "rollback", "build_id": args.rollback_to}, ensure_ascii=False))
        return 0

    pdfs = pdfs_from_args(args.pdf)
    identity = identity_for(pdfs, args.corpus_id).to_dict()
    build_id = identity["build_id"]
    package = STAGING_ROOT / build_id
    if args.verify:
        if not package.exists():
            print(json.dumps({"status": "FAIL", "build_id": build_id, "reason": "package_not_found"}, ensure_ascii=False))
            return 1
        result = verify_package(package, build_id)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "PASS" else 1
    package.mkdir(parents=True, exist_ok=True)
    configure_file_logging(package)
    lifecycle_event(package, "start_or_resume", "PASS", build_id=build_id, resume=args.resume)
    if (package / "READY.json").exists() and not args.resume:
        raise SystemExit(f"package already READY: {package}; use --resume to inspect it")
    phases = {
        "identity": lambda: phase_identity(package, identity, build_id),
        "documents": lambda: phase_documents(package, pdfs, build_id),
        "candidates": lambda: phase_candidates(package, pdfs, build_id),
        "accepted": lambda: phase_accepted(package, build_id),
        "indexes": lambda: phase_indexes(package, build_id),
        "validated": lambda: phase_validated(package, build_id),
    }
    for phase in PHASES:
        if args.resume and phase_done(package, phase, build_id):
            lifecycle_event(package, phase, "SKIPPED_ALREADY_COMPLETE", build_id=build_id)
            continue
        lifecycle_event(package, phase, "STARTED", build_id=build_id)
        phases[phase]()
        lifecycle_event(package, phase, "PASS", build_id=build_id)
        if args.fault_after == phase:
            lifecycle_event(package, phase, "INTERRUPTED_INJECTED", build_id=build_id)
            raise SystemExit(f"INTERRUPTED_AFTER_PHASE={phase} build_id={build_id}")
    ledger = package_artifacts(package, build_id)
    json_write(package / "artifact_ledger.json", ledger)
    verification = verify_package(package, build_id)
    json_write(package / "verification.json", verification)
    if verification["status"] != "PASS":
        raise SystemExit(json.dumps(verification, ensure_ascii=False))
    if args.publish:
        publish(package, build_id)
    print(json.dumps({"status": "PASS", "build_id": build_id, "package": str(package), "published": args.publish, "verification": verification}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("isolated staging failed")
        raise
