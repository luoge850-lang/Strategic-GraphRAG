# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: FastAPI Backend Server
==========================================
REST API for the Strategic-GraphRAG system.

Endpoints:
  POST /query          — GraphRAG financial analysis
  POST /query/vector   — Vector RAG baseline (for comparison)
  GET  /graph/statistics — Knowledge graph statistics
  GET  /graph/subgraph  — Subgraph for visualization
  GET  /evidence/{id}   — Evidence trace for a specific path
"""

import os
import sys
import json
import logging
import hmac
import time
import uuid
from collections import defaultdict, deque
from typing import Optional, List, Dict, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

load_dotenv()

# ── App Setup ──
app = FastAPI(
    title="Strategic-GraphRAG API",
    description="Temporal Causal Knowledge Graph Framework for Financial Risk Inference",
    version="1.0.0",
)

_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

_API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
_API_KEY = os.getenv("API_KEY", "").strip()
_RATE_LIMIT_PER_MINUTE = max(
    int(os.getenv("RATE_LIMIT_PER_MINUTE", "60") or 60),
    1,
)
_RATE_BUCKETS = defaultdict(deque)
_AUTH_EXEMPT_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def request_guard(request: Request, call_next):
    """Add request tracing, basic abuse protection, and optional API auth."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()

    if _API_AUTH_ENABLED and request.url.path not in _AUTH_EXEMPT_PATHS:
        provided_key = request.headers.get("X-API-Key", "")
        if not _API_KEY or not hmac.compare_digest(provided_key, _API_KEY):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "A valid X-API-Key is required.",
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )

    client_key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _RATE_BUCKETS[client_key]
    while bucket and now - bucket[0] >= 60:
        bucket.popleft()
    if request.url.path not in {"/", "/health"} and len(bucket) >= _RATE_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "RATE_LIMITED",
                    "message": "Too many requests. Retry later.",
                    "request_id": request_id,
                }
            },
            headers={"X-Request-ID": request_id, "Retry-After": "60"},
        )
    bucket.append(now)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request error request_id=%s", request_id)
        response = JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "The request could not be completed.",
                    "request_id": request_id,
                }
            },
        )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = str(
        round((time.perf_counter() - started) * 1000, 2)
    )
    return response


@app.exception_handler(HTTPException)
async def structured_http_error(request: Request, exc: HTTPException):
    """Avoid leaking provider/database internals through API errors."""
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    if isinstance(exc.detail, dict):
        error = dict(exc.detail)
        error.setdefault("code", "HTTP_ERROR")
        error.setdefault("message", "The request could not be completed.")
    elif exc.status_code >= 500:
        error = {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "The request could not be completed.",
        }
    else:
        error = {"code": "HTTP_ERROR", "message": str(exc.detail)}
    error["request_id"] = request_id
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error},
        headers={"X-Request-ID": request_id},
    )

# Serve the built Vite application when available.  The public/index.html
# fallback keeps the API usable before a frontend build has been produced.
FRONTEND_ROOT = Path(__file__).resolve().parent.parent.parent / "frontend"


def get_frontend_index() -> Path:
    dist_index = FRONTEND_ROOT / "dist" / "index.html"
    public_index = FRONTEND_ROOT / "public" / "index.html"
    return dist_index if dist_index.exists() else public_index


# Vite emits JavaScript/CSS into ``dist/assets``.  Returning index.html from
# the root route is not enough: without this mount the browser receives the
# shell but every hashed asset URL returns 404.
FRONTEND_ASSETS = FRONTEND_ROOT / "dist" / "assets"
if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="frontend-assets")

# ── Lazy-loaded engines ──
_graph_engine = None
_vector_engine = None
_schema_manager = None


def get_graph_engine():
    global _graph_engine
    if _graph_engine is None:
        from strategic_graphrag.engine.graph_rag_engine import GraphRAGEngine
        _graph_engine = GraphRAGEngine()
    return _graph_engine


def get_vector_engine():
    global _vector_engine
    if _vector_engine is None:
        from strategic_graphrag.engine.vector_rag_baseline import VectorRAGBaseline
        _vector_engine = VectorRAGBaseline()
    return _vector_engine


def get_schema_manager():
    global _schema_manager
    if _schema_manager is None:
        from strategic_graphrag.schema.manager import SchemaManager
        _schema_manager = SchemaManager()
        _schema_manager.connect()
    return _schema_manager


def resolve_active_filing(requested: Optional[str] = None) -> Optional[str]:
    """Resolve the corpus scope used by graph-facing endpoints."""
    return requested or os.getenv("GRAPH_ACTIVE_FILING", "").strip() or None


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language financial question")
    max_paths: int = Field(default=10, ge=1, le=30)
    year_filter: Optional[int] = Field(
        default=None,
        description="Backward-compatible minimum fiscal year filter",
    )
    year_start: Optional[int] = Field(default=None, ge=1900, le=2100)
    year_end: Optional[int] = Field(default=None, ge=1900, le=2100)
    source_filing: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional filing scope; defaults to GRAPH_ACTIVE_FILING",
    )
    retrieval_mode: Literal["graph", "hybrid"] = Field(
        default="hybrid",
        description="Graph-only baseline or vector+graph hybrid retrieval",
    )
    vector_top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    intent: str
    intent_display: str
    answer: str
    structured_report: Optional[Dict] = None
    paths: List[Dict]
    evidence_sentences: List[str]
    metadata: Dict


class VectorQueryResponse(BaseModel):
    query: str
    answer: str
    documents: List[str]


class GraphStats(BaseModel):
    total_nodes: int
    total_relationships: int
    graph_nodes: int = 0
    graph_relationships: int = 0
    by_label: Dict[str, int]
    by_relationship: Dict[str, int]
    source_filing: Optional[str] = None


class TemporalEvent(BaseModel):
    target: Optional[str] = None
    relation: str
    strength: Optional[str] = None
    year: int
    evidence: Optional[str] = None
    page: Optional[int] = None
    filing: Optional[str] = None
    evidence_id: Optional[str] = None


class SubgraphRequest(BaseModel):
    entity_ids: List[str] = Field(default_factory=list, description="Focus entity IDs")
    max_nodes: int = Field(default=50, ge=10, le=200)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    index_path = get_frontend_index()
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "service": "Strategic-GraphRAG API",
        "version": "1.0.0",
        "endpoints": ["POST /query", "POST /query/vector", "GET /graph/statistics", "GET /graph/subgraph", "GET /evidence/{entity_id}", "GET /graph/temporal/{risk_id}"],
    }


@app.post("/query", response_model=QueryResponse)
async def graphrag_query(req: QueryRequest):
    """
    Execute a full GraphRAG inference pipeline.
    Returns structured causal analysis with evidence provenance.
    """
    try:
        engine = get_graph_engine()
        year_start = req.year_start if req.year_start is not None else req.year_filter
        if req.year_end is not None and year_start is not None and req.year_end < year_start:
            raise HTTPException(status_code=422, detail="year_end must be >= year_start")
        vector_engine = None
        if req.retrieval_mode == "hybrid":
            try:
                vector_engine = get_vector_engine()
            except Exception as e:
                logger.warning("Vector engine unavailable; continuing in degraded hybrid mode: %s", e)
        result = engine.query(
            req.question,
            top_k=req.max_paths,
            year_start=year_start,
            year_end=req.year_end,
            source_filing=resolve_active_filing(req.source_filing),
            retrieval_mode=req.retrieval_mode,
            vector_engine=vector_engine,
            vector_top_k=req.vector_top_k,
        )
        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/vector", response_model=VectorQueryResponse)
async def vector_query(req: QueryRequest):
    """
    Execute standard Vector RAG for comparison.
    """
    try:
        engine = get_vector_engine()
        answer, docs = engine.ask(
            req.question,
            k=5,
            source_filing=resolve_active_filing(req.source_filing),
        )
        return VectorQueryResponse(query=req.question, answer=answer, documents=docs)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/statistics", response_model=GraphStats)
async def graph_statistics():
    """
    Get knowledge graph statistics.
    """
    try:
        mgr = get_schema_manager()
        source_filing = resolve_active_filing()
        stats = mgr.stats(source_filing=source_filing)
        return GraphStats(
            total_nodes=stats.get("total_nodes", 0),
            total_relationships=stats.get("total_rels", 0),
            graph_nodes=stats.get("graph_nodes", 0),
            graph_relationships=stats.get("graph_relationships", 0),
            by_label=stats.get("by_label", {}),
            by_relationship=stats.get("by_relationship", {}),
            source_filing=source_filing,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/subgraph")
async def get_subgraph(
    entity: Optional[str] = Query(None, description="Focus entity name/ID"),
    limit: int = Query(500, ge=10, le=2000),
    source_filing: Optional[str] = Query(
        None,
        max_length=200,
        description="Optional filing scope; defaults to GRAPH_ACTIVE_FILING",
    ),
):
    """
    Get subgraph data for visualization (nodes + edges in JSON).
    If entity is provided, returns 2-hop neighborhood around that entity.
    Otherwise returns a representative sample of the full graph.
    """
    mgr = get_schema_manager()
    source_filing = resolve_active_filing(source_filing)
    try:
        if entity:
            cypher = """
            MATCH (n)
            WHERE toLower(coalesce(n.name, n.id)) CONTAINS toLower($entity)
               OR toLower(n.id) CONTAINS toLower($entity)
            WITH n LIMIT 1
            MATCH (n)-[rels*1..2]-(m)
            WHERE NOT n:Sentence AND NOT m:Sentence
              AND ALL(rel IN rels WHERE
                ($source_filing IS NULL OR
                 coalesce(rel.source_filing, rel.filing, '') = $source_filing)
                AND rel.evidence_id IS NOT NULL
                AND EXISTS {
                    MATCH (claim:EvidenceClaim {id: rel.evidence_id})
                    WHERE claim.verification_status = 'VERBATIM'
                })
            UNWIND rels AS rel
            WITH collect(DISTINCT n) + collect(DISTINCT m) AS allNodes, collect(DISTINCT rel) AS allRels
            UNWIND allNodes AS nd
            UNWIND allRels AS rel
            WITH DISTINCT nd, rel
            WHERE startNode(rel) = nd OR endNode(rel) = nd
            WITH DISTINCT
                [x IN collect(DISTINCT nd) | {id: x.id, name: coalesce(x.name, x.id), labels: labels(x)}] AS nodes,
                [x IN collect(DISTINCT rel) | {source: startNode(x).id, target: endNode(x).id, type: type(x), evidence_id: x.evidence_id}] AS edges
            RETURN nodes, edges
            LIMIT 1
            """
            with mgr.driver.session() as session:
                result = session.run(
                    cypher,
                    entity=entity,
                    source_filing=source_filing,
                )
                records = list(result)
                if records and records[0]:
                    data = records[0].data()
                    nodes_raw = data.get("nodes", [])
                    edges_raw = data.get("edges", [])
                    # Preserve separate evidence-backed relation instances.
                    # Two claims can support the same endpoint/type pair.
                    seen_n = set(); nodes = []
                    for nd in nodes_raw:
                        if nd["id"] not in seen_n: seen_n.add(nd["id"]); nodes.append(nd)
                    seen_e = set(); edges = []
                    for e in edges_raw:
                        k = f'{e["source"]}|{e["target"]}|{e["type"]}|{e.get("evidence_id") or ""}'
                        if k not in seen_e: seen_e.add(k); edges.append(e)
                    return {"nodes": nodes[:limit], "edges": edges[:limit * 2]}
        else:
            # Full graph sample: get nodes with highest degree relationships
            cypher = """
            MATCH (n)-[r]->(m)
            WHERE NOT n:Sentence AND NOT m:Sentence
              AND ($source_filing IS NULL OR
                   coalesce(r.source_filing, r.filing, '') = $source_filing)
              AND r.evidence_id IS NOT NULL
              AND EXISTS {
                  MATCH (claim:EvidenceClaim {id: r.evidence_id})
                  WHERE claim.verification_status = 'VERBATIM'
              }
            WITH n, r, m
            WITH collect(DISTINCT n) + collect(DISTINCT m) AS allNodes, collect(DISTINCT r) AS allRels
            UNWIND allNodes AS nd
            UNWIND allRels AS rel
            WITH DISTINCT nd, rel
            WHERE (startNode(rel) = nd OR endNode(rel) = nd)
            WITH DISTINCT
                [x IN collect(DISTINCT nd) | {id: x.id, name: coalesce(x.name, x.id), labels: labels(x)}] AS nodes,
                [x IN collect(DISTINCT rel) | {source: startNode(x).id, target: endNode(x).id, type: type(x), evidence_id: x.evidence_id}] AS edges
            RETURN nodes, edges
            LIMIT 1
            """
            with mgr.driver.session() as session:
                result = session.run(cypher, source_filing=source_filing)
                records = list(result)
                if records and records[0]:
                    data = records[0].data()
                    nodes_raw = data.get("nodes", [])
                    edges_raw = data.get("edges", [])
                    seen_n = set(); nodes = []
                    for nd in nodes_raw:
                        if nd["id"] not in seen_n: seen_n.add(nd["id"]); nodes.append(nd)
                    seen_e = set(); edges = []
                    for e in edges_raw:
                        k = f'{e["source"]}|{e["target"]}|{e["type"]}|{e.get("evidence_id") or ""}'
                        if k not in seen_e: seen_e.add(k); edges.append(e)
                    return {"nodes": nodes[:limit], "edges": edges[:limit * 2]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"nodes": [], "edges": []}


@app.get("/evidence/{entity_id}")
async def get_evidence(entity_id: str, limit: int = Query(10, ge=1, le=50)):
    """
    Get evidence sentences for a specific entity.
    """
    try:
        engine = get_graph_engine()
        evidence = engine.path_finder.find_evidence_for_entity(
            entity_id,
            limit=limit,
            source_filing=resolve_active_filing(),
        )
        return {"entity_id": entity_id, "evidence": evidence}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/temporal/{risk_id}", response_model=List[TemporalEvent])
async def temporal_evolution(
    risk_id: str,
    limit: int = Query(20, ge=1, le=100),
    source_filing: Optional[str] = Query(None, max_length=200),
):
    """Return year-anchored evidence links for one risk factor."""
    try:
        engine = get_graph_engine()
        rows = engine.path_finder.find_temporal_evolution(
            risk_id,
            source_filing=resolve_active_filing(source_filing),
        )
        return rows[:limit]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """System health check."""
    neo4j_ok = False
    try:
        mgr = get_schema_manager()
        mgr.driver.verify_connectivity()
        neo4j_ok = True
    except Exception:
        pass

    llm_available = False
    llm_provider = None
    llm_model = None
    try:
        from strategic_graphrag.llm_provider import get_llm
        llm = get_llm()
        llm_available = llm.available
        llm_provider = llm.provider
        llm_model = llm.default_model
    except Exception:
        pass

    overall = neo4j_ok and llm_available

    return {
        "status": "healthy" if overall else "degraded",
        "neo4j": "connected" if neo4j_ok else "disconnected",
        "llm": "configured" if llm_available else "unavailable",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "active_filing": resolve_active_filing(),
        "auth_enabled": _API_AUTH_ENABLED,
    }


# ── Run ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
