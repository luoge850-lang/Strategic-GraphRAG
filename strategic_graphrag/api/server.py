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

import asyncio
import os
import sys
import json
import logging
import hmac
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from typing import Optional, List, Dict, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
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
    version="3.0.0",
)

_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173,http://localhost:4173",
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
_AUTH_EXEMPT_PATHS = {"/", "/health", "/health/live", "/health/ready", "/docs", "/openapi.json", "/redoc"}
_QUERY_CACHE_TTL_SECONDS = max(int(os.getenv("QUERY_CACHE_TTL_SECONDS", "300") or 300), 0)
_QUERY_CACHE_MAX_ENTRIES = max(int(os.getenv("QUERY_CACHE_MAX_ENTRIES", "128") or 128), 1)
_QUERY_CACHE = OrderedDict()
_GRAPH_CACHE_TTL_SECONDS = max(int(os.getenv("GRAPH_CACHE_TTL_SECONDS", "60") or 60), 0)
_GRAPH_CACHE_MAX_ENTRIES = max(int(os.getenv("GRAPH_CACHE_MAX_ENTRIES", "16") or 16), 1)
_GRAPH_CACHE = OrderedDict()
_STARTED_AT = time.time()
_READINESS = {"status": "unknown", "checked_at": None, "dependencies": {}}


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
    if request.url.path not in {"/", "/health", "/health/live", "/health/ready"} and len(bucket) >= _RATE_LIMIT_PER_MINUTE:
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
_graph_engine_lock = threading.Lock()
_vector_engine_lock = threading.Lock()
_schema_manager_lock = threading.Lock()
_warmup_task = None


def get_graph_engine():
    global _graph_engine
    if _graph_engine is None:
        with _graph_engine_lock:
            if _graph_engine is None:
                from strategic_graphrag.engine.graph_rag_engine import GraphRAGEngine
                _graph_engine = GraphRAGEngine()
    return _graph_engine


def get_vector_engine():
    global _vector_engine
    if _vector_engine is None:
        with _vector_engine_lock:
            if _vector_engine is None:
                from strategic_graphrag.engine.vector_rag_baseline import VectorRAGBaseline
                _vector_engine = VectorRAGBaseline()
    return _vector_engine


def get_schema_manager():
    global _schema_manager
    if _schema_manager is None:
        with _schema_manager_lock:
            if _schema_manager is None:
                from strategic_graphrag.schema.manager import SchemaManager
                candidate = SchemaManager()
                if not candidate.connect():
                    raise RuntimeError("Neo4j connection failed")
                _schema_manager = candidate
    return _schema_manager


@app.on_event("startup")
async def warm_runtime_dependencies():
    """Warm Hybrid retrieval off the request path; readiness stays non-blocking."""
    global _warmup_task
    _warmup_task = asyncio.create_task(run_in_threadpool(get_vector_engine))


def resolve_active_filing(requested: Optional[str] = None) -> Optional[str]:
    """Resolve the corpus scope used by graph-facing endpoints."""
    return requested or os.getenv("GRAPH_ACTIVE_FILING", "").strip() or None


def resolve_source_filing(
    requested: Optional[str],
    cross_filing: bool,
) -> Optional[str]:
    """Resolve an explicit all-filings request without hiding it behind env.

    The UI uses ``cross_filing=true`` for the all-years view.  A single-filing
    request continues to fall back to GRAPH_ACTIVE_FILING for backwards
    compatibility, while an explicit cross-filing request is always global.
    """
    return None if cross_filing else resolve_active_filing(requested)


def _graph_cache_get(key: str):
    cached = _GRAPH_CACHE.get(key)
    if not cached:
        return None
    if time.monotonic() - cached[0] > _GRAPH_CACHE_TTL_SECONDS:
        _GRAPH_CACHE.pop(key, None)
        return None
    _GRAPH_CACHE.move_to_end(key)
    return deepcopy(cached[1])


def _graph_cache_put(key: str, value):
    if _GRAPH_CACHE_TTL_SECONDS <= 0:
        return
    _GRAPH_CACHE[key] = (time.monotonic(), deepcopy(value))
    _GRAPH_CACHE.move_to_end(key)
    while len(_GRAPH_CACHE) > _GRAPH_CACHE_MAX_ENTRIES:
        _GRAPH_CACHE.popitem(last=False)


def _load_strict_subgraph(entity: Optional[str], limit: int, source_filing: Optional[str]):
    """Fetch strict evidence-backed edges without a nodes x edges Cartesian product."""
    mgr = get_schema_manager()
    rows = mgr._read(
        """
        MATCH (claim:EvidenceClaim)-[:ABOUT_SOURCE]->(source)
        MATCH (claim)-[:ABOUT_TARGET]->(target)
        MATCH (source)-[rel]->(target)
        WHERE rel.evidence_id = claim.id
          AND claim.verification_status = 'VERBATIM'
          AND ($source_filing IS NULL OR
               coalesce(rel.source_filing, rel.filing, '') = $source_filing)
          AND ($entity IS NULL OR
               toLower(coalesce(source.name, source.id, '')) CONTAINS toLower($entity) OR
               toLower(coalesce(target.name, target.id, '')) CONTAINS toLower($entity))
        RETURN
          coalesce(source.id, elementId(source)) AS source_id,
          coalesce(source.name, source.id, elementId(source)) AS source_name,
          labels(source) AS source_labels,
          coalesce(target.id, elementId(target)) AS target_id,
          coalesce(target.name, target.id, elementId(target)) AS target_name,
          labels(target) AS target_labels,
          type(rel) AS relationship_type,
          claim.id AS evidence_id
        ORDER BY claim.filing_fiscal_year DESC, claim.page, claim.id
        LIMIT $edge_limit
        """,
        entity=entity,
        source_filing=source_filing,
        edge_limit=limit * 2,
    )
    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []
    seen_edges = set()
    for row in rows:
        for side in ("source", "target"):
            node_id = row[f"{side}_id"]
            nodes[node_id] = {
                "id": node_id,
                "name": row[f"{side}_name"],
                "labels": row[f"{side}_labels"],
            }
        edge = {
            "source": row["source_id"],
            "target": row["target_id"],
            "type": row["relationship_type"],
            "evidence_id": row["evidence_id"],
        }
        key = tuple(edge.values())
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append(edge)
    return {"nodes": list(nodes.values())[:limit], "edges": edges[: limit * 2]}


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
    retrieval_mode: Literal["auto", "graph", "hybrid"] = Field(
        default="auto",
        description="Adaptive, graph-only, or vector+graph hybrid retrieval",
    )
    vector_top_k: int = Field(default=5, ge=1, le=20)
    cross_filing: bool = Field(
        default=False,
        description="Explicitly search all indexed filings; disabled by default",
    )
    use_cache: bool = Field(
        default=True,
        description="Reuse an identical successful query within the bounded TTL cache",
    )


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


class TemporalChangeResponse(BaseModel):
    id: str
    change_type: str
    source_id: str
    relation_type: str
    target_id: str
    from_year: int
    to_year: int
    from_value: Optional[float] = None
    to_value: Optional[float] = None
    absolute_delta: Optional[float] = None
    percent_delta: Optional[float] = None
    earlier_claim_id: str
    later_claim_id: str
    semantics: str


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
        cache_key = json.dumps(
            req.model_dump(exclude={"use_cache"}), sort_keys=True, ensure_ascii=False
        )
        now = time.monotonic()
        cached = _QUERY_CACHE.get(cache_key) if req.use_cache else None
        if cached and now - cached[0] <= _QUERY_CACHE_TTL_SECONDS:
            result = deepcopy(cached[1])
            result.setdefault("metadata", {})["cache"] = {
                "hit": True,
                "age_ms": round((now - cached[0]) * 1000, 2),
                "ttl_seconds": _QUERY_CACHE_TTL_SECONDS,
            }
            _QUERY_CACHE.move_to_end(cache_key)
            return QueryResponse(**result)
        if cached:
            _QUERY_CACHE.pop(cache_key, None)
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
        result = await run_in_threadpool(
            engine.query,
            req.question,
            top_k=req.max_paths,
            year_start=year_start,
            year_end=req.year_end,
            source_filing=resolve_source_filing(req.source_filing, req.cross_filing),
            cross_filing=req.cross_filing,
            retrieval_mode=req.retrieval_mode,
            vector_engine=vector_engine,
            vector_top_k=req.vector_top_k,
        )
        result.setdefault("metadata", {})["cache"] = {
            "hit": False,
            "ttl_seconds": _QUERY_CACHE_TTL_SECONDS,
        }
        cacheable = not str(result.get("answer") or "").startswith("[CONNECTION ERROR]")
        if req.use_cache and _QUERY_CACHE_TTL_SECONDS > 0 and cacheable:
            _QUERY_CACHE[cache_key] = (time.monotonic(), deepcopy(result))
            _QUERY_CACHE.move_to_end(cache_key)
            while len(_QUERY_CACHE) > _QUERY_CACHE_MAX_ENTRIES:
                _QUERY_CACHE.popitem(last=False)
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
            source_filing=resolve_source_filing(req.source_filing, req.cross_filing),
        )
        return VectorQueryResponse(query=req.question, answer=answer, documents=docs)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/statistics", response_model=GraphStats)
async def graph_statistics(
    source_filing: Optional[str] = Query(None, max_length=200),
    cross_filing: bool = Query(False),
):
    """
    Get knowledge graph statistics.
    """
    try:
        scoped_filing = None if cross_filing else resolve_active_filing(source_filing)
        cache_key = f"stats:{scoped_filing or 'all'}"
        stats = _graph_cache_get(cache_key)
        if stats is None:
            mgr = get_schema_manager()
            stats = await asyncio.wait_for(
                run_in_threadpool(mgr.stats, scoped_filing),
                timeout=float(os.getenv("GRAPH_ENDPOINT_TIMEOUT_SECONDS", "12")),
            )
            _graph_cache_put(cache_key, stats)
        return GraphStats(
            total_nodes=stats.get("total_nodes", 0),
            total_relationships=stats.get("total_rels", 0),
            graph_nodes=stats.get("graph_nodes", 0),
            graph_relationships=stats.get("graph_relationships", 0),
            by_label=stats.get("by_label", {}),
            by_relationship=stats.get("by_relationship", {}),
            source_filing=scoped_filing,
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
    cross_filing: bool = Query(False),
):
    """
    Get subgraph data for visualization (nodes + edges in JSON).
    If entity is provided, returns strict incident EvidenceClaim edges.
    Otherwise returns a bounded strict graph sample.
    """
    source_filing = None if cross_filing else resolve_active_filing(source_filing)
    try:
        cache_key = f"subgraph:{source_filing or 'all'}:{entity or ''}:{limit}"
        cached = _graph_cache_get(cache_key)
        if cached is not None:
            return cached
        result = await asyncio.wait_for(
            run_in_threadpool(_load_strict_subgraph, entity, limit, source_filing),
            timeout=float(os.getenv("GRAPH_ENDPOINT_TIMEOUT_SECONDS", "12")),
        )
        _graph_cache_put(cache_key, result)
        return result
        # Legacy Cypher retained below temporarily for rollback reference; the
        # strict query above is the only reachable production path.
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
async def get_evidence(
    entity_id: str,
    limit: int = Query(10, ge=1, le=50),
    source_filing: Optional[str] = Query(None, max_length=200),
    cross_filing: bool = Query(False),
):
    """
    Get evidence sentences for a specific entity.
    """
    try:
        engine = get_graph_engine()
        evidence = engine.path_finder.find_evidence_for_entity(
            entity_id,
            limit=limit,
            source_filing=resolve_source_filing(source_filing, cross_filing),
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


@app.get("/graph/temporal-changes/{entity_id}", response_model=List[TemporalChangeResponse])
async def temporal_changes(entity_id: str, limit: int = Query(50, ge=1, le=200)):
    """Return observed cross-filing changes backed by two EvidenceClaims."""
    try:
        rows = await asyncio.wait_for(
            run_in_threadpool(
                get_schema_manager()._read,
                """
                MATCH (earlier:EvidenceClaim)-[:HAS_TEMPORAL_CHANGE]->(change:TemporalChange)-[:CHANGES_TO]->(later:EvidenceClaim)
                WHERE toLower(change.source_id) CONTAINS toLower($entity_id)
                   OR toLower(change.target_id) CONTAINS toLower($entity_id)
                RETURN change.id AS id, change.change_type AS change_type,
                       change.source_id AS source_id, change.relation_type AS relation_type,
                       change.target_id AS target_id,
                       change.earlier_year AS from_year, change.later_year AS to_year,
                       change.from_value AS from_value, change.to_value AS to_value,
                       change.absolute_delta AS absolute_delta, change.percent_delta AS percent_delta,
                       earlier.id AS earlier_claim_id, later.id AS later_claim_id,
                       change.semantics AS semantics
                ORDER BY from_year, to_year, id LIMIT $limit
                """,
                entity_id=entity_id,
                limit=limit,
            ),
            timeout=float(os.getenv("GRAPH_ENDPOINT_TIMEOUT_SECONDS", "12")),
        )
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/live")
async def health_live():
    """Process liveness only; never waits for Aura, embeddings, or an LLM."""
    return {
        "status": "alive",
        "version": app.version,
        "uptime_seconds": round(time.time() - _STARTED_AT, 2),
    }


def _probe_readiness() -> Dict:
    dependencies: Dict[str, Dict] = {}
    try:
        rows = get_schema_manager()._read("RETURN 1 AS ok")
        dependencies["neo4j"] = {"ready": bool(rows and rows[0].get("ok") == 1)}
    except Exception as exc:
        dependencies["neo4j"] = {"ready": False, "error": type(exc).__name__}
    if _vector_engine is None:
        dependencies["vector"] = {"ready": False, "status": "warming"}
    else:
        try:
            dependencies["vector"] = _vector_engine.diagnostics()
        except Exception as exc:
            dependencies["vector"] = {"ready": False, "error": type(exc).__name__}
    try:
        from strategic_graphrag.llm_provider import get_llm
        llm = get_llm()
        dependencies["llm"] = {
            "ready": bool(llm.available),
            "provider": llm.provider,
            "model": llm.default_model,
        }
    except Exception as exc:
        dependencies["llm"] = {"ready": False, "error": type(exc).__name__}
    return {
        "status": "ready" if all(item.get("ready") for item in dependencies.values()) else "degraded",
        "checked_at": time.time(),
        "dependencies": dependencies,
        "active_filing": resolve_active_filing(),
        "auth_enabled": _API_AUTH_ENABLED,
    }


@app.get("/health/ready")
async def health_ready():
    """Bounded dependency readiness probe for deployment and acceptance tests."""
    global _READINESS
    try:
        _READINESS = await asyncio.wait_for(
            run_in_threadpool(_probe_readiness),
            timeout=float(os.getenv("READINESS_TIMEOUT_SECONDS", "12")),
        )
    except asyncio.TimeoutError:
        _READINESS = {
            "status": "degraded",
            "checked_at": time.time(),
            "dependencies": {"probe": {"ready": False, "error": "timeout"}},
        }
    status_code = 200 if _READINESS["status"] == "ready" else 503
    return JSONResponse(status_code=status_code, content=_READINESS)


@app.get("/health")
async def health_check():
    """Backward-compatible cheap status; use /health/ready for live dependencies."""
    return {
        "status": "healthy",
        "liveness": "alive",
        "readiness": _READINESS,
        "version": app.version,
        "uptime_seconds": round(time.time() - _STARTED_AT, 2),
    }


# ── Run ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
