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
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Serve the built Vite application when available.  The public/index.html
# fallback keeps the API usable before a frontend build has been produced.
FRONTEND_ROOT = Path(__file__).resolve().parent.parent.parent / "frontend"


def get_frontend_index() -> Path:
    dist_index = FRONTEND_ROOT / "dist" / "index.html"
    public_index = FRONTEND_ROOT / "public" / "index.html"
    return dist_index if dist_index.exists() else public_index

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


class QueryResponse(BaseModel):
    query: str
    intent: str
    intent_display: str
    answer: str
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
    by_label: Dict[str, int]
    by_relationship: Dict[str, int]


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

from fastapi.responses import FileResponse

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
        result = engine.query(
            req.question,
            top_k=req.max_paths,
            year_start=year_start,
            year_end=req.year_end,
        )
        return QueryResponse(**result)
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
        answer, docs = engine.ask(req.question, k=5)
        return VectorQueryResponse(query=req.question, answer=answer, documents=docs)
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
        stats = mgr.stats()
        return GraphStats(
            total_nodes=stats.get("total_nodes", 0),
            total_relationships=stats.get("total_rels", 0),
            by_label=stats.get("by_label", {}),
            by_relationship=stats.get("by_relationship", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/subgraph")
async def get_subgraph(
    entity: Optional[str] = Query(None, description="Focus entity name/ID"),
    limit: int = Query(500, ge=10, le=2000),
):
    """
    Get subgraph data for visualization (nodes + edges in JSON).
    If entity is provided, returns 2-hop neighborhood around that entity.
    Otherwise returns a representative sample of the full graph.
    """
    mgr = get_schema_manager()
    try:
        if entity:
            cypher = """
            MATCH (n)
            WHERE toLower(coalesce(n.name, n.id)) CONTAINS toLower($entity)
               OR toLower(n.id) CONTAINS toLower($entity)
            WITH n LIMIT 1
            MATCH (n)-[r*1..2]-(m)
            UNWIND r AS rel
            WITH collect(DISTINCT n) + collect(DISTINCT m) AS allNodes, collect(DISTINCT rel) AS allRels
            UNWIND allNodes AS nd
            UNWIND allRels AS rel
            WITH DISTINCT nd, rel
            WHERE startNode(rel) = nd OR endNode(rel) = nd
            WITH DISTINCT
                [x IN collect(DISTINCT nd) | {id: x.id, name: coalesce(x.name, x.id), labels: labels(x)}] AS nodes,
                [x IN collect(DISTINCT rel) | {source: startNode(x).id, target: endNode(x).id, type: type(x)}] AS edges
            RETURN nodes, edges
            LIMIT 1
            """
            with mgr.driver.session() as session:
                result = session.run(cypher, entity=entity)
                records = list(result)
                if records and records[0]:
                    data = records[0].data()
                    nodes_raw = data.get("nodes", [])
                    edges_raw = data.get("edges", [])
                    # Deduplicate nodes/edges
                    seen_n = set(); nodes = []
                    for nd in nodes_raw:
                        if nd["id"] not in seen_n: seen_n.add(nd["id"]); nodes.append(nd)
                    seen_e = set(); edges = []
                    for e in edges_raw:
                        k = f'{e["source"]}|{e["target"]}|{e["type"]}'
                        if k not in seen_e: seen_e.add(k); edges.append(e)
                    return {"nodes": nodes[:limit], "edges": edges[:limit * 2]}
        else:
            # Full graph sample: get nodes with highest degree relationships
            cypher = """
            MATCH (n)-[r]->(m)
            WHERE NOT n:Sentence AND NOT m:Sentence
            WITH n, r, m
            WITH collect(DISTINCT n) + collect(DISTINCT m) AS allNodes, collect(DISTINCT r) AS allRels
            UNWIND allNodes AS nd
            UNWIND allRels AS rel
            WITH DISTINCT nd, rel
            WHERE (startNode(rel) = nd OR endNode(rel) = nd)
            WITH DISTINCT
                [x IN collect(DISTINCT nd) | {id: x.id, name: coalesce(x.name, x.id), labels: labels(x)}] AS nodes,
                [x IN collect(DISTINCT rel) | {source: startNode(x).id, target: endNode(x).id, type: type(x)}] AS edges
            RETURN nodes, edges
            LIMIT 1
            """
            with mgr.driver.session() as session:
                result = session.run(cypher)
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
                        k = f'{e["source"]}|{e["target"]}|{e["type"]}'
                        if k not in seen_e: seen_e.add(k); edges.append(e)
                    return {"nodes": nodes[:limit], "edges": edges[:limit * 2]}
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
        evidence = engine.path_finder.find_evidence_for_entity(entity_id, limit=limit)
        return {"entity_id": entity_id, "evidence": evidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/temporal/{risk_id}", response_model=List[TemporalEvent])
async def temporal_evolution(
    risk_id: str,
    limit: int = Query(20, ge=1, le=100),
):
    """Return year-anchored evidence links for one risk factor."""
    try:
        engine = get_graph_engine()
        rows = engine.path_finder.find_temporal_evolution(risk_id)
        return rows[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """System health check."""
    neo4j_ok = False
    try:
        mgr = get_schema_manager()
        neo4j_ok = mgr.driver.verify_connectivity() is not None
    except Exception:
        pass

    return {
        "status": "healthy" if neo4j_ok else "degraded",
        "neo4j": "connected" if neo4j_ok else "disconnected",
    }


# ── Run ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
