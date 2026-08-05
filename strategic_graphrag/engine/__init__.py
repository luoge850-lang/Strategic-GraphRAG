"""GraphRAG Reasoning Engine — causal multi-hop path search & inference."""

from .graph_rag_engine import GraphRAGEngine, CausalPathFinder, PathScorer, CausalPath

# Vector RAG is an optional comparison subsystem. GraphRAG and the Neo4j
# validation path must remain importable without the heavier Chroma stack.
try:
    from .vector_rag_baseline import VectorRAGBaseline
except ImportError:  # pragma: no cover - depends on local optional extras
    VectorRAGBaseline = None

__all__ = [
    "GraphRAGEngine",
    "CausalPathFinder",
    "PathScorer",
    "CausalPath",
    "VectorRAGBaseline",
]
