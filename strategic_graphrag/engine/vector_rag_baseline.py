# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Vector RAG Baseline Engine
==============================================
Standard vector-based RAG for comparative ablation studies.
Serves as the control group against which GraphRAG is evaluated.

Uses ChromaDB + Sentence-Transformers + LLM.
"""

import os
import logging
from typing import Any, Dict, List, Tuple, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("VectorRAG")


class VectorRAGBaseline:
    """
    Standard Vector RAG: Embed → Retrieve → LLM Generate.

    This is the ablation baseline. It has NO knowledge graph,
    NO causal reasoning, and NO evidence provenance chain.
    """

    def __init__(
        self,
        db_path: str = "data/chroma_db",
        collection_name: str = None,
        embedding_model: str = None,
        model_name: str = None,
    ):
        self.db_path = db_path
        self.collection_name = collection_name or os.getenv(
            "GRAPH_VECTOR_COLLECTION", "nvidia_sec_filings"
        )
        self.model_name = model_name
        self.embedding_model = embedding_model or os.getenv(
            "GRAPH_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.embedding_backend = os.getenv(
            "GRAPH_EMBEDDING_BACKEND", "chroma_onnx"
        ).strip().lower()

        # ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        if self.embedding_backend == "sentence_transformers":
            # Production inference uses the pinned local model cache and must
            # not add Hugging Face network retries to request latency.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
        elif self.embedding_backend == "chroma_onnx":
            # Chroma's ONNX MiniLM backend keeps Hybrid retrieval independent
            # from PyTorch/Sentence-Transformers at service startup.
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            raise ValueError(
                "GRAPH_EMBEDDING_BACKEND must be chroma_onnx or sentence_transformers"
            )
        self.collection = None
        self._init_collection()

        # LLM via unified provider
        from ..llm_provider import get_llm
        self.llm = get_llm()
        self.model_name = self.model_name or self.llm.get_task_model("report")
        self._has_llm = self.llm.available
        if not self._has_llm:
            logger.warning("No LLM provider — synthesis disabled")

    def _init_collection(self):
        """Load or create ChromaDB collection."""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name, embedding_function=self.embedding_fn
            )
            count = self.collection.count()
            logger.info(f"Vector collection loaded: {count} chunks")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name, embedding_function=self.embedding_fn
            )
            logger.warning(f"Created new empty collection: {self.collection_name}")

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve top-K chunks via semantic similarity."""
        result = self.retrieve_with_metadata(query, k=k)
        return [hit["document"] for hit in result["hits"]]

    def retrieve_with_metadata(
        self,
        query: str,
        k: int = 5,
        source_filing: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve semantic chunks with scope and ranking diagnostics.

        Hybrid retrieval must not silently use an unscoped legacy collection.
        When a filing is requested, the collection is filtered by its
        ``source_filing`` metadata field and failures are reported explicitly.
        """
        if not self.collection or self.collection.count() == 0:
            return {"status": "EMPTY", "hits": [], "collection": self.collection_name}

        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if source_filing:
            kwargs["where"] = {"source_filing": source_filing}

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            if source_filing:
                logger.warning(
                    "Scoped vector retrieval failed for collection %s and filing %s: %s",
                    self.collection_name,
                    source_filing,
                    e,
                )
                return {
                    "status": "ERROR",
                    "hits": [],
                    "collection": self.collection_name,
                    "source_filing": source_filing,
                    "error": type(e).__name__,
                }
            logger.error(f"Retrieval error: {e}")
            return {"status": "ERROR", "hits": [], "collection": self.collection_name}

        documents = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []
        hits = []
        for rank, document in enumerate(documents, start=1):
            metadata = metadatas[rank - 1] if rank - 1 < len(metadatas) else {}
            distance = distances[rank - 1] if rank - 1 < len(distances) else None
            hits.append({
                "rank": rank,
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
                "rank_score": round(1.0 / rank, 6),
            })

        return {
            "status": "OK" if hits else "NO_HITS",
            "hits": hits,
            "collection": self.collection_name,
            "source_filing": source_filing,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
        }

    def diagnostics(self) -> Dict[str, Any]:
        """Return machine-readable readiness without invoking the LLM."""
        count = self.collection.count() if self.collection is not None else 0
        return {
            "ready": count > 0,
            "collection": self.collection_name,
            "count": count,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
        }

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """Generate an answer from retrieved context chunks."""
        if not context_chunks:
            return "[No relevant documents found.]"

        context = "\n---\n".join(context_chunks)
        prompt = f"""You are a financial analyst. Answer based ONLY on the context below.

[Context]:
{context}

[Question]:
{query}

Provide a concise, factual answer. Do not make up information not in the context."""

        if not self._has_llm:
            return f"[LLM unavailable]\n\nRelevant excerpts:\n{context[:2000]}"

        result = self.llm.chat(
            prompt=prompt,
            model=self.model_name,
            temperature=0.1,
            max_tokens=1000,
        )
        if result is None:
            return f"[Generation error: LLM call failed]"
        return result

    def ask(
        self,
        query: str,
        k: int = 5,
        source_filing: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """Full RAG pipeline: retrieve + generate."""
        retrieval = self.retrieve_with_metadata(
            query,
            k=k,
            source_filing=source_filing,
        )
        docs = [hit["document"] for hit in retrieval["hits"]]
        answer = self.generate(query, docs)
        return answer, docs
