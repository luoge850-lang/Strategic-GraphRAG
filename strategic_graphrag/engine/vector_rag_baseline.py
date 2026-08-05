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
from typing import List, Tuple, Optional

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
        collection_name: str = "nvidia_sec_filings",
        embedding_model: str = "all-MiniLM-L6-v2",
        model_name: str = "llama-3.3-70b-versatile",
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name

        # ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = None
        self._init_collection()

        # LLM via unified provider
        from ..llm_provider import get_llm
        self.llm = get_llm()
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
        if not self.collection or self.collection.count() == 0:
            return []
        try:
            results = self.collection.query(query_texts=[query], n_results=k)
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

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

        result = self.llm.chat(prompt=prompt, temperature=0.1, max_tokens=1000)
        if result is None:
            return f"[Generation error: LLM call failed]"
        return result

    def ask(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """Full RAG pipeline: retrieve + generate."""
        docs = self.retrieve(query, k)
        answer = self.generate(query, docs)
        return answer, docs
