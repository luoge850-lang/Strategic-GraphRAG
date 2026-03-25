"""
Module: step3_vector_rag_engine.py
Project: Strategic GraphRAG Analysis for SEC Filings (NUS Research)
-----------------------------------------------------------------------
Description:
    This module implements a standard Vector-based Retrieval-Augmented 
    Generation (RAG) engine. It serves as the experimental control group 
    (Baseline) for the ablation study, providing a performance benchmark 
    against the proposed topological GraphRAG architecture.

Key Features:
    - Semantic Similarity Search via ChromaDB.
    - Context-Injected Generation using Llama-3.3-70B.
    - Formal Academic English Response Synthesis.

Author: Louis Harrington
Date: 2026-03-24 (Optimized Version)
"""

import os
import logging
from typing import List, Tuple, Optional

import chromadb
from groq import Groq
from dotenv import load_dotenv

# Load environment variables for API access
load_dotenv()

# --- Network Resilience: Preventing Proxy Interference ---
# Essential for stable connectivity to external API endpoints
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# Professional Academic Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("BaselineRAGEngine")

class BaselineRAG:
    """
    Standard Vector-based Retrieval-Augmented Generation (RAG) Engine.
    Acts as the scientific control group (Baseline) for academic evaluation.
    """

    def __init__(self, db_path: str = "data/chroma_db") -> None:
        """
        Initializes the inference client and retrieval parameters.

        Args:
            db_path (str): Path to the persistent ChromaDB directory.
        """
        self.db_path = db_path
        self.ai_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Unified to the flagship 70B model for high-fidelity strategic synthesis
        self.model_name = "llama-3.3-70b-versatile" 
        self.collection = None
        
        # Immediate initialization of DB connection
        self.build_vector_db()

    def build_vector_db(self) -> None:
        """
        Initializes the connection to the local ChromaDB vector store.
        Loads the pre-processed semantic collection.
        """
        logger.info("Initializing Baseline Vector RAG Subsystem...")
        client = chromadb.PersistentClient(path=self.db_path)
        try:
            self.collection = client.get_collection(name="nvidia_sec_filings")
            logger.info("Vector collection 'nvidia_sec_filings' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vector collection: {e}. Verify ingestion in Step 1.")

    def retrieve(self, query: str, n_results: int = 5) -> List[str]:
        """
        Retrieves top-K text chunks based on semantic similarity.

        Args:
            query (str): The natural language query.
            n_results (int): Number of chunks to retrieve.

        Returns:
            List[str]: A list of retrieved text documents.
        """
        if not self.collection:
            logger.warning("Retrieval attempted on uninitialized collection.")
            return []
        
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return []

    def ask(self, query: str) -> Tuple[str, List[str]]:
        """
        Generates a strategic answer based on vector-retrieved context.

        Args:
            query (str): The strategic question for the baseline model.

        Returns:
            Tuple[str, List[str]]: The generated response and the supporting context chunks.
        """
        docs = self.retrieve(query)
        if not docs:
            return "Cannot conclude based on the given documents.", []
            
        # Context Injection with explicit strategic markers
        context = "\n---\n".join(docs)
        prompt = f"""
        You are a Strategic Analyst. Answer the question based ONLY on the provided context.
        Respond entirely in formal academic English.
        
        [Context]:
        {context}
        
        [Question]:
        {query}
        """
        
        logger.info(f"Synthesizing baseline response for query: {query[:50]}...")
        try:
            response = self.ai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.1  # Low temperature for deterministic analysis
            )
            return response.choices[0].message.content, docs
        except Exception as e:
            logger.error(f"Inference Failure: {e}")
            return f"[Error in Baseline Generation]: {str(e)}", []

if __name__ == "__main__":
    # Integration test for Baseline Engine
    engine = BaselineRAG()
    sample_query = "What are the primary strategic risks mentioned in the 10-K?"
    answer, source_docs = engine.ask(sample_query)
    print(f"\n[Generated Answer]:\n{answer}\n")