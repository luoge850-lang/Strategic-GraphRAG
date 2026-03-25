"""
Module: step4_graphrag_query_engine.py
Project: Strategic GraphRAG Analysis for SEC Filings (NUS Research)
-----------------------------------------------------------------------
Description:
    Advanced Topological Retrieval Engine designed for high-stakes financial 
    reasoning. This engine integrates a Neuro-symbolic approach by combining 
    Neo4j Graph traversal (shortestPath) with a Cross-Encoder reranking model 
    to mitigate semantic noise and resolve multi-hop causal dependencies.

Key Features:
    - Agentic Keyword Extraction (Llama-3.3-70B).
    - Topological Multi-hop Traversal (ShortestPath up to 3 hops).
    - Semantic Reranking (ms-marco-MiniLM) for context pruning.
    - Grounded Answer Synthesis with strategic inference markers.

Author: Louis Harrington
Date: 2026-03-24 (Optimized Version)
"""

import os
import time
import logging
import warnings
import certifi
from typing import List, Tuple, Dict, Optional

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase, TrustCustomCAs
from sentence_transformers import CrossEncoder

# --- Configuration & Environment Management ---
warnings.filterwarnings("ignore")
load_dotenv()

# Network Resilience: Preventing Proxy Interference
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# Professional Academic Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GraphRAGQueryEngine")


class GraphRAGQueryEngine:
    """
    Orchestrates topological knowledge retrieval and grounded LLM synthesis.
    """

    def __init__(self) -> None:
        """Initializes AI clients, DB drivers, and the neural reranking layer."""
        self.ai_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile" 
        self.db_driver: Optional[GraphDatabase.driver] = None
        
        # [Architecture Update]: Reranking model to prune semantic noise
        logger.info("Loading Neural Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.connect_db()

    def connect_db(self) -> None:
        """Establishes a secure connection to the Neo4j Knowledge Graph."""
        try:
            self.db_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
                encrypted=True,
                trusted_certificates=TrustCustomCAs(certifi.where()),
                max_connection_lifetime=200
            )
            logger.info("GraphRAG Subsystem: Secure connection established with Neo4j.")
        except Exception as e:
            logger.error(f"Graph Database Connectivity Failure: {e}")

    def extract_keywords(self, query: str) -> List[str]:
        """
        Extracts strategic entities from the query to serve as graph entry points.

        Args:
            query (str): Natural language strategic question.

        Returns:
            List[str]: A list of normalized, uppercase strategic entities.
        """
        prompt = f"Extract 2-4 core entities from: {query}. Return ONLY comma-separated uppercase list."
        try:
            response = self.ai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name, 
                temperature=0.0
            )
            entities = [k.strip().upper() for k in response.choices[0].message.content.split(',')]
            logger.debug(f"Extracted Entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"Keyword Extraction Failed: {e}")
            return []

    def fetch_subgraph(self, query: str, keywords: List[str]) -> str:
        """
        Traverses the graph topology using shortestPath logic and neural reranking.
        
        Args:
            query (str): The original user query for relevance scoring.
            keywords (List[str]): Extracted entry nodes.

        Returns:
            str: A reranked context string of high-precision triplets.
        """
        if not keywords: return ""
        raw_triplets: List[str] = []
        
        with self.db_driver.session() as session:
            # 1. Topological Traversal: Shortest Path to bridge causal gaps
            if len(keywords) >= 2:
                path_cypher = """
                MATCH (e1:Entity), (e2:Entity)
                WHERE toUpper(e1.id) CONTAINS $k1 AND toUpper(e2.id) CONTAINS $k2 AND e1 <> e2
                MATCH p = shortestPath((e1)-[*..3]-(e2))
                RETURN p
                """
                logger.info(f"Traversing topological paths between {keywords[0]} and {keywords[1]}...")
                res = session.run(path_cypher, k1=keywords[0], k2=keywords[1])
                for record in res:
                    for rel in record['p'].relationships:
                        raw_triplets.append(f"[{rel.start_node['id']}] -{rel.type}-> [{rel.end_node['id']}]")

            # 2. Neighborhood Retrieval: Capturing immediate strategic context
            for kw in keywords:
                neighbor_cypher = """
                MATCH (n:Entity)-[r]-(m)
                WHERE toUpper(n.id) CONTAINS $kw
                RETURN n.id AS s, type(r) AS rel, m.id AS t LIMIT 10
                """
                res = session.run(neighbor_cypher, kw=kw)
                for rec in res:
                    raw_triplets.append(f"[{rec['s']}] -{rec['rel']}-> [{rec['t']}]")

        # --- [Neuro-Symbolic Filtering: Reranking for Precision] ---
        unique_triplets = list(set(raw_triplets))
        if not unique_triplets: return ""
        
        logger.info(f"Retrieved {len(unique_triplets)} candidate triplets. Initializing reranking...")
        pairs = [[query, t] for t in unique_triplets]
        scores = self.reranker.predict(pairs)
        
        # Rank by neural relevance and prune low-signal context
        ranked = sorted(zip(scores, unique_triplets), reverse=True, key=lambda x: x[0])
        final_context = "\n".join([t for s, t in ranked[:12]])
        
        return final_context

    def ask(self, query: str) -> Tuple[str, List[str]]:
        """
        Synthesizes a grounded answer using the topological context and an LLM Judge.

        Args:
            query (str): The strategic question.

        Returns:
            Tuple[str, List[str]]: The reasoned response and supporting evidence paths.
        """
        keywords = self.extract_keywords(query)
        graph_context = self.fetch_subgraph(query, keywords)
        
        if not graph_context.strip(): 
            return "Insufficient knowledge in graph to conclude.", []
            
        # [Academic Prompt Enhancement]: Strict citation and logical inference labeling
        prompt = f"""
        You are a Strategic Analyst. Answer the question based ONLY on the triplets provided.
        
        [Strict Rules]:
        1. For every claim, cite the triplet in brackets, e.g., ([NVIDIA] -SUPPLIES-> [TSMC]).
        2. If you need to connect two facts, label it [Strategic Inference].
        3. Respond in formal academic English.

        [Knowledge Graph Context]:
        {graph_context}
        
        [Question]: {query}
        """
        
        logger.info("Synthesizing strategic response with Llama-3.3-70B...")
        try:
            response = self.ai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name, 
                temperature=0.1
            )
            return response.choices[0].message.content, graph_context.split('\n')
        except Exception as e:
            logger.error(f"Synthesis Failure: {e}")
            return f"Error in synthesis: {e}", []

    def close(self) -> None:
        """Terminates the Neo4j driver session."""
        if self.db_driver: 
            self.db_driver.close()
            logger.info("Graph Database Driver disconnected.")

if __name__ == "__main__":
    # Integration Test for GraphRAG Engine
    engine = GraphRAGQueryEngine()
    test_query = "How do supply chain risks in Taiwan impact NVIDIA's data center revenue?"
    ans, paths = engine.ask(test_query)
    print(f"\n[Strategic Analysis]:\n{ans}\n")
    engine.close()