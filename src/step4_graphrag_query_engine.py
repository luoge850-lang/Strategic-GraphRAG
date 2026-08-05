"""
Module: step4_graphrag_query_engine.py
Project: Strategic GraphRAG Analysis for SEC Filings
Description: Academic-Grade Engine with Logit Calibration, Ontology Mapping, and Structured Grounding.
"""

import os
import numpy as np
import logging
import warnings
import certifi
from typing import List, Tuple

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase, TrustCustomCAs
from sentence_transformers import CrossEncoder

warnings.filterwarnings("ignore")
load_dotenv()
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("GraphRAG_Production")

class ProductionGraphRAG:
    def __init__(self):
        self.ai_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile" 
        
        self.db_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            encrypted=True,
            trusted_certificates=TrustCustomCAs(certifi.where())
        )
        
        logger.info("Loading Cross-Encoder Reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

    def _extract_search_anchors(self, query: str) -> list:
        prompt = f"""
        Extract 2-4 critical macro-economic or financial entity keywords from the query.
        Return ONLY a comma-separated list of UPPERCASE strings. Query: "{query}"
        """
        try:
            res = self.ai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], model=self.model_name, temperature=0.0
            )
            return [ent.strip().upper() for ent in res.choices[0].message.content.strip().split(',')]
        except Exception: return []

    def _structure_mechanism(self, text: str) -> str:
        """[OPTIMIZATION]: Dynamically structure text into actionable polarities for the LLM."""
        text_lower = text.lower()
        if "no material impact" in text_lower or "not had a material impact" in text_lower or "no significant" in text_lower:
            magnitude = "Immaterial"
            polarity = "Neutral"
        elif "adversely" in text_lower or "risk" in text_lower or "reduce" in text_lower:
            magnitude = "Material_Risk"
            polarity = "Negative"
        else:
            magnitude = "Contextual"
            polarity = "Unknown"
        return f"{{Polarity: '{polarity}', Magnitude: '{magnitude}', RawText: '{text}'}}"

    def _retrieve_and_rerank_subgraph(self, anchors: list, user_query: str, top_k: int = 10) -> Tuple[str, List[str]]:
        if not anchors: return "", []
        
        candidate_paths = []
        with self.db_driver.session() as session:
            # [OPTIMIZATION]: Ontology Mapping directly in Cypher
            query = """
            UNWIND $anchors AS ent
            MATCH (anchor:EntityV3)
            WHERE anchor.id CONTAINS ent OR ent CONTAINS anchor.id
            
            MATCH path = (anchor)-[*1..2]-(neighbor:EntityV3)
            UNWIND relationships(path) AS r
            WITH DISTINCT r
            WHERE startNode(r).id <> 'BUSINESS' AND endNode(r).id <> 'BUSINESS'
            
            // Map wild edges to Canonical Financial Ontology
            WITH r, startNode(r).id AS src, endNode(r).id AS tgt, r.description AS desc, r.source AS file, r.page AS page,
            CASE 
                WHEN type(r) IN ['POSES_RISK_TO', 'REDUCES', 'RESTRICTS', 'THREATENS'] THEN 'RISK_EXPOSURE'
                WHEN type(r) IN ['IMPACTS', 'AFFECTS', 'DRIVES'] THEN 'CAUSAL_INFLUENCE'
                WHEN type(r) IN ['INTEGRATES', 'MITIGATES', 'PAID'] THEN 'STRATEGIC_ACTION'
                ELSE 'ASSOCIATION' 
            END AS canonical_rel
            
            RETURN src, canonical_rel, type(r) AS raw_rel, tgt, desc, file, page
            LIMIT 150
            """
            results = session.run(query, anchors=anchors)
            for rec in results:
                path_sig = f"(Node: {rec['src']}) -[Edge: {rec['canonical_rel']}]-> (Node: {rec['tgt']})"
                structured_mech = self._structure_mechanism(rec['desc'])
                full_path = f"{path_sig} | Evidence: {structured_mech} | Source: {rec['file']}, P{rec['page']}"
                candidate_paths.append(full_path)

        if not candidate_paths: return "", []
        candidate_paths = list(set(candidate_paths))

        # Semantic Reranking
        pairs = [[user_query, path] for path in candidate_paths]
        raw_logits = self.reranker.predict(pairs)
        
        # [OPTIMIZATION]: Sigmoid Calibration (Raw Logits -> Probabilities 0 to 1)
        probabilities = 1 / (1 + np.exp(-raw_logits))
        scored_paths = sorted(zip(candidate_paths, probabilities), key=lambda x: x[1], reverse=True)
        
        audit_trace_lines = []
        for i, (path, prob) in enumerate(scored_paths[:top_k]):
            if prob < 0.10: continue # Hard Cutoff: Drop garbage logic
            audit_trace_lines.append(f"Rank {i+1} (Confidence: {prob*100:.1f}%): {path}")
            
        return "\n".join(audit_trace_lines), audit_trace_lines

    def ask(self, user_query: str) -> str:
        anchors = self._extract_search_anchors(user_query)
        graph_context, audit_trace = self._retrieve_and_rerank_subgraph(anchors, user_query)

        if not audit_trace:
            return "[Error]: No high-confidence topological paths found."

        synthesis_prompt = f"""
        You are a deterministic Financial Reasoning Agent. Synthesize an analysis based EXACTLY on the Provided Structured Graph Trace.

        [STRICT PRODUCTION CONSTRAINTS]:
        1. SCHEMA COMPLIANCE: Refer to relations using their canonical ontology (e.g., RISK_EXPOSURE).
        2. STRUCTURED EVIDENCE PARSING: 
           - Look at the `Evidence: {{Polarity, Magnitude}}` dictionary.
           - If Magnitude is "Immaterial", you MUST explicitly state the relationship is currently disclosed as non-material or theoretical, with no financial impact.
        3. AVOID INTERPRETIVE JARGON: Do not use terms like "boilerplate" unless it is in the RawText. Maintain a neutral reporting tone.
        4. GROUNDING: Inline citations must include the exact Node-Edge-Node path.

        [Provided Structured Graph Trace]:
        {graph_context}
        
        [User Query]: {user_query}
        """
        try:
            response = self.ai_client.chat.completions.create(
                messages=[{"role": "user", "content": synthesis_prompt}], model=self.model_name, temperature=0.0  
            )
            return f"========== [CALIBRATED SUBGRAPH TRACE] ==========\n{chr(10).join(audit_trace)}\n\n========== [PRODUCTION SYNTHESIS] ==========\n{response.choices[0].message.content}"
        except Exception as e:
            return f"System Error: {e}"

if __name__ == "__main__":
    engine = ProductionGraphRAG()
    test_query = "Explain the ripple effect: How does climate change and global sustainability regulations collectively impact NVIDIA's operational results?"
    print(engine.ask(test_query))