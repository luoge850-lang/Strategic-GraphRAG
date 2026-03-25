"""
Module: step2_graph_ingestion.py
Project: Strategic GraphRAG Analysis for SEC Filings (NUS Research)
-----------------------------------------------------------------------
Description:
    This module implements a high-throughput ingestion pipeline designed 
    to extract strategic causal ontologies from non-structured SEC 10-K 
    filings. It leverages Large Language Models (LLMs) for triplet 
    extraction and synchronizes them with a Neo4j Knowledge Graph.

Key Features:
    - Heuristic Administrative Noise Mitigation.
    - Idempotent Graph Mapping (MERGE operations).
    - Network Resilience for Bolt protocol stability.
    - Targeted High-Density Strategic Scanning (Item 1 & 1A).

Author: Louis Harrington
Date: 2026-03-24 (Optimized Version)
"""

import os
import re
import json
import time
import glob
import logging
import certifi
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterable

import pdfplumber 
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv
from neo4j import GraphDatabase, TrustCustomCAs
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# --- Network Resilience: Preventing Proxy Interference for Bolt Protocol ---
# Necessary for stable connectivity to Neo4j Aura Cloud instances
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# Professional Academic Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GraphIngestionPipeline")

class StrategicIngestionPipeline:
    """
    Orchestrates the transformation of raw SEC filings into a structured 
    Strategic Knowledge Graph.
    """

    def __init__(self) -> None:
        """Initializes AI inference clients, DB drivers, and semantic splitters."""
        # Neural Inference Layer (Llama-3.1-8B optimized for strategic inference)
        self.ai_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.1-8b-instant" 
        
        # Persistence Layer (Neo4j Aura Cloud Connectivity)
        self.db_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            encrypted=True,
            trusted_certificates=TrustCustomCAs(certifi.where())
        )
        
        # Granular Semantic Splitting (Optimized for complex causal chains)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        logger.info("Ingestion Subsystem Online: Heuristic noise shield active.")

    def _extract_strategic_logic(self, text_chunk: str) -> Dict[str, Any]:
        """
        Invokes LLM agent to distill unstructured text into semantic triplets.
        
        Args:
            text_chunk (str): Raw text segment from PDF.
            
        Returns:
            Dict[str, Any]: Extracted triplets in structured JSON format.
        """
        prompt = f"""
        You are a Top-tier Hedge Fund Strategy Analyst. Extract only high-value STRATEGIC relationships.
        
        [Constraints]:
        - IGNORE administrative boilerplate (addresses, legal filings, Delaware codes).
        - FOCUS on NVIDIA's technology (Blackwell, CUDA), supply chain (TSMC), and market risks.
        
        [Output]: Return ONLY a valid JSON object.
        {{ "triples": [{{ "source": "ENTITY", "relation": "ACTION", "target": "ENTITY", "description": "Analysis" }}] }}

        Target Text:
        {text_chunk}
        """
        try:
            response = self.ai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0, 
                response_format={"type": "json_object"} 
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.debug(f"LLM Extraction Error: {e}")
            return {"triples": []}

    def _validate_entity(self, entity_id: str) -> bool:
        """
        Heuristically prunes administrative fragments to maintain high KG density.
        
        Args:
            entity_id (str): Candidate entity name for validation.
            
        Returns:
            bool: True if the entity passes semantic quality checks.
        """
        token = str(entity_id).strip().upper()
        if len(token) < 2 or len(token) > 55: return False
        
        # Reject postal codes, dates, and long numeric noise
        if re.search(r'[\d-]{5,}', token): return False 
        
        # SEC-Specific Administrative Noise Lexicon
        NOISE_TOKENS = [
            "NASDAQ", "STOCK", "ACT OF", "REGISTRANT", "SECTION", "FISCAL YEAR", 
            "DELAWARE", "SANTA CLARA", "CALIFORNIA", "95051", "TELEPHONE", 
            "FORM 10-K", "SECURITIES", "COMMISSION", "EXCHANGE", "PURSUANT"
        ]
        if any(noise in token for noise in NOISE_TOKENS): return False
        
        # Reject raw financial figures or currency symbols
        if re.match(r'^[\d.,$%-]+$', token): return False
        
        return True

    def _ingest_to_graph(self, triples: List[Dict], filename: str, page: int) -> int:
        """
        Serializes semantic triplets into Neo4j using idempotent MERGE operations.
        
        Args:
            triples (List[Dict]): Extracted relationships.
            filename (str): Source document metadata.
            page (int): Source page metadata for provenance.
            
        Returns:
            int: Number of successfully ingested relations.
        """
        if not triples: return 0
        
        success_count = 0
        with self.db_driver.session() as session:
            for item in triples:
                s_raw = item.get("source")
                t_raw = item.get("target")
                
                if not s_raw or not t_raw or not self._validate_entity(s_raw) or not self._validate_entity(t_raw):
                    continue
                
                # Entity Normalization (Entity Resolution for 'NVIDIA Corporation')
                s = s_raw.replace("NVIDIA CORPORATION", "NVIDIA").upper()
                t = t_raw.replace("NVIDIA CORPORATION", "NVIDIA").upper()
                
                # Cypher Relationship Type Sanitization
                rel_type = str(item.get('relation')).upper().replace(' ', '_')
                
                # Idempotent Cypher Query
                query = f"""
                MERGE (source:Entity {{id: $s}})
                MERGE (target:Entity {{id: $t}})
                MERGE (source)-[rel:`{rel_type}`]->(target)
                SET rel.description = $desc, 
                    rel.source = $filename, 
                    rel.page = $page
                """
                session.run(
                    query, 
                    s=s, t=t, 
                    desc=item.get("description"), 
                    filename=filename, 
                    page=str(page + 1)
                )
                success_count += 1
        return success_count

    def process_batch(self, input_folder: str = "data/pdfs") -> None:
        """
        Orchestrates batch processing with targeted strategic scanning.
        
        Args:
            input_folder (str): Directory containing SEC PDF filings.
        """
        # Heuristic: Targeting Item 1 (Business) and 1A (Risk Factors)
        CORE_STRATEGIC_PAGES = range(5, 35) 
        pdf_paths = glob.glob(f"{input_folder}/*.pdf")
        
        if not pdf_paths:
            logger.warning(f"No source PDFs found in {input_folder}")
            return

        for path in pdf_paths:
            fname = os.path.basename(path)
            logger.info(f"Scanning High-Density Strategic Regions: {fname}")
            
            try:
                with pdfplumber.open(path) as pdf:
                    # TQDM provides visual feedback for large document processing
                    for page_idx in tqdm(CORE_STRATEGIC_PAGES, desc=f"Ingesting {fname}"):
                        if page_idx >= len(pdf.pages): break
                        
                        page_content = pdf.pages[page_idx].extract_text()
                        if not page_content: continue
                        
                        chunks = self.text_splitter.split_text(page_content)
                        for chunk in chunks:
                            data = self._extract_strategic_logic(chunk)
                            count = self._ingest_to_graph(
                                data.get("triples", []), 
                                fname, 
                                page_idx
                            )
                            
                            if count > 0:
                                logger.debug(f"Captured {count} relations on Page {page_idx+1}")
                            
                            # API Rate Limit Compliance Protection
                            time.sleep(1.2)
            except Exception as e:
                logger.error(f"Processing Failure for {fname}: {e}")

    def close(self) -> None:
        """Terminates the database driver connection."""
        self.db_driver.close()

if __name__ == "__main__":
    pipeline = StrategicIngestionPipeline()
    try:
        pipeline.process_batch()
        logger.info("--- Graph Ingestion Pipeline Completed Successfully ---")
    except Exception as e:
        logger.error(f"Critical Pipeline Interruption: {e}")
    finally:
        pipeline.close()