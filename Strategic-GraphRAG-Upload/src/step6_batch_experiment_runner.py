"""
Module: step6_batch_experiment_runner.py
Project: Strategic GraphRAG Analysis for SEC Filings (NUS Research)
-----------------------------------------------------------------------
Description:
    This module serves as the central orchestration engine for the 
    comparative ablation study. It systematically executes queries against 
    two competing architectures:
    
    1. Baseline (Vector-based RAG)
    2. Proposed (Graph-based RAG)
    
    Each response-context pair is then audited by an autonomous academic 
    judge to quantify the performance gap in strategic intelligence.

Key Features:
    - Checkpoint Persistence: Resumes execution from the last saved state.
    - Fault-tolerant Ingestion: Real-time JSON dumping to prevent data loss.
    - API Rate-Limit Protection: Heuristic sleep intervals for stable inference.

Author: Louis Harrington
Date: 2026-03-24 (Optimized Version)
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# Internal Module Imports (Renamed for Step Consistency)
from step4_graphrag_query_engine import GraphRAGQueryEngine
from step3_vector_rag_engine import BaselineRAG
from step5_academic_evaluator import AcademicRAGEvaluator

# ==========================================
# 1. Path Configuration & Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ExperimentRunner")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_EVAL_DIR = ROOT_DIR / "data" / "evaluation"
INPUT_DATASET = DATA_EVAL_DIR / "golden_dataset.json"
OUTPUT_RESULTS = DATA_EVAL_DIR / "ablation_results_final.json"


# ==========================================
# 2. Core Logic: ExperimentRunner
# ==========================================
class ExperimentRunner:
    """
    Manages the lifecycle of the ablation study, ensuring symmetric 
    testing across the vector and graph retrieval pipelines.
    """

    def __init__(self) -> None:
        """Initializes retrieval engines and the evaluation judge."""
        logger.info("Initializing Experiment Subsystems: Loading Neural Retrieval Engines...")
        
        # Initializing Baseline Engine (Vector Space)
        self.baseline_engine = BaselineRAG(db_path=str(ROOT_DIR / "data" / "chroma_db"))
        
        # Initializing Proposed Engine (Topological Space)
        self.graph_engine = GraphRAGQueryEngine()
        
        # Initializing Academic Judge (LLM-as-a-Judge)
        self.judge = AcademicRAGEvaluator()

    def run_ablation_study(self) -> None:
        """
        Executes the batch evaluation loop with integrated checkpoint management 
        and real-time results persistence.
        """
        if not INPUT_DATASET.exists():
            logger.critical(f"Input Dataset not found at {INPUT_DATASET}. Execution aborted.")
            return

        with open(INPUT_DATASET, "r", encoding="utf-8") as f:
            dataset: List[Dict[str, Any]] = json.load(f)

        results: List[Dict[str, Any]] = []
        
        # --- [Checkpoint Persistence Logic] ---
        # Resumes from previous run if the results file already exists
        if OUTPUT_RESULTS.exists():
            logger.info(f"Existing results detected. Synchronizing state from {OUTPUT_RESULTS.name}...")
            with open(OUTPUT_RESULTS, "r", encoding="utf-8") as f:
                results = json.load(f)

        processed_ids = [r["id"] for r in results]
        logger.info(f"Target Queue: {len(dataset)} items. Completed: {len(processed_ids)}.")

        for idx, item in enumerate(dataset, 1):
            if item["id"] in processed_ids: 
                continue # Skip already processed cases to save tokens

            logger.info(f"--- [Experiment {idx}/{len(dataset)}] Processing Case ID: {item['id']} ---")
            
            # 1. Baseline Evaluation (Vector RAG)
            logger.info(f"[{item['id']}] Executing Vector Baseline Retrieval...")
            base_ans, base_docs = self.baseline_engine.ask(item["question"])
            base_context = "\n".join(base_docs)
            base_scores = self.judge.evaluate_response(
                item["question"], 
                item["expected_strategic_intent"], 
                base_context, 
                base_ans
            )
            
            # API Safety Interval (Protecting Rate Limits)
            time.sleep(2.0)

            # 2. Proposed Evaluation (Strategic GraphRAG)
            logger.info(f"[{item['id']}] Executing Topological Graph Retrieval...")
            graph_ans, graph_docs = self.graph_engine.ask(item["question"])
            graph_context = "\n".join(graph_docs)
            graph_scores = self.judge.evaluate_response(
                item["question"], 
                item["expected_strategic_intent"], 
                graph_context, 
                graph_ans
            )

            # 3. Data Integration
            results.append({
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "baseline_vector_rag": {"answer": base_ans, "scores": base_scores},
                "proposed_graph_rag": {"answer": graph_ans, "scores": graph_scores}
            })
            
            # 4. Real-time Persistence (I/O Flush)
            # Ensures results are written to disk after every test case to prevent data loss
            with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            logger.info(f"[{item['id']}] Metrics successfully recorded and persisted.")
            time.sleep(2.0)

        # Teardown
        self.graph_engine.close()
        logger.info("=== Optimized Ablation Study Completed: All metrics synchronized ===")

if __name__ == "__main__":
    # Starting the Experiment Orchestrator
    runner = ExperimentRunner()
    try:
        runner.run_ablation_study()
    except KeyboardInterrupt:
        logger.warning("Experiment manually interrupted. Current progress has been saved.")
    except Exception as e:
        logger.error(f"Critical Experiment Failure: {e}")