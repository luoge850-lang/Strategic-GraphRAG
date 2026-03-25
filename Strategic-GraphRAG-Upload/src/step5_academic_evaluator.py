"""
Module: step5_academic_evaluator.py
Project: Strategic GraphRAG Analysis for SEC Filings (NUS Research)
-----------------------------------------------------------------------
Description:
    This module implements an automated "LLM-as-a-Judge" evaluation framework. 
    It quantifies the performance of RAG systems by analyzing the alignment 
    between the retrieved context, the generated answer, and the expected 
    strategic intent.

Metrics Evaluated:
    - Faithfulness: Factual consistency with the retrieved context.
    - Answer Relevance: Directness and alignment with the user's strategic query.
    - Context Precision: Signal-to-noise ratio of the retrieved information.

Author: Louis Harrington
Date: 2026-03-24 (Optimized Version)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import certifi
from dotenv import load_dotenv
from groq import Groq

# ==========================================
# 1. System Initialization & Path Management
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AcademicEvaluator")

# Robust Relative Pathing for Cross-Platform Deployment
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_EVAL_DIR = ROOT_DIR / "data" / "evaluation"
ENV_PATH = ROOT_DIR / ".env"

# Load environment configuration
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    logger.warning("Environmental configuration (.env) not found in root directory.")

class AcademicRAGEvaluator:
    """
    V2 Academic Evaluator leveraging the LLM-as-a-Judge paradigm for 
    objective RAG performance quantification.
    """
    
    def __init__(self) -> None:
        """
        Initializes the Groq client and selects the flagship model for grading.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.critical("GROQ_API_KEY is missing. Evaluation subsystem will fail.")
            raise ValueError("CRITICAL: GROQ_API_KEY missing from environment variables.")
        
        self.judge_client = Groq(api_key=api_key)
        # Using the flagship 70B model to ensure high-reasoning academic grading
        self.judge_model = "llama-3.3-70b-versatile"
        logger.info(f"Evaluator initialized using model: {self.judge_model}")

    def evaluate_response(
        self, 
        question: str, 
        expected_intent: str, 
        context: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        Passes the RAG trace (Q-C-A) to the LLM Judge for multi-dimensional scoring.

        Args:
            question (str): The original strategic query.
            expected_intent (str): The desired strategic focus of the answer.
            context (str): The retrieved evidence (text chunks or triplets).
            answer (str): The final generated response from the RAG engine.

        Returns:
            Dict[str, Any]: A dictionary containing normalized scores and academic justification.
        """
        judge_prompt = f"""
        You are an impartial, rigorous academic evaluator assessing a GraphRAG system for a university research paper.
        Evaluate the AI's response based on three strict metrics. 
        Score each from 1 to 5 (5 being perfect).

        [Metrics Definition]
        1. Faithfulness (忠实度): Is the Answer strictly derived from the Context? Did it avoid hallucinating external facts?
        2. Answer Relevance (相关性): Does the Answer directly address the Question and align with the Expected Strategic Intent?
        3. Context Precision (上下文精准度): How relevant is the provided Context? Is there a lot of useless semantic noise?

        [Trace Data]
        Question: {question}
        Expected Intent: {expected_intent}
        Retrieved Context (Evidence): {context}
        AI Answer: {answer}

        Output STRICTLY in the following JSON format without Markdown blocks:
        {{
            "faithfulness_score": <int>,
            "relevance_score": <int>,
            "context_precision_score": <int>,
            "academic_justification": "<A brief, highly professional 2-sentence explanation for the scores>"
        }}
        """

        try:
            # Deterministic evaluation using temperature=0.0
            response = self.judge_client.chat.completions.create(
                messages=[{"role": "user", "content": judge_prompt}],
                model=self.judge_model,
                temperature=0.0, 
                response_format={"type": "json_object"}
            )
            
            result_json = response.choices[0].message.content
            if result_json:
                return json.loads(result_json)
            return {"error": "Empty response from academic judge."}
            
        except Exception as e:
            logger.error(f"Evaluation Failure for query: {question[:30]}... Error: {e}")
            return {
                "faithfulness_score": 0,
                "relevance_score": 0,
                "context_precision_score": 0,
                "academic_justification": f"Evaluation Subsystem Error: {str(e)}"
            }

# ==========================================
# 2. Standalone Validation Block
# ==========================================
if __name__ == "__main__":
    # Integration Test for the Evaluator
    logger.info("Starting Evaluator Sanity Check...")
    evaluator = AcademicRAGEvaluator()
    
    # Mock data simulating a typical GraphRAG retrieval trace
    mock_q = "What specific threats do competitors like AMD pose to NVIDIA?"
    mock_intent = "Aggregate competitive threats including AMD's alternative GPUs."
    mock_ctx = (
        "[AMD] - SUPPLIES -> [GPUS] (Context: Alternative to NVIDIA)\n"
        "[TRAINING COURSES] - PROVIDE -> [EMPLOYEES]"
    )
    mock_ans = (
        "AMD poses a threat by supplying alternative GPUs (Citation: [AMD] - SUPPLIES -> [GPUS]). "
        "However, the graph also mentions training courses, which is irrelevant to the threat."
    )
    
    logger.info("Executing test evaluation on cross-domain mock data...")
    result = evaluator.evaluate_response(mock_q, mock_intent, mock_ctx, mock_ans)
    
    print("\n" + "="*60)
    print("🎓 ACADEMIC JUDGE EVALUATION RESULT (JSON):")
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print("="*60 + "\n")
    logger.info("Evaluation module validation complete.")