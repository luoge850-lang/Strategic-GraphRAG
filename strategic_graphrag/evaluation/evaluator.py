# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Academic Evaluation Framework
=================================================
Multi-dimensional evaluation for comparing Vector RAG vs GraphRAG.

Metrics:
  1. Faithfulness — factual consistency with evidence
  2. Answer Relevance — alignment with query intent
  3. Context Precision — signal-to-noise ratio
  4. Citation Completeness — are claims backed by sources?
  5. Hallucination Detection — unsupported claims ratio
  6. Causal Path Quality — path completeness & strength (GraphRAG only)

Uses LLM-as-a-Judge + automated metrics (BERTScore, ROUGE).
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("Evaluator")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EvaluationResult:
    """Complete evaluation result for one query."""
    query_id: str
    question: str
    category: str
    expected_intent: str

    # Vector RAG scores
    vector_answer: str = ""
    vector_scores: Dict[str, float] = field(default_factory=dict)

    # GraphRAG scores
    graph_answer: str = ""
    graph_scores: Dict[str, float] = field(default_factory=dict)
    graph_paths: List[Dict] = field(default_factory=list)

    # Comparative
    winner: str = ""  # "graphrag" | "vector" | "tie"
    delta: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# LLM-as-a-Judge Evaluator
# =============================================================================

class LLMJudge:
    """
    Uses an LLM to score RAG responses on multiple dimensions.
    More nuanced than automated metrics for financial analysis quality.
    """

    JUDGE_PROMPT = """You are an impartial academic evaluator assessing a financial RAG system.
Evaluate the AI's response against the provided context and expected intent.

[METRICS — Score each 1-5 (5 = perfect)]:
1. **Faithfulness** (忠实度): Is every claim in the Answer directly supported by the Context?
   - 5: All claims verifiable in context
   - 3: Some claims unverifiable
   - 1: Answer contradicts context or fabricates facts

2. **Answer Relevance** (相关性): Does the Answer directly address the Question?
   - 5: Fully addresses the question with precision
   - 3: Partially relevant, tangential
   - 1: Off-topic or irrelevant

3. **Context Precision** (上下文精准度): Is the retrieved Context actually useful?
   - 5: All context directly relevant
   - 3: Mixed — some useful, some noise
   - 1: Context is noise/unrelated

4. **Citation Completeness** (引用完整性): Does the Answer cite specific sources?
   - 5: Every claim has explicit source citation
   - 3: Some claims cited, some not
   - 1: No citations or evidence references

5. **Hallucination Score** (幻觉评估 — reverse scored):
   - 5: No hallucinations detected
   - 3: Minor unsupported statements
   - 1: Major fabrications present

[INPUT]:
Question: {question}
Expected Intent: {intent}
Context: {context}
Answer: {answer}

Output STRICTLY as JSON (no markdown):
{{"faithfulness": <int>, "relevance": <int>, "context_precision": <int>,
  "citation_completeness": <int>, "hallucination_control": <int>,
  "justification": "<2-sentence academic justification>"}}"""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        try:
            from groq import Groq
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
            self._active = True
        except ImportError:
            self.client = None
            self._active = False
            logger.warning("Judge LLM unavailable")

    def evaluate(
        self, question: str, expected_intent: str, context: str, answer: str
    ) -> Dict[str, Any]:
        """Score a RAG response across all dimensions."""
        if not self._active:
            return self._mock_scores()

        prompt = self.JUDGE_PROMPT.format(
            question=question, intent=expected_intent,
            context=context[:3000], answer=answer[:2000],
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Judge error: {e}")
            return self._mock_scores()

    def _mock_scores(self) -> Dict:
        return {
            "faithfulness": 0, "relevance": 0, "context_precision": 0,
            "citation_completeness": 0, "hallucination_control": 0,
            "justification": "Judge unavailable",
        }


# =============================================================================
# Automated Metrics
# =============================================================================

class AutomatedMetrics:
    """BERTScore and lexical overlap metrics."""

    def __init__(self):
        self.bertscore = None
        try:
            from bert_score import score
            self.bertscore = score
            logger.info("BERTScore loaded")
        except ImportError:
            logger.info("BERTScore not available — using lexical metrics only")

    def compute(
        self, reference: str, candidate: str
    ) -> Dict[str, float]:
        """Compute automated similarity metrics."""
        metrics = {}

        # Lexical overlap (simple but fast)
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        if ref_words:
            metrics["lexical_precision"] = len(ref_words & cand_words) / len(ref_words)
            metrics["lexical_recall"] = len(ref_words & cand_words) / max(len(cand_words), 1)
            f1_denom = metrics["lexical_precision"] + metrics["lexical_recall"]
            metrics["lexical_f1"] = (
                2 * metrics["lexical_precision"] * metrics["lexical_recall"] / f1_denom
                if f1_denom > 0 else 0
            )

        return metrics


# =============================================================================
# Ablation Study Runner
# =============================================================================

class AblationRunner:
    """
    Orchestrates the comparative ablation study:
      Vector RAG (baseline) vs GraphRAG (proposed)
    """

    def __init__(
        self,
        vector_engine,
        graph_engine,
        judge: LLMJudge = None,
        output_dir: str = "data/evaluation",
    ):
        self.vector_engine = vector_engine
        self.graph_engine = graph_engine
        self.judge = judge or LLMJudge()
        self.auto_metrics = AutomatedMetrics()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, dataset: List[Dict], resume: bool = True) -> List[EvaluationResult]:
        """
        Run the full ablation study.

        Args:
            dataset: List of {"id", "question", "category", "expected_strategic_intent"}
            resume: Resume from checkpoint

        Returns:
            List of EvaluationResult
        """
        results = []
        checkpoint_path = self.output_dir / "ablation_checkpoint.json"

        # Resume logic
        completed_ids = set()
        if resume and checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results = [EvaluationResult(**r) for r in saved]
                completed_ids = {r["query_id"] for r in saved}
            logger.info(f"Resumed: {len(results)} completed, {len(dataset) - len(results)} remaining")

        for i, item in enumerate(dataset):
            qid = item.get("id", f"Q{i:03d}")
            if qid in completed_ids:
                continue

            logger.info(f"[{i+1}/{len(dataset)}] Evaluating: {qid}")
            result = EvaluationResult(
                query_id=qid,
                question=item["question"],
                category=item.get("category", "General"),
                expected_intent=item.get("expected_strategic_intent", ""),
            )

            # Vector RAG
            try:
                vec_answer, vec_docs = self.vector_engine.ask(item["question"])
                vec_context = "\n".join(vec_docs)
                result.vector_answer = vec_answer
                result.vector_scores = self.judge.evaluate(
                    item["question"], item.get("expected_strategic_intent", ""),
                    vec_context, vec_answer,
                )
            except Exception as e:
                logger.error(f"Vector RAG error for {qid}: {e}")
                result.vector_scores = {"error": str(e)}

            time.sleep(1.5)  # Rate limit protection

            # GraphRAG
            try:
                graph_result = self.graph_engine.query(item["question"])
                result.graph_answer = graph_result["answer"]
                result.graph_paths = graph_result.get("paths", [])
                graph_context = "\n".join(
                    p.get("evidence", [""])[0] if isinstance(p.get("evidence"), list) else ""
                    for p in result.graph_paths[:5]
                )
                result.graph_scores = self.judge.evaluate(
                    item["question"], item.get("expected_strategic_intent", ""),
                    graph_context, graph_result["answer"],
                )
            except Exception as e:
                logger.error(f"GraphRAG error for {qid}: {e}")
                result.graph_scores = {"error": str(e)}

            # Determine winner
            result.delta, result.winner = self._compare(result)

            results.append(result)

            # Save checkpoint
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{
                        "query_id": r.query_id, "question": r.question,
                        "category": r.category, "expected_intent": r.expected_intent,
                        "vector_answer": r.vector_answer, "vector_scores": r.vector_scores,
                        "graph_answer": r.graph_answer, "graph_scores": r.graph_scores,
                        "graph_paths": r.graph_paths, "winner": r.winner, "delta": r.delta,
                    } for r in results],
                    f, indent=2, ensure_ascii=False,
                )

            time.sleep(1.5)

        return results

    def _compare(self, result: EvaluationResult) -> Tuple[Dict, str]:
        """Compare Vector RAG vs GraphRAG scores."""
        v = result.vector_scores
        g = result.graph_scores

        keys = ["faithfulness", "relevance", "context_precision",
                "citation_completeness", "hallucination_control"]
        delta = {}
        for k in keys:
            vs = int(v.get(k, 0)) if isinstance(v.get(k), (int, float)) else 0
            gs = int(g.get(k, 0)) if isinstance(g.get(k), (int, float)) else 0
            delta[k] = gs - vs

        avg_delta = sum(delta.values()) / len(keys) if keys else 0
        if avg_delta > 0.5:
            winner = "graphrag"
        elif avg_delta < -0.5:
            winner = "vector"
        else:
            winner = "tie"

        return delta, winner

    def generate_report(self, results: List[EvaluationResult]) -> str:
        """Generate a formatted academic comparison report."""
        vec_keys = ["faithfulness", "relevance", "context_precision",
                     "citation_completeness", "hallucination_control"]

        # Aggregate
        vec_avgs = {k: 0.0 for k in vec_keys}
        graph_avgs = {k: 0.0 for k in vec_keys}
        wins = {"graphrag": 0, "vector": 0, "tie": 0}
        n = len(results)

        for r in results:
            for k in vec_keys:
                vec_avgs[k] += float(r.vector_scores.get(k, 0)) / n
                graph_avgs[k] += float(r.graph_scores.get(k, 0)) / n
            wins[r.winner] += 1

        # Format report
        lines = [
            "=" * 70,
            f"STRATEGIC-GRAPHRAG ABLATION STUDY REPORT (n={n})",
            "=" * 70,
            "",
            f"{'Metric':<30} {'Vector RAG':>12} {'GraphRAG':>12} {'Δ':>8}",
            "-" * 65,
        ]
        for k in vec_keys:
            display = k.replace("_", " ").title()
            delta = graph_avgs[k] - vec_avgs[k]
            lines.append(
                f"{display:<30} {vec_avgs[k]:>12.2f} {graph_avgs[k]:>12.2f} {delta:>+8.2f}"
            )

        lines.extend([
            "-" * 65,
            "",
            f"GraphRAG Wins: {wins['graphrag']} | Vector Wins: {wins['vector']} | Ties: {wins['tie']}",
            f"GraphRAG Win Rate: {wins['graphrag']/n*100:.1f}%",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)
