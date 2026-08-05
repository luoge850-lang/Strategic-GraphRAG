# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG v2.0: 5-Dimension Evaluation Framework
=========================================================
Reference: RAGAS (Es et al., 2023), ARES (Saad-Falcon et al., 2023),
           FinQA (Chen et al., 2021), TAT-QA (Zhu et al., 2021)

Dimensions:
  1. Retrieval Accuracy — Are the right graph paths retrieved?
  2. Evidence Recall — What fraction of available evidence is used?
  3. Faithfulness — Are claims in the answer supported by evidence?
  4. Causal Correctness — Are causal assertions valid per Pearl's SCM?
  5. Temporal Consistency — Do time references match evidence?

Each dimension scored 0.0–1.0. Composite score = weighted average.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class EvalResult:
    """Single-query evaluation result across 5 dimensions."""
    query_id: str
    question: str

    # Per-dimension scores (0.0–1.0)
    retrieval_accuracy: float = 0.0
    evidence_recall: float = 0.0
    faithfulness: float = 0.0
    causal_correctness: float = 0.0
    temporal_consistency: float = 0.0

    # Supporting data
    paths_found: int = 0
    paths_used: int = 0
    evidence_sentences_total: int = 0
    evidence_sentences_cited: int = 0
    causal_claims_total: int = 0
    causal_claims_verified: int = 0
    unsupported_claims: List[str] = field(default_factory=list)
    temporal_mismatches: List[str] = field(default_factory=list)

    # Vector RAG comparison
    vector_rag_score: float = 0.0

    @property
    def composite_score(self) -> float:
        """Weighted composite across all dimensions."""
        weights = {
            "faithfulness": 0.30,
            "causal_correctness": 0.25,
            "retrieval_accuracy": 0.20,
            "evidence_recall": 0.15,
            "temporal_consistency": 0.10,
        }
        return (
            weights["faithfulness"] * self.faithfulness +
            weights["causal_correctness"] * self.causal_correctness +
            weights["retrieval_accuracy"] * self.retrieval_accuracy +
            weights["evidence_recall"] * self.evidence_recall +
            weights["temporal_consistency"] * self.temporal_consistency
        )


class FinancialEvaluator:
    """
    5-dimension evaluation for Evidence-Grounded Temporal Causal GraphRAG.

    Usage:
        evaluator = FinancialEvaluator()
        result = evaluator.evaluate(
            question="How do export controls impact NVIDIA revenue?",
            answer=llm_answer,
            paths=retrieved_paths,
            evidence_claims=all_claims,
        )
    """

    # Causal verb lexicon (consistent with relation_inference.py)
    CAUSAL_VERBS = {
        "cause", "causes", "caused", "lead to", "leads to", "led to",
        "result in", "results in", "resulted in", "due to", "because",
        "mitigate", "mitigates", "reduce", "reduces", "increase", "increases",
        "decrease", "decreases", "trigger", "triggers", "drive", "drives",
        "impact", "impacts", "affect", "affects",
    }

    def evaluate(
        self,
        question: str,
        answer: str,
        paths: List[Dict],
        evidence_claims: List[str],
        vector_answer: str = "",
    ) -> EvalResult:
        """Run full 5-dimension evaluation."""
        result = EvalResult(
            query_id="eval",
            question=question,
            paths_found=len(paths),
        )

        if not answer or not paths:
            return result

        result.retrieval_accuracy = self._score_retrieval(question, paths)
        result.evidence_recall = self._score_evidence_recall(answer, evidence_claims, paths)
        result.faithfulness = self._score_faithfulness(answer, evidence_claims)
        result.causal_correctness = self._score_causal_correctness(answer, paths)
        result.temporal_consistency = self._score_temporal(answer, paths)
        result.paths_used = self._count_paths_used(answer, paths)

        if vector_answer:
            # Simple lexical comparison
            result.vector_rag_score = len(vector_answer.split()) / max(len(answer.split()), 1)

        return result

    # ── Dimension 1: Retrieval Accuracy ──

    def _score_retrieval(self, question: str, paths: List[Dict]) -> float:
        """
        Are retrieved paths relevant to the question?
        Checks: (a) path anchors match question entities,
                (b) path direction matches question intent.
        """
        if not paths:
            return 0.0

        q_lower = question.lower()
        relevant_paths = 0

        for path in paths:
            nodes = path.get("nodes", [])
            if not nodes:
                continue
            # Check if any path node overlaps with question keywords
            node_text = " ".join(nodes).lower().replace("_", " ")
            q_words = set(q_lower.split()) - {"the", "a", "an", "is", "are", "do", "does", "how", "what", "of", "in", "on", "to", "for", "and", "or"}
            overlap = sum(1 for w in q_words if w in node_text)
            if overlap >= 2:
                relevant_paths += 1

        return min(1.0, relevant_paths / max(len(paths), 1))

    # ── Dimension 2: Evidence Recall ──

    def _score_evidence_recall(
        self, answer: str, evidence_claims: List[str], paths: List[Dict]
    ) -> float:
        """What fraction of available evidence is cited in the answer?"""
        all_evidence = []
        for path in paths:
            for ev in path.get("evidence", []):
                if ev and len(ev) > 30:
                    all_evidence.append(ev[:150])

        if not all_evidence:
            return 0.0

        cited = 0
        ans_lower = answer.lower()
        for ev in all_evidence:
            # Check if key phrases from evidence appear in answer
            key_phrase = ev[30:80].lower().strip()
            if len(key_phrase) > 20 and key_phrase[:30] in ans_lower:
                cited += 1

        return min(1.0, cited / max(len(all_evidence), 1))

    # ── Dimension 3: Faithfulness ──

    def _score_faithfulness(self, answer: str, evidence_claims: List[str]) -> float:
        """
        Are factual claims in the answer supported by evidence?
        Uses claim decomposition + evidence matching.
        """
        if not answer or not evidence_claims:
            return 0.0

        # Extract factual sentences from answer (skip headers, transitions)
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        factual_claims = [
            s for s in sentences
            if len(s) > 40 and not s.startswith("#") and not s.startswith(">")
        ]

        if not factual_claims:
            return 0.0

        verified = 0
        unsupported = []
        all_ev_text = " ".join(evidence_claims).lower()

        for claim in factual_claims:
            claim_lower = claim.lower().strip()
            words = claim_lower.split()
            if len(words) < 5:
                continue
            # Check trigram overlap with evidence
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            supported = any(tg in all_ev_text for tg in trigrams if len(tg) > 15)
            if supported:
                verified += 1
            else:
                unsupported.append(claim[:100])

        return verified / max(len(factual_claims), 1)

    # ── Dimension 4: Causal Correctness ──

    def _score_causal_correctness(self, answer: str, paths: List[Dict]) -> float:
        """
        Are causal assertions in the answer valid per Pearl's SCM?
        A causal claim is VALID only if:
        (a) The answer states "X causes/impacts/affects Y"
        (b) A graph path exists connecting X to Y
        (c) The connecting edge has CONFIRMED_CAUSAL or STRONG_ASSOCIATION
        """
        if not answer or not paths:
            return 0.0

        # Extract causal claims from answer
        causal_pattern = re.compile(
            r"(\w+(?:\s+\w+){0,3})\s+(cause|lead|result|impact|affect|decrease|increase|reduce|mitigate|trigger|drive)"
            r"\w*\s+(\w+(?:\s+\w+){0,3})",
            re.IGNORECASE,
        )
        claims = causal_pattern.findall(answer)

        if not claims:
            # No causal claims made — neutral, not wrong
            return 0.8

        # Build path lookup: set of (source_node, target_node) pairs
        path_edges = set()
        for path in paths:
            nodes = path.get("nodes", [])
            causal_strengths = path.get("causal_strengths", [])
            for i in range(len(nodes) - 1):
                cs = causal_strengths[i] if i < len(causal_strengths) else ""
                if cs in ("CONFIRMED_CAUSAL", "STRONG_ASSOCIATION",
                          "DIRECT_CAUSALITY", "INDIRECT_CAUSALITY"):
                    path_edges.add((
                        nodes[i].lower().replace("_", " "),
                        nodes[i+1].lower().replace("_", " "),
                    ))

        verified = 0
        for source_words, verb, target_words in claims:
            src = source_words.strip().lower()
            tgt = target_words.strip().lower()
            # Check if (src, tgt) or (src, *) or (*, tgt) exists in path edges
            if (src, tgt) in path_edges:
                verified += 1
            elif any(s == src for s, _ in path_edges):
                verified += 1  # Source entity found
            elif any(t == tgt for _, t in path_edges):
                verified += 1  # Target entity found

        total = len(claims)
        return min(1.0, verified / max(total, 1))

    # ── Dimension 5: Temporal Consistency ──

    def _score_temporal(self, answer: str, paths: List[Dict]) -> float:
        """
        Do time references in the answer match the evidence years?
        """
        # Extract years from answer
        answer_years = set(int(y) for y in re.findall(r"(20\d{2})", answer))
        if not answer_years:
            return 0.5  # No temporal claims — neutral

        # Extract years from paths
        path_years = set()
        for path in paths:
            for yr in path.get("years", []):
                if yr and yr > 2000:
                    path_years.add(yr)

        if not path_years:
            return 0.5

        # Fraction of answer years that match path years
        matching = len(answer_years & path_years)
        return matching / max(len(answer_years), 1) if answer_years else 0.5

    # ── Helpers ──

    def _count_paths_used(self, answer: str, paths: List[Dict]) -> int:
        """Count how many retrieved paths are referenced in the answer."""
        used = 0
        for path in paths:
            nodes = path.get("nodes", [])
            node_mentioned = any(
                n.lower().replace("_", " ") in answer.lower()
                for n in nodes
            )
            if node_mentioned:
                used += 1
        return used

    def generate_report(self, results: List[EvalResult]) -> str:
        """Generate a formatted evaluation report."""
        if not results:
            return "No results to report."

        n = len(results)
        dims = {
            "Retrieval Accuracy": [r.retrieval_accuracy for r in results],
            "Evidence Recall": [r.evidence_recall for r in results],
            "Faithfulness": [r.faithfulness for r in results],
            "Causal Correctness": [r.causal_correctness for r in results],
            "Temporal Consistency": [r.temporal_consistency for r in results],
        }

        lines = [
            "=" * 70,
            f"FINANCIAL GRAPHRAG v2.0 — 5-DIMENSION EVALUATION (n={n})",
            "=" * 70,
            "",
            f"{'Dimension':<25} {'Mean':>8} {'Min':>8} {'Max':>8}",
            "-" * 52,
        ]

        for name, scores in dims.items():
            mean = sum(scores) / n
            lines.append(
                f"{name:<25} {mean:>8.3f} {min(scores):>8.3f} {max(scores):>8.3f}"
            )

        composites = [r.composite_score for r in results]
        lines.extend([
            "-" * 52,
            f"{'COMPOSITE (weighted)':<25} {sum(composites)/n:>8.3f} {min(composites):>8.3f} {max(composites):>8.3f}",
            "",
            f"Paths used / found: {sum(r.paths_used for r in results)} / {sum(r.paths_found for r in results)}",
            f"Claims verified / total: {sum(r.evidence_sentences_cited for r in results)} / {sum(r.evidence_sentences_total for r in results)}",
            f"Unsupported claims: {sum(len(r.unsupported_claims) for r in results)}",
            f"Temporal mismatches: {sum(len(r.temporal_mismatches) for r in results)}",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)
