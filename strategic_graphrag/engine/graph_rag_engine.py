# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Causal Multi-Hop Reasoning Engine
=====================================================
Core inference engine that performs structured causal path search
over the 6-layer temporal financial knowledge graph.

Architecture:
  1. Query Analysis → Intent + Anchor Entities
  2. Multi-Hop Path Search → Cypher traversal with constraints
  3. Path Scoring → Causal logic scoring (not just semantic similarity)
  4. Evidence Collection → Provenance chain per path
  5. LLM Synthesis → Structured financial report with citations

Key differentiator from Vector RAG:
  - Real causal path traversal (not semantic neighbor matching)
  - Temporal constraint filtering (risk evolution over time)
  - Evidence provenance (every claim traced to Document+Page+Sentence)
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv

from ..ontology.intent_classifier import classify_intent, get_retrieval_strategy, extract_financial_entities_from_query
from ..ontology.relation_inference import VALID_RELATIONS, CAUSAL_STRENGTHS, detect_causal_strength

load_dotenv()
logger = logging.getLogger("GraphRAGEngine")

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CausalPath:
    """A single causal path through the knowledge graph."""
    path_id: str
    nodes: List[str]           # Ordered node names
    node_labels: List[str]     # Node labels (Company, RiskFactor, etc.)
    relationships: List[str]   # Relationship types
    causal_strengths: List[str] # Causal strength per hop
    evidence: List[str]        # Evidence sentences per hop
    pages: List[int]           # Source pages per hop
    years: List[int]           # Years per hop
    evidence_ids: List[str]    # Claim-level provenance IDs per hop
    filings: List[str]         # Source filing per hop
    total_hops: int = 0
    aggregate_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_trace_string(self) -> str:
        """Format as human-readable causal chain trace."""
        parts = []
        for i in range(len(self.nodes) - 1):
            parts.append(
                f"[{self.nodes[i]}] "
                f"--({self.relationships[i]})--> "
                f"[{self.nodes[i+1]}] "
                f"(Strength: {self.causal_strengths[i]}, "
                f"Year: {self.years[i]}, "
                f"Page: {self.pages[i]})"
            )
        return "\n".join(parts)

    def to_evidence_chain(self) -> str:
        """Format as evidence chain with citations."""
        parts = []
        for i, ev in enumerate(self.evidence):
            if ev:
                parts.append(
                    f"[Evidence {i+1}] {ev[:300]} "
                    f"(Year: {self.years[i]}, Page: {self.pages[i]})"
                )
        return "\n".join(parts)


# =============================================================================
# Causal Path Scoring
# =============================================================================

class PathScorer:
    """
    Scores causal paths based on multiple dimensions:
      1. Causal Strength: DIRECT > INDIRECT > ASSOCIATION > SPECULATIVE
      2. Path Completeness: Full chains (Risk→Mech→Metric) score higher
      3. Temporal Coherence: Paths progressing through time correctly
      4. Evidence Quality: Multiple evidence sentences + high confidence
    """

    STRENGTH_WEIGHTS = {
        "CONFIRMED_CAUSAL": 1.0,
        "STRONG_ASSOCIATION": 0.7,
        "WEAK_ASSOCIATION": 0.4,
        "DISCLOSED_ONLY": 0.3,
        "INFERRED": 0.2,
        "DIRECT_CAUSALITY": 1.0,
        "INDIRECT_CAUSALITY": 0.7,
        "RISK_ASSOCIATION": 0.4,
        "SPECULATIVE_RELATION": 0.2,
        "DISCLOSED_EXPOSURE": 0.5,
    }

    def __init__(self):
        pass

    def score_path(self, path: CausalPath) -> float:
        """Score a single causal path across all dimensions."""
        scores = {}

        # 1. Causal Strength Score (mean across hops)
        strength_scores = [
            self.STRENGTH_WEIGHTS.get(cs, 0.3)
            for cs in path.causal_strengths
        ]
        scores["causal_strength"] = np.mean(strength_scores) if strength_scores else 0.0

        # 2. Path Completeness Score
        scores["completeness"] = self._score_completeness(path)

        # 3. Temporal Coherence Score
        scores["temporal_coherence"] = self._score_temporal(path)

        # 4. Evidence Quality Score
        scores["evidence_quality"] = self._score_evidence(path)

        # 5. Hop Penalty (prefer shorter, more direct paths)
        scores["hop_efficiency"] = 1.0 / (1.0 + 0.2 * path.total_hops)

        # Weighted aggregate
        weights = {
            "causal_strength": 0.35,
            "completeness": 0.25,
            "temporal_coherence": 0.15,
            "evidence_quality": 0.15,
            "hop_efficiency": 0.10,
        }
        aggregate = sum(scores[k] * weights[k] for k in weights)
        path.aggregate_score = round(aggregate, 4)
        path.score_breakdown = {k: round(v, 4) for k, v in scores.items()}
        return path.aggregate_score

    def _score_completeness(self, path: CausalPath) -> float:
        """
        Score path structural completeness.
        Ideal: RiskFactor → Mechanism → FinancialMetric (3-hop causal chain)
        Also good: Strategy → RiskFactor (mitigation chain)
        """
        labels = path.node_labels
        score = 0.5  # base

        # Check for complete causal chain: Risk → Mech → Metric
        has_risk = any("RiskFactor" in l for l in labels)
        has_mechanism = any("Mechanism" in l for l in labels)
        has_metric = any("FinancialMetric" in l for l in labels)
        has_strategy = any("Strategy" in l for l in labels)
        has_regulation = any("Regulation" in l for l in labels)

        if has_risk and has_mechanism and has_metric:
            score = 1.0  # Perfect causal chain
        elif has_risk and has_metric:
            score = 0.85  # Direct risk-to-metric
        elif has_strategy and has_risk:
            score = 0.8   # Mitigation chain
        elif has_regulation and has_risk:
            score = 0.75  # Regulatory chain
        elif has_mechanism and has_metric:
            score = 0.7   # Mechanism-to-impact
        elif has_risk:
            score = 0.6   # Risk only

        return score

    def _score_temporal(self, path: CausalPath) -> float:
        """Score temporal coherence: do years progress logically?"""
        if len(path.years) < 2:
            return 0.5

        # Check if years are non-decreasing (causality flows forward in time)
        increasing = 0
        for i in range(len(path.years) - 1):
            if path.years[i + 1] >= path.years[i]:
                increasing += 1

        ratio = increasing / max(len(path.years) - 1, 1)
        return 0.5 + 0.5 * ratio

    def _score_evidence(self, path: CausalPath) -> float:
        """Score evidence quality based on presence and length."""
        if not path.evidence:
            return 0.1

        non_empty = [e for e in path.evidence if e and len(e) > 20]
        if not non_empty:
            return 0.1

        # Ratio of hops with evidence
        evidence_ratio = len(non_empty) / max(len(path.evidence), 1)

        # Average evidence length (longer = more detailed = better, up to a point)
        avg_len = np.mean([min(len(e), 500) for e in non_empty])
        length_score = min(avg_len / 200, 1.0)

        return 0.5 * evidence_ratio + 0.5 * length_score


# =============================================================================
# Neo4j Path Finder
# =============================================================================

class CausalPathFinder:
    """
    Finds multi-hop causal paths through the Neo4j knowledge graph.
    Uses intent-aware Cypher generation with temporal constraints.
    """

    # Causal relationship types used for path traversal
    # Default causal relation types — fallback only; actual types are discovered at runtime
    CAUSAL_REL_TYPES = [
        "CAUSES", "TRIGGERS", "AMPLIFIES",
        "INCREASES", "DECREASES", "EXPOSED_TO", "MITIGATES",
        "CONSTRAINS", "OPERATES_IN", "DEPENDS_ON", "IMPLEMENTS",
    ]

    def __init__(self, driver):
        self.driver = driver
        self._available_rel_types: List[str] = []
        self._rel_types_fetched: float = 0.0  # epoch timestamp of last fetch

    def _get_available_rel_types(self) -> List[str]:
        """Discover which relationship types actually exist in Neo4j. Cached for 5 min."""
        import time
        now = time.time()
        if self._available_rel_types and (now - self._rel_types_fetched) < 300:
            return self._available_rel_types
        try:
            with self.driver.session() as session:
                result = session.run("CALL db.relationshipTypes()")
                self._available_rel_types = [r[0] for r in result]
                self._rel_types_fetched = now
                return self._available_rel_types
        except Exception:
            return self.CAUSAL_REL_TYPES  # fallback to hardcoded list

    def find_paths(
        self,
        anchor_entities: List[str],
        max_hops: int = 4,
        intent: str = "CAUSAL_CHAIN",
        relation_preference: List[str] = None,
        year_constraint: int = None,
        year_start: int = None,
        year_end: int = None,
        max_paths: int = 20,
    ) -> List[CausalPath]:
        """
        Find causal paths originating from anchor entities.

        Args:
            anchor_entities: Entity names/IDs to start traversal from
            max_hops: Maximum path length in hops
            intent: Query intent for relationship filtering
            relation_preference: Preferred relationship types
            year_constraint: Only return paths observed in this year or later
            max_paths: Maximum number of paths to return

        Returns:
            List of CausalPath objects, ranked by score
        """
        if not anchor_entities:
            return []

        # Discover which relationship types ACTUALLY exist in the graph
        available = self._get_available_rel_types()
        rel_types = relation_preference or self.CAUSAL_REL_TYPES
        # Intersect with what actually exists — avoid Cypher errors on missing types
        rel_types = [r for r in rel_types if r in available]
        if not rel_types:
            rel_types = [r for r in self.CAUSAL_REL_TYPES if r in available]
        if not rel_types:  # truly nothing — return empty
            return []

        # Build Cypher query for multi-hop path search. Relationship types
        # come from the controlled ontology/runtime registry; anchors and
        # temporal bounds are passed as parameters.
        rel_types = [
            rel for rel in rel_types
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rel or "")
        ]
        rel_pattern = "|".join(rel_types)

        # Match anchors by name or ID
        anchor_conditions = []
        query_params = {}
        for i, anchor in enumerate(anchor_entities):
            clean = anchor.strip().upper().replace(" ", "_")
            anchor_conditions.append(
                f"(toLower(coalesce(n.id, '')) CONTAINS toLower($anchor_id_{i}) OR "
                f"toLower(coalesce(n.name, '')) CONTAINS toLower($anchor_name_{i}))"
            )
            query_params[f"anchor_id_{i}"] = clean
            query_params[f"anchor_name_{i}"] = anchor.strip()
        anchor_clause = " OR ".join(anchor_conditions)

        if year_start is None:
            year_start = year_constraint
        year_filter = ""
        temporal_conditions = []
        if year_start is not None:
            temporal_conditions.append("r.year IS NOT NULL AND r.year >= $year_start")
            query_params["year_start"] = year_start
        if year_end is not None:
            temporal_conditions.append("r.year IS NOT NULL AND r.year <= $year_end")
            query_params["year_end"] = year_end
        if temporal_conditions:
            year_filter = (
                "AND ALL(r IN relationships(p) WHERE "
                + " AND ".join(temporal_conditions)
                + ")"
            )

        cypher = f"""
        MATCH (n)
        WHERE {anchor_clause}
        MATCH p = (n)-[:{rel_pattern}*1..{max_hops}]->(m)
        WHERE n.id <> m.id
        {year_filter}
        WITH p, relationships(p) AS rels, nodes(p) AS nds
        WHERE ALL(r IN rels WHERE type(r) IS NOT NULL
                  AND r.year IS NOT NULL
                  AND r.evidence_id IS NOT NULL
                  AND size(trim(coalesce(r.evidence_sentence, ''))) >= 20
                  AND EXISTS {{
                      MATCH (claim:EvidenceClaim {{id: r.evidence_id}})
                      WHERE claim.verification_status = 'VERBATIM'
                  }})
        RETURN
            [nd IN nds | coalesce(nd.name, nd.id)] AS node_names,
            [nd IN nds | labels(nd)[0]] AS node_labels,
            [r IN rels | type(r)] AS relationships,
            [r IN rels | coalesce(r.causal_strength, 'DISCLOSED_EXPOSURE')] AS causal_strengths,
            [r IN rels | coalesce(r.evidence_sentence, '')] AS evidence,
            [r IN rels | coalesce(r.page, 0)] AS pages,
            [r IN rels | coalesce(r.year, 0)] AS years,
            [r IN rels | coalesce(r.evidence_id, '')] AS evidence_ids,
            [r IN rels | coalesce(r.source_filing, r.filing, '')] AS filings,
            length(p) AS hops
        ORDER BY hops ASC
        LIMIT {max_paths * 3}
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher, **query_params)
                records = list(result)
        except Neo4jError as e:
            logger.error(f"Path search error: {e}")
            return []

        if not records:
            logger.info("No causal paths found for anchors: %s", anchor_entities)
            return []

        # Convert to CausalPath objects
        paths = []
        for i, rec in enumerate(records):
            path = CausalPath(
                path_id=f"path_{i:03d}",
                nodes=rec["node_names"],
                node_labels=rec["node_labels"],
                relationships=rec["relationships"],
                causal_strengths=rec["causal_strengths"],
                evidence=rec["evidence"],
                pages=rec["pages"],
                years=rec["years"],
                evidence_ids=rec["evidence_ids"],
                filings=rec["filings"],
                total_hops=rec["hops"],
            )
            paths.append(path)

        logger.info(f"Found {len(paths)} candidate causal paths")
        return paths

    def find_evidence_for_entity(self, entity_name: str, limit: int = 10) -> List[Dict]:
        """Find all evidence sentences related to a specific entity."""
        cypher = """
        MATCH (n)
        WHERE (n.name = $name OR n.id = $name)
        MATCH (claim:EvidenceClaim)-[:ABOUT_SOURCE|ABOUT_TARGET]->(n)
        MATCH (claim)-[:SUPPORTED_BY]->(s:Sentence)
        RETURN claim.text AS evidence,
               claim.page AS page,
               claim.section AS section,
               claim.relation_type AS relation,
               claim.doc_id AS filing,
               claim.fiscal_year AS fiscal_year,
               claim.id AS evidence_id,
               CASE WHEN claim.source_id = n.id
                    THEN claim.target_id ELSE claim.source_id END AS connected_to
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                results = session.run(cypher, name=entity_name, limit=limit)
                return [r.data() for r in results]
        except Neo4jError as e:
            logger.warning(f"Evidence search error: {e}")
            return []

    def find_temporal_evolution(self, risk_name: str) -> List[Dict]:
        """Find how a risk evolves across fiscal years."""
        cypher = """
        MATCH (risk:RiskFactor)
        WHERE risk.id = $risk_id OR toLower(risk.name) = toLower($risk_id)
        MATCH (risk)-[r]->(target)
        WHERE r.year IS NOT NULL
        MATCH (claim:EvidenceClaim {id: r.evidence_id})
        WHERE claim.verification_status = 'VERBATIM'
        RETURN target.name AS target, type(r) AS relation,
               r.causal_strength AS strength, r.year AS year,
               claim.text AS evidence, claim.page AS page,
               claim.doc_id AS filing, claim.id AS evidence_id
        ORDER BY r.year, r.causal_strength DESC
        LIMIT 20
        """
        try:
            with self.driver.session() as session:
                results = session.run(cypher, risk_id=risk_name)
                return [r.data() for r in results]
        except Neo4jError as e:
            logger.warning(f"Temporal evolution error: {e}")
            return []


# =============================================================================
# Main GraphRAG Engine
# =============================================================================

class GraphRAGEngine:
    """
    Primary inference engine for Strategic-GraphRAG.

    Pipeline:
      Query → Intent Analysis → Anchor Extraction → Multi-Hop Path Search
      → Path Scoring → Evidence Collection → LLM Synthesis → Response
    """

    REPORT_SYSTEM_PROMPT = """You are a senior financial risk analyst at a top-tier investment research firm.
Your specialty is tracing causal chains through SEC filings and synthesizing actionable insights.

[CAUSAL VERIFICATION RULES — VIOLATION MEANS THE ANALYSIS IS WRONG]:
1. EXPLICIT CAUSATION ONLY: A causal link is valid ONLY if the evidence text EXPLICITLY states
   that entity A causes/mitigates/impacts entity B. Two entities appearing in the same
   section is NOT causation. Corporate action (e.g. "acquired a supplier") is NOT mitigation
   unless the filing explicitly links it to risk reduction.
2. CONFIDENCE TIERS — label every causal link:
   [CONFIRMED] — Evidence explicitly states "X causes/mitigates/impacts Y"
   [PLAUSIBLE] — Evidence strongly implies causation but does not use explicit causal language
   [SPECULATIVE] — Entities are connected in the graph but evidence only describes each separately
   If a path contains only [SPECULATIVE] links, say so and downgrade the conclusion.
3. CLAIM-EVIDENCE ALIGNMENT: Every sentence you write that makes a factual claim MUST be
   anchored to a specific EvidenceClaim ID and evidence quote. If no evidence directly supports a claim,
   either (a) remove the claim, or (b) label it as analyst inference.
4. NO IMPLICIT ASSUMPTIONS: Do NOT assume that because a company "implements" a strategy,
   that strategy successfully mitigates a risk. The evidence must say so explicitly.

[ANALYSIS DEPTH REQUIREMENTS]:
5. ANSWER THE QUESTION ASKED: If the query is about "export controls → revenue",
   your primary focus must be the export-controls-to-revenue causal chain.
   Do not drift to general supply chain topics unless they are part of that specific chain.
6. TEMPORAL CAUSAL EVOLUTION: Trace how risks evolve across fiscal years.
   "In FY2024, export controls were described as a compliance matter (p.12).
   By FY2025, the filing language shifted to material revenue impact (p.18),
   suggesting escalation." If only one year of evidence exists, state this limitation.
7. QUANTIFY WHEN POSSIBLE: Report dollar amounts, percentages, and materiality thresholds
   from the evidence. If the filing says a risk "could materially affect" results,
   that is significant. If it says "has not had a material impact," report that too.
8. IDENTIFY GAPS: End every analysis with what the evidence CANNOT conclude.
   "The graph does not contain evidence that export controls have reduced NVIDIA's
   actual revenue — only that the company acknowledges this as a risk factor."

[OUTPUT FORMAT — STRICT]:
## Executive Summary
[2-3 sentences. Answer the query directly. State the strongest finding, then the limitation.]

## Analysis
[Narrative prose only. For each causal chain: explain in natural language what the evidence shows,
   quote supporting text from the filing with EvidenceClaim IDs, page numbers, and assess confidence.
DO NOT output raw step traces like "Step 1: X → Y → Z".
DO NOT output machine-looking pathway headers like "Pathway 1" or "Evidence Chain 1".
Write continuous paragraphs that a business reader would find natural.]

## Evidence Quality
[One paragraph assessing overall evidence strength: explicit vs implied, corroboration, temporal span.]

## Limitations
[What the evidence CANNOT conclude. What additional data would help. Be honest about gaps.]

CRITICAL STYLE RULES:
- NO raw machine traces, step numbers, or arrow chains in your output.
- NO pathway/chain numbering in headers.
- Write as a senior analyst briefing an institutional investor — natural, authoritative, evidence-grounded.
- Every factual statement must reference a specific page from the filing."""

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        llm_provider=None,
        model_name: str = None,
    ):
        # Neo4j connection with AuraDB-optimized pooling
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password

        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            max_connection_lifetime=1800,    # 30min — AuraDB free tier timeout
            max_connection_pool_size=8,       # Small pool for single-user dev
            connection_acquisition_timeout=15,  # Quick timeout, trigger reconnect
            keep_alive=True,                   # Keep connections warm
        )

        # Initialize components
        self.path_finder = CausalPathFinder(self.driver)
        self.path_scorer = PathScorer()

        # Initialize LLM via unified provider
        from ..llm_provider import get_llm
        if llm_provider is not None:
            self.llm = llm_provider
        else:
            self.llm = get_llm()
        self.model_name = model_name or self.llm.default_model
        self._has_llm = self.llm.available

        if not self._has_llm:
            logger.warning("No LLM provider available. Will return graph traces without synthesis.")

        # Initialize Cross-Encoder for semantic reranking (optional)
        self.reranker = None
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
            )
            logger.info("Cross-Encoder reranker loaded")
        except Exception:
            logger.info("Cross-Encoder not available — using causal scoring only")

    def close(self):
        if self.driver:
            self.driver.close()

    # ── Main Inference Pipeline ──

    def _ensure_connection(self) -> bool:
        """Verify Neo4j connectivity. Reconnect driver if connection is dead."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"Neo4j connection lost: {e}. Reconnecting...")
            try:
                if self.driver:
                    self.driver.close()
            except Exception:
                pass
            self.driver = GraphDatabase.driver(
                self._neo4j_uri,
                auth=(self._neo4j_user, self._neo4j_password),
                max_connection_lifetime=1800,
                max_connection_pool_size=8,
                connection_acquisition_timeout=15,
                keep_alive=True,
            )
            self.path_finder.driver = self.driver
            try:
                self.driver.verify_connectivity()
                logger.info("Neo4j reconnected successfully")
                return True
            except Exception as e2:
                logger.error(f"Neo4j reconnection failed: {e2}")
                return False

    def query(
        self,
        user_query: str,
        top_k: int = 10,
        year_start: int = None,
        year_end: int = None,
    ) -> Dict:
        """
        Execute the complete GraphRAG inference pipeline.

        Args:
            user_query: Natural language financial question
            top_k: Number of top paths to include in the report
            year_start: Optional inclusive lower fiscal-year bound
            year_end: Optional inclusive upper fiscal-year bound

        Returns:
            Dict with: answer, paths, evidence, metadata
        """
        logger.info(f"Query: {user_query[:80]}...")

        # Pre-flight: ensure Neo4j is connected
        if not self._ensure_connection():
            return {
                "query": user_query,
                "intent": "FALLBACK",
                "intent_display": "Connection Error",
                "answer": "[CONNECTION ERROR] Neo4j database is unavailable. The AuraDB free tier may be restarting. Please wait 30 seconds and retry.",
                "paths": [],
                "evidence_sentences": [],
                "metadata": {"total_candidates": 0, "top_paths": 0, "anchors_used": [], "avg_score": 0},
            }

        # Step 1: Intent Analysis
        intent_id, intent_sig = classify_intent(user_query)
        strategy = get_retrieval_strategy(user_query)
        logger.info(f"Intent: {intent_id} | Max hops: {strategy['max_hops']}")

        # Step 2: Entity Extraction
        query_entities = extract_financial_entities_from_query(user_query)
        # Add LLM-extracted anchors
        llm_anchors = self._llm_extract_anchors(user_query)
        all_anchors = list(dict.fromkeys(query_entities + llm_anchors))
        logger.info(f"Anchors: {all_anchors}")

        # Step 3: Multi-Hop Path Search
        candidate_paths = self.path_finder.find_paths(
            anchor_entities=all_anchors,
            max_hops=strategy["max_hops"],
            intent=intent_id,
            relation_preference=strategy["relation_preference"],
            year_start=year_start,
            year_end=year_end,
            max_paths=top_k * 3,
        )

        if not candidate_paths:
            return self._fallback_response(user_query, all_anchors)

        # Step 4: Score Paths
        for path in candidate_paths:
            self.path_scorer.score_path(path)
        candidate_paths.sort(key=lambda p: p.aggregate_score, reverse=True)

        # Step 5: Semantic Reranking (if Cross-Encoder available)
        if self.reranker and len(candidate_paths) > top_k:
            reranked = self._rerank_paths(user_query, candidate_paths)
            top_paths = reranked[:top_k]
        else:
            top_paths = candidate_paths[:top_k]

        # Step 6: Evidence Collection
        all_evidence = []
        for path in top_paths:
            all_evidence.extend(
                [e for e in path.evidence if e and len(e) > 10]
            )

        # Step 7: LLM Synthesis
        answer = self._synthesize_report(user_query, top_paths, intent_id)

        return {
            "query": user_query,
            "intent": intent_id,
            "intent_display": intent_sig.display_name,
            "answer": answer,
            "paths": [self._serialize_path(p) for p in top_paths],
            "evidence_sentences": all_evidence[:20],
            "metadata": {
                "total_candidates": len(candidate_paths),
                "top_paths": len(top_paths),
                "anchors_used": all_anchors,
                "avg_score": round(np.mean([p.aggregate_score for p in top_paths]), 4) if top_paths else 0,
            },
        }

    # ── LLM Synthesis ──

    def _synthesize_report(
        self, query: str, paths: List[CausalPath], intent: str
    ) -> str:
        """Generate a professional financial analysis report from causal paths."""
        if not paths:
            return "[INSUFFICIENT EVIDENCE] No causal pathways found."

        # Build rich, human-readable context with full evidence
        path_descriptions = []
        for i, path in enumerate(paths[:5]):  # top 5 paths for depth
            desc = f"## Evidence Chain {i+1} (Score: {path.aggregate_score:.2f}, {path.total_hops} hops)\n"

            # Step-by-step chain — format as readable narrative, not machine trace
            for j in range(len(path.nodes) - 1):
                strength = path.causal_strengths[j] if j < len(path.causal_strengths) else "UNKNOWN"
                page = path.pages[j] if j < len(path.pages) else "?"
                year = path.years[j] if j < len(path.years) else "?"
                claim_id = path.evidence_ids[j] if j < len(path.evidence_ids) else "?"
                rel = path.relationships[j] if j < len(path.relationships) else "RELATES_TO"
                desc += (
                    f"Link {j+1}: {path.nodes[j]} {rel} {path.nodes[j+1]}\n"
                    f"  Strength: {strength} | FY{year} | p.{page} | EvidenceClaim: {claim_id}\n"
                )

            # Full evidence sentences
            desc += "\nEvidence excerpts from SEC filing:\n"
            for j, ev in enumerate(path.evidence):
                if ev and ev.strip():
                    yr = path.years[j] if j < len(path.years) else "?"
                    pg = path.pages[j] if j < len(path.pages) else "?"
                    claim_id = path.evidence_ids[j] if j < len(path.evidence_ids) else "?"
                    filing = path.filings[j] if j < len(path.filings) else "?"
                    # Use full evidence, not truncated
                    desc += (
                        f"  [{j+1}] (EvidenceClaim: {claim_id}; filing: {filing}; "
                        f"p.{pg}, FY{yr}) {ev.strip()}\n"
                    )

            desc += "\n"
            path_descriptions.append(desc)

        context = "\n".join(path_descriptions)

        if not self._has_llm:
            return (
                f"[GRAPH TRACE — No LLM Synthesis Available]\n\n"
                f"Intent: {intent}\n\n{context}"
            )

        prompt = f"""Below are causal evidence chains extracted from NVIDIA's SEC filings.

[EVIDENCE DATA]:
{context}

[USER QUERY]:
{query}

[CRITICAL INSTRUCTIONS]:
1. ANSWER THE EXACT QUERY. Do not drift to adjacent topics.
2. Write in NATURAL PROSE. Do NOT output raw step traces, "Pathway N" headers, or arrow chains.
3. For each causal link: check if the evidence EXPLICITLY states causation or merely mentions both entities.
   Label in prose: "...this link is well-supported by direct evidence..." NOT "[CONFIRMED]".
4. Every claim MUST reference the EvidenceClaim ID and page from the evidence data.
5. Follow the output format: Executive Summary → Analysis → Evidence Quality → Limitations.
6. Be honest about weak links — that is valuable analysis.

Now generate the analysis report:"""

        result = self.llm.chat(
            prompt=prompt,
            system_prompt=self.REPORT_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=3000,
        )

        # Auto-fallback: if primary provider fails, try the other free provider
        if result is None:
            fallback_provider = "gemini" if self.llm.provider == "groq" else "groq"
            logger.warning(f"Primary LLM ({self.llm.provider}) failed, trying fallback ({fallback_provider})...")
            try:
                from ..llm_provider import get_llm
                fallback_llm = get_llm(provider=fallback_provider)
                if fallback_llm.available:
                    result = fallback_llm.chat(
                        prompt=prompt,
                        system_prompt=self.REPORT_SYSTEM_PROMPT,
                        temperature=0.3,
                        max_tokens=3000,
                    )
                    if result:
                        logger.info(f"Fallback LLM ({fallback_provider}) succeeded")
            except Exception as e:
                logger.error(f"Fallback LLM also failed: {e}")

        if result is None:
            return f"[SYNTHESIS ERROR: Both LLM providers unavailable]\n\n{context}"
        return result

    # ── Helpers ──

    def _llm_extract_anchors(self, query: str) -> List[str]:
        """Use LLM to extract strategic entity keywords from query."""
        if not self._has_llm:
            return []
        prompt = (
            "Extract 2-4 specific financial entities from this query. "
            "Use canonical UPPER_SNAKE_CASE names. "
            "Examples: CHIP_EXPORT_RESTRICTION, REVENUE, DATA_CENTER_MARKET, "
            "SUPPLY_CHAIN_DIVERSIFICATION. "
            "Return ONLY comma-separated list, no explanations.\n"
            f"Query: {query}"
        )
        content = self.llm.chat(prompt=prompt, temperature=0.0, max_tokens=100)
        if content is None:
            return []
        return [
            w.strip().upper().replace(" ", "_")
            for w in content.split(",")
            if len(w.strip()) > 2
        ]

    def _rerank_paths(
        self, query: str, paths: List[CausalPath]
    ) -> List[CausalPath]:
        """Semantically rerank paths using Cross-Encoder."""
        if not self.reranker or not paths:
            return paths

        path_texts = [p.to_trace_string() for p in paths]
        pairs = [[query, pt] for pt in path_texts]
        try:
            scores = self.reranker.predict(pairs)
            # Combine causal score with semantic score
            for i, path in enumerate(paths):
                semantic_score = 1.0 / (1.0 + np.exp(-float(scores[i])))
                path.aggregate_score = 0.6 * path.aggregate_score + 0.4 * semantic_score
            paths.sort(key=lambda p: p.aggregate_score, reverse=True)
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")

        return paths

    def _fallback_response(self, query: str, anchors: List[str]) -> Dict:
        """Generate a response when no paths are found."""
        return {
            "query": query,
            "intent": "FALLBACK",
            "intent_display": "No Results",
            "answer": (
                "[INSUFFICIENT EVIDENCE] The knowledge graph does not contain "
                "sufficient causal pathways to answer this query. The graph may "
                "need additional SEC filings or a broader set of financial documents "
                "to build the necessary relationship network.\n\n"
                f"Searched for: {', '.join(anchors) if anchors else 'all entities'}"
            ),
            "paths": [],
            "evidence_sentences": [],
            "metadata": {
                "total_candidates": 0,
                "top_paths": 0,
                "anchors_used": anchors,
            },
        }

    def _serialize_path(self, path: CausalPath) -> Dict:
        """Serialize a CausalPath to a JSON-safe dict."""
        return {
            "path_id": path.path_id,
            "nodes": path.nodes,
            "node_labels": path.node_labels,
            "relationships": path.relationships,
            "causal_strengths": path.causal_strengths,
            "evidence": path.evidence,
            "pages": path.pages,
            "years": path.years,
            "evidence_ids": path.evidence_ids,
            "filings": path.filings,
            "total_hops": path.total_hops,
            "score": path.aggregate_score,
            "score_breakdown": path.score_breakdown,
        }

    # ── Convenience ──

    def ask(self, query: str) -> str:
        """Simple Q&A interface returning just the answer text."""
        result = self.query(query)
        return result["answer"]

    def ask_with_paths(self, query: str) -> Tuple[str, List[Dict]]:
        """Q&A interface returning answer + path data."""
        result = self.query(query)
        return result["answer"], result["paths"]
