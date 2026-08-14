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
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv

from ..ontology.intent_classifier import classify_intent, get_retrieval_strategy, extract_financial_entities_from_query
from .query_understanding import parse_query
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
    causal_forms: List[str] = field(default_factory=list)  # Direct vs mediated form
    total_hops: int = 0
    aggregate_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    duplicate_count: int = 1
    evidence_variants: List[List[Dict[str, Any]]] = field(default_factory=list)

    def to_trace_string(self) -> str:
        """Format as human-readable causal chain trace."""
        parts = []
        for i in range(len(self.nodes) - 1):
            parts.append(
                f"[{self.nodes[i]}] "
                f"--({self.relationships[i]})--> "
                f"[{self.nodes[i+1]}] "
                f"(Strength: {self.causal_strengths[i]}, "
                f"Form: {self.causal_forms[i] if i < len(self.causal_forms) else 'UNKNOWN'}, "
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

    def fingerprint(self) -> str:
        """Return a stable identifier for evaluation and UI traceability."""
        payload = {
            "nodes": self.nodes,
            "relationships": self.relationships,
            "years": self.years,
            "pages": self.pages,
            "evidence_ids": self.evidence_ids,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def semantic_key(self) -> Tuple:
        """Identify one semantic path independent of duplicate evidence claims."""
        return (
            tuple(self.nodes),
            tuple(self.node_labels),
            tuple(self.relationships),
            tuple(self.years),
        )

    def semantic_fingerprint(self) -> str:
        """Return a stable ID for a deduplicated semantic path."""
        encoded = json.dumps(self.semantic_key(), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


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

    def find_text_anchors(self, query: str, limit: int = 8) -> List[str]:
        """Resolve query terms to canonical entities using Neo4j full-text search."""
        if not query or limit <= 0:
            return []
        try:
            with self.driver.session() as session:
                rows = list(session.run(
                    """
                    CALL db.index.fulltext.queryNodes('entity_fulltext', $search_query)
                    YIELD node, score
                    WHERE node.id IS NOT NULL
                    RETURN node.id AS id, node.name AS name, score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    search_query=query,
                    limit=int(limit),
                ))
            anchors = []
            for row in rows:
                for value in (row.get("id"), row.get("name")):
                    if value and value not in anchors:
                        anchors.append(str(value))
            return anchors[:limit]
        except Exception as exc:
            logger.info("Entity full-text lookup unavailable: %s", type(exc).__name__)
            return []

    def find_vector_evidence_anchors(
        self, vector_hits: List[Dict[str, Any]], limit: int = 12
    ) -> List[str]:
        """Join semantic filing/page hits back to strict EvidenceClaim entities."""
        pages = []
        for hit in vector_hits or []:
            metadata = hit.get("metadata") or {}
            filing = metadata.get("source_filing") or metadata.get("doc_id")
            try:
                page = int(metadata.get("page"))
            except (TypeError, ValueError):
                continue
            if filing and page > 0:
                pages.append({"doc_id": str(filing).removesuffix(".pdf"), "page": page})
        if not pages:
            return []
        try:
            with self.driver.session() as session:
                rows = session.run(
                    """
                    UNWIND $pages AS item
                    MATCH (claim:EvidenceClaim {doc_id:item.doc_id, page:item.page})
                    WHERE claim.verification_status='VERBATIM'
                    RETURN claim.source_id AS source_id, claim.target_id AS target_id,
                           count(*) AS support
                    ORDER BY support DESC LIMIT $limit
                    """,
                    pages=pages,
                    limit=int(limit),
                )
                anchors = []
                for row in rows:
                    for value in (row.get("source_id"), row.get("target_id")):
                        if value and value not in anchors:
                            anchors.append(str(value))
                return anchors[:limit]
        except Exception as exc:
            logger.info("Vector evidence anchor join unavailable: %s", type(exc).__name__)
            return []

    def find_paths(
        self,
        anchor_entities: List[str],
        max_hops: int = 4,
        intent: str = "CAUSAL_CHAIN",
        relation_preference: List[str] = None,
        year_constraint: int = None,
        year_start: int = None,
        year_end: int = None,
        source_filing: str = None,
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
                f"toLower(coalesce(n.name, '')) CONTAINS toLower($anchor_name_{i}) OR "
                f"toLower(coalesce(m.id, '')) CONTAINS toLower($anchor_id_{i}) OR "
                f"toLower(coalesce(m.name, '')) CONTAINS toLower($anchor_name_{i}))"
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
        filing_filter = ""
        if source_filing:
            filing_filter = "AND coalesce(r.source_filing, r.filing, '') = $source_filing"
            query_params["source_filing"] = source_filing
        if temporal_conditions:
            year_filter = (
                "AND ALL(r IN relationships(p) WHERE "
                + " AND ".join(temporal_conditions)
                + ")"
            )

        cypher = f"""
        MATCH p = (n)-[:{rel_pattern}*1..{max_hops}]->(m)
        WHERE n.id <> m.id
          AND ({anchor_clause})
        {year_filter}
        WITH p, relationships(p) AS rels, nodes(p) AS nds
        WHERE ALL(r IN rels WHERE type(r) IS NOT NULL
                  AND r.year IS NOT NULL
                  AND r.evidence_id IS NOT NULL
                  AND size(trim(coalesce(r.evidence_sentence, ''))) >= 20
                  {filing_filter}
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
            [r IN rels | coalesce(r.causal_form, 'UNMODELED_DIRECT')] AS causal_forms,
            length(p) AS hops
        ORDER BY hops ASC, node_names ASC, relationships ASC, evidence_ids ASC
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
                causal_forms=rec["causal_forms"],
                total_hops=rec["hops"],
            )
            paths.append(path)

        # Neo4j does not guarantee result order unless it is explicitly
        # ordered. Keep a second deterministic ordering here so direct callers
        # of find_paths() receive the same candidate sequence as the API.
        paths.sort(key=lambda p: (
            p.total_hops,
            tuple(p.nodes),
            tuple(p.relationships),
            tuple(p.years),
            tuple(p.evidence_ids),
        ))
        for i, path in enumerate(paths):
            path.path_id = f"path_{i:03d}"

        logger.info(f"Found {len(paths)} candidate causal paths")
        return paths

    def find_evidence_for_entity(
        self,
        entity_name: str,
        limit: int = 10,
        source_filing: str = None,
    ) -> List[Dict]:
        """Find all evidence sentences related to a specific entity."""
        cypher = """
        MATCH (n)
        WHERE (toLower(coalesce(n.name, '')) = toLower($name)
               OR toLower(coalesce(n.id, '')) = toLower($name))
        MATCH (claim:EvidenceClaim)-[:ABOUT_SOURCE|ABOUT_TARGET]->(n)
        WHERE $source_filing IS NULL
           OR claim.doc_id = $source_filing
           OR claim.doc_id = replace($source_filing, '.pdf', '')
        MATCH (claim)-[:SUPPORTED_BY]->(s:Sentence)
        RETURN claim.text AS evidence,
               claim.page AS page,
               claim.section AS section,
               claim.relation_type AS relation,
               claim.doc_id AS filing,
               claim.fiscal_year AS fiscal_year,
               claim.id AS evidence_id,
               CASE WHEN claim.source_id = n.id
                    THEN claim.target_id ELSE claim.source_id END AS connected_to,
               claim.metric_value AS metric_value,
               claim.metric_unit AS metric_unit,
               claim.metric_period AS metric_period,
               claim.metric_values_json AS metric_values_json
        ORDER BY CASE WHEN claim.relation_type = 'REPORTS_METRIC' THEN 0 ELSE 1 END,
                 claim.page ASC, claim.id ASC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                results = session.run(
                    cypher,
                    name=entity_name,
                    limit=limit,
                    source_filing=source_filing,
                )
                return [r.data() for r in results]
        except Neo4jError as e:
            logger.warning(f"Evidence search error: {e}")
            return []

    def find_temporal_evolution(
        self,
        risk_name: str,
        source_filing: str = None,
    ) -> List[Dict]:
        """Find how a risk evolves across fiscal years."""
        cypher = """
        MATCH (risk:RiskFactor)
        WHERE risk.id = $risk_id OR toLower(risk.name) = toLower($risk_id)
        MATCH (risk)-[r]->(target)
        WHERE r.year IS NOT NULL
          AND ($source_filing IS NULL OR coalesce(r.source_filing, r.filing, '') = $source_filing)
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
                results = session.run(
                    cypher,
                    risk_id=risk_name,
                    source_filing=source_filing,
                )
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
        self.model_name = model_name or self.llm.get_task_model("report")
        self._has_llm = self.llm.available
        # Query anchor extraction is deterministic for repeated dashboard
        # questions. A small bounded cache removes one remote LLM round-trip
        # on retries without making graph results persistent or stale.
        self._anchor_cache: Dict[str, List[str]] = {}
        self._anchor_cache_limit = 128

        if not self._has_llm:
            logger.warning("No LLM provider available. Will return graph traces without synthesis.")

        # Initialize Cross-Encoder for semantic reranking (optional)
        self.reranker = None
        try:
            if os.getenv("CROSS_ENCODER_ENABLED", "false").strip().lower() not in {
                "1", "true", "yes", "on"
            }:
                raise RuntimeError("disabled by CROSS_ENCODER_ENABLED")
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
        source_filing: str = None,
        cross_filing: bool = False,
        retrieval_mode: str = "auto",
        vector_engine=None,
        vector_top_k: int = 5,
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
        started_at = time.perf_counter()
        stage_times: Dict[str, float] = {}
        # An explicit cross-filing query must not silently fall back to the
        # active filing from .env.  Single-filing callers retain the stable
        # default scope for backwards compatibility.
        if cross_filing:
            source_filing = None
        else:
            source_filing = source_filing or os.getenv("GRAPH_ACTIVE_FILING", "").strip() or None
        requested_retrieval_mode = (retrieval_mode or "auto").strip().lower()
        structured_query = parse_query(user_query)
        retrieval_mode = self._resolve_retrieval_mode(
            user_query, requested_retrieval_mode, structured_query.target_metric
        )
        logger.info(f"Query: {user_query[:80]}... | filing={source_filing or 'ALL'}")

        vector_retrieval = {
            "status": "NOT_REQUESTED",
            "hits": [],
            "collection": None,
            "source_filing": source_filing,
        }
        vector_started = time.perf_counter()
        if retrieval_mode == "hybrid":
            try:
                if vector_engine is None:
                    from .vector_rag_baseline import VectorRAGBaseline
                    vector_engine = VectorRAGBaseline()
                vector_retrieval = vector_engine.retrieve_with_metadata(
                    user_query,
                    k=max(1, min(vector_top_k, 20)),
                    source_filing=source_filing,
                )
            except Exception as e:
                logger.warning("Hybrid vector retrieval unavailable: %s", e)
                vector_retrieval = {
                    "status": "UNAVAILABLE",
                    "hits": [],
                    "collection": None,
                    "source_filing": source_filing,
                    "error": type(e).__name__,
                }
        stage_times["vector_retrieval_ms"] = round(
            (time.perf_counter() - vector_started) * 1000, 2
        )

        # Pre-flight: ensure Neo4j is connected
        if not self._ensure_connection():
            return {
                "query": user_query,
                "intent": "FALLBACK",
                "intent_display": "Connection Error",
                "answer": "[CONNECTION ERROR] Neo4j database is unavailable. The AuraDB free tier may be restarting. Please wait 30 seconds and retry.",
                "paths": [],
                "evidence_sentences": [],
                "metadata": {
                    "total_candidates": 0,
                    "top_paths": 0,
                    "anchors_used": [],
                    "avg_score": 0,
                    "latency_ms": {"total_ms": round((time.perf_counter() - started_at) * 1000, 2)},
                },
            }

        # Step 1: Intent Analysis
        intent_started = time.perf_counter()
        intent_id, intent_sig = classify_intent(user_query)
        strategy = get_retrieval_strategy(user_query)
        # Respect explicit ontology terms in evaluation/debug questions. The
        # default strategy is risk-causal, but a user asking for PRODUCES or
        # OPERATES_IN should not have that relation silently filtered out.
        query_upper = user_query.upper().replace("-", "_")
        explicit_relations = [
            relation
            for relation in VALID_RELATIONS
            if relation in query_upper
        ]
        if explicit_relations:
            strategy["relation_preference"] = list(
                dict.fromkeys(explicit_relations + strategy["relation_preference"])
            )
        temporal_context = self._build_temporal_context(
            user_query=user_query,
            year_start=year_start,
            year_end=year_end,
        )
        # A natural-language year range must constrain retrieval even when the
        # API caller did not send explicit year_start/year_end fields.
        effective_year_start = temporal_context["year_start"]
        effective_year_end = temporal_context["year_end"]
        logger.info(f"Intent: {intent_id} | Max hops: {strategy['max_hops']}")
        stage_times["query_understanding_ms"] = round(
            (time.perf_counter() - intent_started) * 1000, 2
        )

        # Step 2: Entity Extraction
        anchor_started = time.perf_counter()
        query_entities = extract_financial_entities_from_query(user_query)
        explicit_entity_tokens = [
            token
            for token in re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", user_query)
            if token not in VALID_RELATIONS
        ]
        query_entities = list(dict.fromkeys(query_entities + explicit_entity_tokens))
        # Add LLM-extracted anchors
        llm_anchors = self._llm_extract_anchors(user_query)
        lexical_anchors = self.path_finder.find_text_anchors(user_query, limit=8)
        vector_anchors = self.path_finder.find_vector_evidence_anchors(
            vector_retrieval.get("hits", []), limit=12
        ) if retrieval_mode == "hybrid" else []
        all_anchors = list(dict.fromkeys(query_entities + lexical_anchors + vector_anchors + llm_anchors))
        target_metric = structured_query.target_metric
        metric_only = bool(target_metric) and not re.search(
            r"\b(affect|impact|cause|risk|control|constraint|exposure|supply chain|why|how)\b",
            user_query,
            re.IGNORECASE,
        )
        if metric_only:
            # Exact financial questions should search from the named metric.
            # A ubiquitous Company anchor can otherwise fill Neo4j's bounded
            # result window before the requested REPORTS_METRIC edge appears.
            all_anchors = [target_metric]
            # Exact metric questions must retrieve reported accounting facts,
            # not merely risk edges that happen to point at the same metric.
            strategy["relation_preference"] = ["REPORTS_METRIC"]
        logger.info(f"Anchors: {all_anchors}")
        stage_times["anchor_resolution_ms"] = round(
            (time.perf_counter() - anchor_started) * 1000, 2
        )

        # Step 3: Multi-Hop Path Search
        graph_started = time.perf_counter()
        # Multi-year comparisons need a wider candidate pool before the
        # year-coverage selector can reserve one path per requested endpoint.
        # A small default pool can otherwise be filled entirely by the active
        # year's lexicographically first paths.
        path_budget = top_k * 3
        if temporal_context["require_multi_year"] and source_filing is None:
            path_budget = max(path_budget, 30)
        candidate_paths = self.path_finder.find_paths(
            anchor_entities=all_anchors,
            max_hops=strategy["max_hops"],
            intent=intent_id,
            relation_preference=strategy["relation_preference"],
            year_start=effective_year_start,
            year_end=effective_year_end,
            source_filing=source_filing,
            max_paths=path_budget,
        )
        # For an explicitly identified metric, prefer paths that actually
        # contain that metric. Generic Company anchors otherwise dominate the
        # bounded candidate pool and can displace an exact REPORTS_METRIC edge.
        if target_metric:
            target_key = target_metric.lower().replace("_", " ")
            exact_metric_paths = [
                path for path in candidate_paths
                if any(
                    str(node).lower().replace("_", " ") == target_key
                    for node in path.nodes
                )
            ]
            if exact_metric_paths:
                candidate_paths = exact_metric_paths
        stage_times["graph_path_search_ms"] = round(
            (time.perf_counter() - graph_started) * 1000, 2
        )

        if not candidate_paths:
            return self._fallback_response(
                user_query,
                all_anchors,
                temporal_context=temporal_context,
            )

        # Step 4: Score Paths
        scoring_started = time.perf_counter()
        for path in candidate_paths:
            self.path_scorer.score_path(path)
        raw_candidate_count = len(candidate_paths)
        retrieval = self._apply_vector_fusion(candidate_paths, vector_retrieval)
        candidate_paths = self._deduplicate_paths(candidate_paths)
        candidate_paths.sort(key=self._path_sort_key)

        # Step 5: Semantic Reranking (if Cross-Encoder available)
        if self.reranker and len(candidate_paths) > top_k:
            candidate_paths = self._rerank_paths(user_query, candidate_paths)
        top_paths = self._select_temporal_paths(
            candidate_paths,
            temporal_context,
            limit=top_k,
        )
        stage_times["scoring_rerank_ms"] = round(
            (time.perf_counter() - scoring_started) * 1000, 2
        )

        temporal_status = self._evaluate_temporal_coverage(
            top_paths,
            temporal_context,
        )

        # Do not let the synthesizer invent a comparison when the retrieved
        # evidence does not cover the requested years. Returning the paths is
        # intentional: the UI can still show the available evidence while the
        # answer itself clearly refuses the unsupported comparison.
        if temporal_status["status"] == "INSUFFICIENT_EVIDENCE":
            return self._temporal_guard_response(
                user_query,
                top_paths,
                all_anchors,
                temporal_status,
                retrieval=retrieval,
            )

        # Step 6: Evidence Collection
        all_evidence = []
        for path in top_paths:
            all_evidence.extend(
                [e for e in path.evidence if e and len(e) > 10]
            )

        # Step 7: LLM Synthesis
        synthesis_started = time.perf_counter()
        structured_report = self._synthesize_report(
            user_query,
            top_paths,
            intent_id,
            vector_hits=vector_retrieval.get("hits", []),
        )
        structured_report = self._canonicalize_report_citations(
            structured_report,
            top_paths,
        )
        stage_times["llm_synthesis_ms"] = round(
            (time.perf_counter() - synthesis_started) * 1000, 2
        )
        answer = structured_report.get("narrative", "")
        insufficiency_text = str(
            structured_report.get("executive_summary", "")
        ).strip().lower()
        if re.search(
            r"^(?:the (?:provided )?.{0,40}(?:evidence|filing)|this filing).{0,100}"
            r"(?:does not (?:contain|disclose|provide|include)|"
            r"cannot (?:establish|determine))",
            insufficiency_text,
        ):
            answer = "[INSUFFICIENT EVIDENCE] " + answer
            structured_report["support_status"] = "INSUFFICIENT_EVIDENCE"
        grounding = self._validate_report_grounding(
            answer,
            top_paths,
            structured_report=structured_report,
        )
        llm_route = self._llm_route_metadata()
        if grounding["status"] != "VERIFIED":
            logger.warning(
                "Discarding ungrounded synthesis: status=%s unknown_ids=%s "
                "unknown_pages=%s unknown_years=%s",
                grounding["status"],
                grounding["unknown_evidence_ids"],
                grounding["unknown_pages"],
                grounding["unknown_years"],
            )
            answer = self._grounding_failure_response(grounding, top_paths)
            structured_report = self._grounding_failure_report(grounding, top_paths)

        return {
            "query": user_query,
            "intent": intent_id,
            "intent_display": intent_sig.display_name,
            "answer": answer,
            "paths": [self._serialize_path(p) for p in top_paths],
            "evidence_sentences": all_evidence[:20],
            "structured_report": structured_report,
            "metadata": {
                "total_candidates": len(candidate_paths),
                "raw_candidates": raw_candidate_count,
                "deduplicated_candidates": len(candidate_paths),
                "top_paths": len(top_paths),
                "anchors_used": all_anchors,
                "avg_score": round(np.mean([p.aggregate_score for p in top_paths]), 4) if top_paths else 0,
                "temporal": temporal_status,
                "grounding": grounding,
                "llm": llm_route,
                "source_filing": source_filing,
                "retrieval": retrieval,
                "retrieval_mode_requested": requested_retrieval_mode,
                "retrieval_mode_selected": retrieval_mode,
                "latency_ms": {
                    **stage_times,
                    "total_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            },
        }

    @staticmethod
    def _resolve_retrieval_mode(
        user_query: str,
        requested_mode: str,
        target_metric: Optional[str] = None,
    ) -> str:
        """Choose graph-only retrieval for exact structured questions.

        Vector retrieval remains useful for exploratory natural-language
        questions, while explicit metrics and ontology relations are already
        resolved deterministically and do not benefit from a Chroma round trip.
        """
        if requested_mode in {"graph", "hybrid"}:
            return requested_mode
        query_upper = str(user_query or "").upper().replace("-", "_")
        causal_or_exploratory = re.search(
            r"\b(affect|impact|cause|risk|control|constraint|exposure|supply chain|why|how)\b",
            str(user_query or ""),
            re.IGNORECASE,
        )
        if (target_metric and not causal_or_exploratory) or any(rel in query_upper for rel in VALID_RELATIONS):
            return "graph"
        return "hybrid"

    @staticmethod
    def _apply_vector_fusion(
        paths: List[CausalPath],
        vector_retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fuse semantic page hits with graph path scores.

        The current evidence model joins graph edges to filing/page pairs, so
        page-level fusion is safer than pretending that arbitrary text chunks
        are equivalent to causal claims. Vector hits affect ranking only when
        their filing and page metadata are present.
        """
        status = vector_retrieval.get("status", "NOT_REQUESTED")
        hits = vector_retrieval.get("hits", []) or []
        diagnostics: Dict[str, Any] = {
            "mode": "GRAPH_ONLY" if status == "NOT_REQUESTED" else "HYBRID_DEGRADED",
            "vector_status": status,
            "vector_hits": len(hits),
            "matched_paths": 0,
            "fusion": "none",
            "collection": vector_retrieval.get("collection"),
        }
        if status == "NOT_REQUESTED" or not hits or not paths:
            return diagnostics

        page_scores: Dict[Tuple[str, int], float] = {}
        for hit in hits:
            metadata = hit.get("metadata") or {}
            filing = metadata.get("source_filing") or metadata.get("doc_id")
            page = metadata.get("page")
            try:
                page = int(page)
            except (TypeError, ValueError):
                continue
            if not filing or page <= 0:
                continue
            rank = max(int(hit.get("rank", 1)), 1)
            page_scores[(str(filing), page)] = max(
                page_scores.get((str(filing), page), 0.0),
                1.0 / rank,
            )

        if not page_scores:
            diagnostics["mode"] = "HYBRID_CONTEXT_ONLY"
            diagnostics["fusion"] = "vector_context_only"
            return diagnostics

        matched = 0
        for path in paths:
            vector_score = 0.0
            for filing, page in zip(path.filings, path.pages):
                vector_score = max(
                    vector_score,
                    page_scores.get((str(filing), int(page)), 0.0),
                )
            path.score_breakdown["vector_page_rank"] = round(vector_score, 4)
            if vector_score > 0:
                matched += 1
            path.aggregate_score = round(
                0.75 * path.aggregate_score + 0.25 * vector_score,
                4,
            )

        diagnostics.update({
            "mode": "HYBRID" if matched else "HYBRID_CONTEXT_ONLY",
            "matched_paths": matched,
            "fusion": "0.75_graph_score + 0.25_vector_page_rank",
        })
        return diagnostics

    def _llm_route_metadata(self) -> Dict[str, Any]:
        """Expose provider routing without exposing credentials."""
        llm = getattr(self, "llm", None)
        if llm is None:
            return {"status": "NOT_CONFIGURED"}
        return {
            "configured_provider": getattr(llm, "provider", None),
            "configured_model": getattr(llm, "default_model", None),
            "success_provider": getattr(llm, "last_success_provider", None),
            "success_model": getattr(llm, "last_success_model", None),
            "fallback_configured": bool(os.getenv("LLM_FALLBACK_PROVIDERS", "").strip()),
        }

    @staticmethod
    def _build_temporal_context(
        user_query: str,
        year_start: Optional[int],
        year_end: Optional[int],
    ) -> Dict[str, Any]:
        """Resolve temporal intent and bounds before graph retrieval.

        The classifier intentionally has broad financial-impact categories,
        so temporal correctness is derived from the structured parser and the
        explicit API window rather than from the intent label alone.
        """
        parsed = parse_query(user_query)
        explicit_years = sorted({int(y) for y in re.findall(r"20\d{2}", user_query)})

        resolved_start = year_start if year_start is not None else parsed.fiscal_year_start
        resolved_end = year_end if year_end is not None else parsed.fiscal_year_end
        explicit_range = (
            resolved_start is not None
            and resolved_end is not None
            and resolved_start != resolved_end
        )
        temporal_requested = bool(parsed.temporal_required or explicit_range)
        require_multi_year = bool(parsed.require_multi_year or explicit_range)

        requested_years: List[int] = []
        if explicit_range:
            # Every explicitly named comparison year must be represented. A
            # question naming 2023, 2024 and 2025 must not pass with endpoints
            # only. A natural-language range retains endpoint semantics.
            requested_years = explicit_years or [resolved_start, resolved_end]

        return {
            "requested": temporal_requested,
            "require_multi_year": require_multi_year,
            "year_start": resolved_start,
            "year_end": resolved_end,
            "requested_years": requested_years,
            "question_years": explicit_years,
            "minimum_distinct_years": 2 if require_multi_year else 0,
        }

    @staticmethod
    def _evaluate_temporal_coverage(
        paths: List[CausalPath],
        temporal_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a machine-readable temporal evidence verdict."""
        covered_years = sorted({
            int(year)
            for path in paths
            for year in path.years
            if isinstance(year, (int, np.integer)) and int(year) > 0
        })

        if not temporal_context["requested"]:
            status = "NOT_REQUESTED"
        else:
            endpoint_years = set(temporal_context["requested_years"])
            enough_distinct_years = (
                len(covered_years) >= temporal_context["minimum_distinct_years"]
            )
            endpoints_covered = not endpoint_years or endpoint_years.issubset(covered_years)
            status = "SUFFICIENT" if enough_distinct_years and endpoints_covered else "INSUFFICIENT_EVIDENCE"

        return {
            "status": status,
            "requested": temporal_context["requested"],
            "require_multi_year": temporal_context["require_multi_year"],
            "requested_years": temporal_context["requested_years"],
            "covered_years": covered_years,
            "missing_requested_years": [
                year for year in temporal_context["requested_years"]
                if year not in covered_years
            ],
            "minimum_distinct_years": temporal_context["minimum_distinct_years"],
        }

    @classmethod
    def _select_temporal_paths(
        cls,
        ranked_paths: List[CausalPath],
        temporal_context: Dict[str, Any],
        limit: int,
    ) -> List[CausalPath]:
        """Select ranked paths while preserving requested-year coverage.

        A global reranker can otherwise fill Top-K with the newest filing,
        even when the query explicitly asks for a multi-year comparison. This
        selector reserves the best available strict path for each requested
        endpoint year, then fills remaining slots by the normal ranking.
        It never creates a path or relaxes the EvidenceClaim filter.
        """
        ranked_paths = list(ranked_paths or [])
        if limit <= 0 or not ranked_paths:
            return []
        if not temporal_context.get("require_multi_year"):
            return ranked_paths[:limit]

        requested_years = list(dict.fromkeys(temporal_context.get("requested_years") or []))
        selected: List[CausalPath] = []
        selected_ids: Set[int] = set()
        for year in requested_years:
            matches = [path for path in ranked_paths if year in path.years]
            if not matches:
                continue
            best = min(matches, key=cls._path_sort_key)
            marker = id(best)
            if marker not in selected_ids:
                selected.append(best)
                selected_ids.add(marker)

        for path in ranked_paths:
            if len(selected) >= limit:
                break
            if id(path) not in selected_ids:
                selected.append(path)
                selected_ids.add(id(path))
        return selected[:limit]

    def _temporal_guard_response(
        self,
        query: str,
        paths: List[CausalPath],
        anchors: List[str],
        temporal_status: Dict[str, Any],
        retrieval: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Return evidence without synthesizing an unsupported time trend."""
        covered = ", ".join(f"FY{year}" for year in temporal_status["covered_years"]) or "none"
        requested = ", ".join(
            f"FY{year}" for year in temporal_status["requested_years"]
        ) or "at least two fiscal years"
        evidence = [
            evidence
            for path in paths
            for evidence in path.evidence
            if evidence and len(evidence) > 10
        ][:20]
        return {
            "query": query,
            "intent": "TEMPORAL_GUARD",
            "intent_display": "Insufficient Temporal Evidence",
            "answer": (
                "[INSUFFICIENT TEMPORAL EVIDENCE] The requested comparison "
                f"requires {requested}, but the retrieved evidence covers {covered}. "
                "No cross-year trend or change conclusion was generated. "
                "Add the missing filing(s) or narrow the question to the available year."
            ),
            "structured_report": {
                "format": "evidence_claim_v1",
                "status": "INSUFFICIENT_TEMPORAL_EVIDENCE",
                "executive_summary": "The requested time comparison is not supported.",
                "claims": [],
                "evidence_quality": "Retrieved evidence does not cover all requested years.",
                "limitations": "No cross-year trend or change conclusion was generated.",
            },
            "paths": [self._serialize_path(path) for path in paths],
            "evidence_sentences": evidence,
            "metadata": {
                "total_candidates": len(paths),
                "top_paths": len(paths),
                "anchors_used": anchors,
                "temporal": temporal_status,
                "llm": self._llm_route_metadata(),
                "grounding": {"status": "NOT_APPLICABLE"},
                "retrieval": retrieval or {"mode": "GRAPH_ONLY"},
            },
        }

    @staticmethod
    def _canonicalize_report_citations(
        structured_report: Optional[Dict[str, Any]],
        paths: List[CausalPath],
    ) -> Optional[Dict[str, Any]]:
        """Replace model-supplied pages/years with graph-backed metadata.

        The LLM is allowed to choose which retrieved EvidenceClaims support a
        statement, but it is not the source of truth for a Claim's page or
        fiscal year.  Models occasionally copy a nearby page from the
        context, producing a false citation mismatch even when the Claim ID is
        valid.  Canonicalizing from the retrieved paths preserves fail-closed
        behavior for unknown IDs while making valid citations deterministic.
        """
        if not structured_report or not paths:
            return structured_report

        evidence_index: Dict[str, set[Tuple[int, int]]] = {}
        for path in paths:
            for index, evidence_id in enumerate(path.evidence_ids):
                if not evidence_id or evidence_id == "?":
                    continue
                if index >= len(path.pages) or index >= len(path.years):
                    continue
                evidence_index.setdefault(str(evidence_id), set()).add(
                    (int(path.pages[index]), int(path.years[index]))
                )

        normalized_claims = []
        for claim in structured_report.get("claims", []) or []:
            if not isinstance(claim, dict):
                continue
            claim_ids = [
                str(value)
                for value in claim.get("evidence_claim_ids", []) or []
                if value
            ]
            canonical_pairs = {
                pair
                for claim_id in claim_ids
                for pair in evidence_index.get(claim_id, set())
            }
            if not canonical_pairs:
                # Drop empty/malformed claim objects from the presentation
                # layer.  Unknown IDs remain visible in the executive
                # summary and therefore still fail the global validator.
                continue
            claim["pages"] = sorted({page for page, _ in canonical_pairs})
            claim["fiscal_years"] = sorted({year for _, year in canonical_pairs})
            normalized_claims.append(claim)

        # Some provider responses omit the claims array even though the
        # executive summary contains valid inline EvidenceClaim citations.
        # Recover those IDs from the retrieved graph and keep the report
        # contract strict; unknown inline IDs are not recovered.
        summary_text = str(structured_report.get("executive_summary", ""))
        inline_ids = [
            value
            for value in re.findall(
                r"\b[A-Za-z0-9][A-Za-z0-9_-]*_claim\b",
                summary_text,
            )
            if value in evidence_index
        ]
        if not normalized_claims and inline_ids:
            for claim_id in dict.fromkeys(inline_ids):
                pairs = evidence_index[claim_id]
                normalized_claims.append(
                    {
                        "statement": summary_text,
                        "evidence_claim_ids": [claim_id],
                        "pages": sorted({page for page, _ in pairs}),
                        "fiscal_years": sorted({year for _, year in pairs}),
                        "support_level": "LIMITED",
                    }
                )
        structured_report["claims"] = normalized_claims

        # The narrative is generated from the structured claims, so rebuild it
        # after canonicalization and remove stale model-supplied page values.
        narrative_parts = []
        summary = str(structured_report.get("executive_summary", "")).strip()
        if summary:
            narrative_parts.append(f"Executive Summary\n{summary}")
        for claim in normalized_claims:
            if not isinstance(claim, dict) or not claim.get("statement"):
                continue
            citation_parts = [
                str(value)
                for value in claim.get("evidence_claim_ids", []) or []
                if value
            ]
            pages = claim.get("pages", []) or []
            if pages:
                citation_parts.append(
                    "p." + ", p.".join(str(page) for page in pages)
                )
            narrative_parts.append(
                f"{claim['statement']}\n[EvidenceClaim: "
                f"{'; '.join(citation_parts) or 'missing citation'}]"
            )
        quality = str(structured_report.get("evidence_quality", "")).strip()
        limitations = str(structured_report.get("limitations", "")).strip()
        if quality:
            narrative_parts.append(f"Evidence Quality\n{quality}")
        if limitations:
            narrative_parts.append(f"Limitations\n{limitations}")
        structured_report["narrative"] = "\n\n".join(narrative_parts)
        return structured_report

    @staticmethod
    def _validate_report_grounding(
        answer: str,
        paths: List[CausalPath],
        structured_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate report citations against the exact retrieved evidence.

        This is deliberately conservative. It does not attempt to prove that
        every English sentence is factually correct, but it rejects unknown
        claim IDs, pages, years, and reports with no traceable citation at all.
        The remaining semantic claim verification is an evaluation concern,
        while these structural checks are safe to enforce at runtime.
        """
        if not paths:
            return {"status": "NOT_APPLICABLE"}

        valid_ids = {
            evidence_id
            for path in paths
            for evidence_id in path.evidence_ids
            if evidence_id and evidence_id != "?"
        }
        valid_pages = {
            int(page)
            for path in paths
            for page in path.pages
            if isinstance(page, (int, np.integer)) and int(page) > 0
        }
        valid_years = {
            int(year)
            for path in paths
            for year in path.years
            if isinstance(year, (int, np.integer)) and int(year) > 0
        }

        text = answer or ""
        cited_ids = set(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]*_claim\b", text))
        cited_pages = {
            int(page)
            for page in re.findall(r"\b(?:p|page)\s*\.?\s*(\d+)\b", text, re.IGNORECASE)
        }
        # Years inside the prose can describe the question or a disclosed
        # metric period. The structured claim contract below is authoritative;
        # treating every narrative year as a citation caused valid multi-year
        # answers to fail when an executive summary named all requested years.
        cited_years = set()
        claim_citation_mismatches = []

        # Structured citations are authoritative. The narrative is only the
        # presentation layer and does not need to repeat every identifier.
        if structured_report:
            for claim in structured_report.get("claims", []) or []:
                claim_ids = {
                    str(value)
                    for value in claim.get("evidence_claim_ids", []) or []
                    if value
                }
                claim_pages = {
                    int(value)
                    for value in claim.get("pages", []) or []
                    if str(value).isdigit()
                }
                claim_years = {
                    int(value)
                    for value in claim.get("fiscal_years", []) or []
                    if str(value).isdigit()
                }
                cited_ids.update(claim_ids)
                cited_pages.update(claim_pages)
                cited_years.update(claim_years)
                if not claim_ids or not claim_pages:
                    claim_citation_mismatches.append({
                        "reason": "CLAIM_MISSING_ID_OR_PAGE",
                        "claim_ids": sorted(claim_ids),
                        "pages": sorted(claim_pages),
                    })
                    continue
                for claim_id in claim_ids:
                    matching_hops = [
                        (path.pages[index], path.years[index])
                        for path in paths
                        for index, evidence_id in enumerate(path.evidence_ids)
                        if evidence_id == claim_id
                    ]
                    if not any(
                        page in claim_pages and (not claim_years or year in claim_years)
                        for page, year in matching_hops
                    ):
                        claim_citation_mismatches.append({
                            "reason": "CLAIM_ID_PAGE_YEAR_MISMATCH",
                            "claim_id": claim_id,
                            "pages": sorted(claim_pages),
                            "years": sorted(claim_years),
                        })

        cited_ids = sorted(cited_ids)
        cited_pages = sorted(cited_pages)
        cited_years = sorted(cited_years)

        unknown_ids = sorted(set(cited_ids) - valid_ids)
        unknown_pages = sorted(set(cited_pages) - valid_pages)
        unknown_years = sorted(set(cited_years) - valid_years)
        missing_required_citation = not cited_ids or not cited_pages

        if unknown_ids or unknown_pages or unknown_years or claim_citation_mismatches:
            status = "UNSUPPORTED"
        elif missing_required_citation:
            status = "PARTIALLY_VERIFIED"
        else:
            status = "VERIFIED"

        return {
            "status": status,
            "cited_evidence_ids": cited_ids,
            "cited_pages": cited_pages,
            "cited_years": cited_years,
            "unknown_evidence_ids": unknown_ids,
            "unknown_pages": unknown_pages,
            "unknown_years": unknown_years,
            "claim_citation_mismatches": claim_citation_mismatches,
            "available_evidence_ids": sorted(valid_ids),
            "available_pages": sorted(valid_pages),
            "available_years": sorted(valid_years),
        }

    @staticmethod
    def _grounding_failure_response(
        grounding: Dict[str, Any],
        paths: List[CausalPath],
    ) -> str:
        """Fail closed while still returning a deterministic evidence trace."""
        reasons = []
        if grounding.get("unknown_evidence_ids"):
            reasons.append("unknown EvidenceClaim ID")
        if grounding.get("unknown_pages"):
            reasons.append("page not present in retrieved evidence")
        if grounding.get("unknown_years"):
            reasons.append("year not present in retrieved evidence")
        if grounding.get("claim_citation_mismatches"):
            reasons.append("EvidenceClaim ID does not match its cited page/year")
        if not grounding.get("cited_evidence_ids"):
            reasons.append("no EvidenceClaim citation")
        if not grounding.get("cited_pages"):
            reasons.append("no page citation")
        reason_text = "; ".join(reasons) or "citation validation failed"
        verified_lines = []
        for path in paths[:5]:
            for index, evidence in enumerate(path.evidence):
                if not evidence or not evidence.strip():
                    continue
                page = path.pages[index] if index < len(path.pages) else "?"
                year = path.years[index] if index < len(path.years) else "?"
                claim_id = (
                    path.evidence_ids[index]
                    if index < len(path.evidence_ids)
                    else "?"
                )
                verified_lines.append(
                    f"- {evidence.strip()} "
                    f"[EvidenceClaim: {claim_id}; FY{year}; p.{page}]"
                )

        evidence_trace = "\n".join(verified_lines) or "- No verified evidence sentence was retrieved."
        return (
            "[GROUNDING FAILURE] The generated report was withheld because its "
            f"citations could not be fully mapped to the retrieved evidence ({reason_text}). "
            "No unsupported LLM statement is returned.\n\n"
            "Verified evidence trace:\n"
            f"{evidence_trace}\n\n"
            "Interpretation is intentionally limited to these cited sentences; "
            "add more filings or improve retrieval before making a broader conclusion."
        )

    # ── LLM Synthesis ──

    @staticmethod
    def _grounding_failure_report(
        grounding: Dict[str, Any],
        paths: List[CausalPath],
    ) -> Dict[str, Any]:
        """Return a deterministic structured trace when synthesis is rejected."""
        claims = []
        for path in paths[:5]:
            for index, evidence in enumerate(path.evidence):
                if not evidence or not evidence.strip():
                    continue
                claims.append(
                    {
                        "statement": evidence.strip(),
                        "evidence_claim_ids": [
                            path.evidence_ids[index]
                            if index < len(path.evidence_ids)
                            else ""
                        ],
                        "pages": [
                            path.pages[index]
                            if index < len(path.pages)
                            else None
                        ],
                        "fiscal_years": [
                            path.years[index]
                            if index < len(path.years)
                            else None
                        ],
                        "support_level": "VERIFIED_TRACE",
                    }
                )
        return {
            "format": "evidence_claim_v1",
            "status": "GROUNDING_FAILURE",
            "executive_summary": "LLM synthesis withheld; only verified evidence is returned.",
            "claims": claims,
            "evidence_quality": "The generated narrative failed structural citation validation.",
            "limitations": "Interpretation is limited to the retrieved EvidenceClaim records.",
            "grounding": grounding,
            "narrative": "",
        }

    def _synthesize_report(
        self,
        query: str,
        paths: List[CausalPath],
        intent: str,
        vector_hits: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate a professional financial analysis report from causal paths."""
        if not paths:
            return {
                "format": "evidence_claim_v1",
                "status": "INSUFFICIENT_EVIDENCE",
                "executive_summary": "No causal pathways found.",
                "claims": [],
                "evidence_quality": "No verified causal path was retrieved.",
                "limitations": "The current filing scope cannot support a conclusion.",
                "narrative": "[INSUFFICIENT EVIDENCE] No causal pathways found.",
            }

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
                causal_form = path.causal_forms[j] if j < len(path.causal_forms) else "UNMODELED_DIRECT"
                rel = path.relationships[j] if j < len(path.relationships) else "RELATES_TO"
                desc += (
                    f"Link {j+1}: {path.nodes[j]} {rel} {path.nodes[j+1]}\n"
                    f"  Strength: {strength} | Form: {causal_form} | FY{year} | "
                    f"p.{page} | EvidenceClaim: {claim_id}\n"
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

        # Vector hits are used for recall and path fusion, but they are not
        # report evidence.  Passing their raw page snippets to the synthesis
        # model creates a subtle citation hazard: the model may copy a vector
        # hit's page into a structured claim even though that page is not part
        # of the retrieved EvidenceClaim chain.  Keep the synthesis context
        # closed over graph-backed evidence only; this makes the runtime
        # citation contract enforceable instead of relying on a prompt warning.

        if not self._has_llm:
            return self._trace_report(paths, intent, context)
            return (
                f"[GRAPH TRACE — No LLM Synthesis Available]\n\n"
                f"Intent: {intent}\n\n{context}"
            )

        prompt = f"""Below are causal evidence chains extracted from NVIDIA's SEC filings.

[EVIDENCE DATA — THE ONLY CITABLE SOURCE]:
{context}

[USER QUERY]:
{query}

[CRITICAL INSTRUCTIONS]:
1. ANSWER THE EXACT QUERY. Do not drift to adjacent topics.
2. Return ONLY valid JSON. Do not use Markdown fences or any text outside the JSON object.
3. For each causal link: check if the evidence EXPLICITLY states causation or merely mentions both entities.
   Label in prose: "...this link is well-supported by direct evidence..." NOT "[CONFIRMED]".
   A REPORTS_METRIC link is a quantitative disclosure, not a causal link. Preserve its
   reported value, unit, and period exactly and do not describe it as causing an outcome.
4. Every claim MUST reference an EvidenceClaim ID and page copied exactly from the evidence data.
   Only use years and pages that appear in the evidence data. Never introduce prior-year
    values, general knowledge, or uncited pages. If the evidence cannot support a claim,
   state that it is not established by this filing.
5. Use exactly this JSON shape:
   {{
     "executive_summary": "concise answer to the user",
     "claims": [
       {{
         "statement": "one supported analytical statement",
         "evidence_claim_ids": ["exact EvidenceClaim ID(s)"],
         "pages": [integer page number(s)],
         "fiscal_years": [integer fiscal year(s)],
         "support_level": "DIRECT|INDIRECT|LIMITED"
       }}
     ],
     "evidence_quality": "explicit vs implied support and weaknesses",
     "limitations": "what this filing cannot establish"
   }}
6. Every claim must include at least one exact EvidenceClaim ID and page from the evidence data.
5. Use exactly this JSON shape (the JSON contract overrides any legacy prose format):
   {{
     "executive_summary": "concise answer to the user",
     "claims": [
       {{
         "statement": "one supported analytical statement",
         "evidence_claim_ids": ["exact EvidenceClaim ID(s)"],
         "pages": [integer page number(s)],
         "fiscal_years": [integer fiscal year(s)],
         "support_level": "DIRECT|INDIRECT|LIMITED"
       }}
     ],
     "evidence_quality": "explicit vs implied support and weaknesses",
     "limitations": "what this filing cannot establish"
   }}
6. Every claim must include at least one exact EvidenceClaim ID and page from the evidence data.
5. Follow the output format: Executive Summary → Analysis → Evidence Quality → Limitations.
6. Be honest about weak links — that is valuable analysis.

Now generate the analysis report:"""

        # The provider object owns the fallback policy.  This keeps DeepSeek
        # as the selected route unless the operator explicitly configures
        # LLM_FALLBACK_PROVIDERS, and prevents hidden vendor switching.
        result = self.llm.chat_with_fallback(
            prompt=prompt,
            system_prompt=self.REPORT_SYSTEM_PROMPT,
            model=self.model_name,
            temperature=0.3,
            max_tokens=3000,
        )

        if result is None:
            return self._trace_report(
                paths,
                intent,
                f"[SYNTHESIS ERROR: Configured LLM route unavailable]\n\n{context}",
                status="SYNTHESIS_ERROR",
            )
        return self._parse_structured_report(result)

    @staticmethod
    def _trace_report(
        paths: List[CausalPath],
        intent: str,
        context: str,
        status: str = "TRACE_ONLY",
    ) -> Dict[str, Any]:
        """Create a citation-complete report when synthesis is unavailable."""
        claims = []
        for path in paths[:5]:
            for index, evidence in enumerate(path.evidence):
                if not evidence or not evidence.strip():
                    continue
                claims.append(
                    {
                        "statement": evidence.strip(),
                        "evidence_claim_ids": [path.evidence_ids[index]],
                        "pages": [path.pages[index]],
                        "fiscal_years": [path.years[index]],
                        "support_level": "VERIFIED_TRACE",
                    }
                )
        return {
            "format": "evidence_claim_v1",
            "status": status,
            "executive_summary": f"Graph evidence trace for intent {intent}.",
            "claims": claims,
            "evidence_quality": "No free-form synthesis was trusted; claims are verbatim evidence excerpts.",
            "limitations": "Interpretation is limited to the retrieved EvidenceClaim records.",
            "narrative": context,
        }

    @staticmethod
    def _parse_structured_report(content: str) -> Dict[str, Any]:
        """Parse and normalize the LLM EvidenceClaim citation contract."""
        raw = (content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {
                "format": "evidence_claim_v1",
                "status": "INVALID_JSON",
                "executive_summary": "",
                "claims": [],
                "evidence_quality": "The LLM did not follow the structured citation contract.",
                "limitations": "The generated report was not accepted as a structured report.",
                "narrative": content,
            }
        if not isinstance(parsed, dict):
            return {
                "format": "evidence_claim_v1",
                "status": "INVALID_SCHEMA",
                "executive_summary": "",
                "claims": [],
                "evidence_quality": "The LLM returned a non-object JSON value.",
                "limitations": "The generated report was not accepted as a structured report.",
                "narrative": content,
            }

        normalized_claims = []
        for item in parsed.get("claims", []) or []:
            if not isinstance(item, dict):
                continue
            ids = item.get("evidence_claim_ids", item.get("evidence_ids", []))
            pages = item.get("pages", [])
            years = item.get("fiscal_years", item.get("years", []))
            if isinstance(ids, str):
                ids = [ids]
            if isinstance(pages, (int, str)):
                pages = [pages]
            if isinstance(years, (int, str)):
                years = [years]
            normalized_claims.append(
                {
                    "statement": str(item.get("statement", "")).strip(),
                    "evidence_claim_ids": [str(value) for value in ids if value],
                    "pages": [int(value) for value in pages if str(value).isdigit()],
                    "fiscal_years": [int(value) for value in years if str(value).isdigit()],
                    "support_level": str(item.get("support_level", "LIMITED")).upper(),
                }
            )

        summary = str(parsed.get("executive_summary", "")).strip()
        quality = str(parsed.get("evidence_quality", "")).strip()
        limitations = str(parsed.get("limitations", "")).strip()
        narrative_parts = []
        if summary:
            narrative_parts.append(f"Executive Summary\n{summary}")
        for claim in normalized_claims:
            if not claim["statement"]:
                continue
            citation_parts = []
            for claim_id in claim["evidence_claim_ids"]:
                citation_parts.append(claim_id)
            if claim["pages"]:
                citation_parts.append("p." + ", p.".join(str(page) for page in claim["pages"]))
            narrative_parts.append(
                f"{claim['statement']}\n[EvidenceClaim: {'; '.join(citation_parts) or 'missing citation'}]"
            )
        if quality:
            narrative_parts.append(f"Evidence Quality\n{quality}")
        if limitations:
            narrative_parts.append(f"Limitations\n{limitations}")
        return {
            "format": "evidence_claim_v1",
            "status": "GENERATED",
            "executive_summary": summary,
            "claims": normalized_claims,
            "evidence_quality": quality,
            "limitations": limitations,
            "narrative": "\n\n".join(narrative_parts),
        }

    # ── Helpers ──

    def _llm_extract_anchors(self, query: str) -> List[str]:
        """Use LLM to extract strategic entity keywords from query."""
        if not self._has_llm:
            return []
        cache_key = str(query or "").strip().lower()
        if cache_key in self._anchor_cache:
            return list(self._anchor_cache[cache_key])
        prompt = (
            "Extract 2-4 specific financial entities from this query. "
            "Use canonical UPPER_SNAKE_CASE names. "
            "Examples: CHIP_EXPORT_RESTRICTION, REVENUE, DATA_CENTER_MARKET, "
            "SUPPLY_CHAIN_DIVERSIFICATION. "
            "Return ONLY comma-separated list, no explanations.\n"
            f"Query: {query}"
        )
        content = self.llm.chat(
            prompt=prompt,
            model=self.llm.get_task_model("query"),
            temperature=0.0,
            max_tokens=100,
        )
        if content is None:
            return []
        anchors = [
            w.strip().upper().replace(" ", "_")
            for w in content.split(",")
            if len(w.strip()) > 2
        ]
        if len(self._anchor_cache) >= self._anchor_cache_limit:
            self._anchor_cache.pop(next(iter(self._anchor_cache)))
        self._anchor_cache[cache_key] = anchors
        return list(anchors)

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
            paths.sort(key=self._path_sort_key)
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")

        return paths

    @classmethod
    def _deduplicate_paths(cls, paths: List[CausalPath]) -> List[CausalPath]:
        """Collapse duplicate semantic paths while preserving evidence variants.

        Neo4j returns one path per relationship instance. In this graph, the
        same semantic chain can be supported by several EvidenceClaim nodes,
        which should not consume separate Top-K slots. The highest-scoring
        representative is kept and all distinct evidence variants are exposed
        in the response for provenance review.
        """
        grouped: Dict[Tuple, CausalPath] = {}
        ordered = sorted(paths, key=cls._path_sort_key)

        for path in ordered:
            key = path.semantic_key()
            representative = grouped.get(key)
            if representative is None:
                path.duplicate_count = 0
                path.evidence_variants = [[] for _ in path.relationships]
                grouped[key] = path
                representative = path

            representative.duplicate_count += 1
            for hop in range(len(representative.relationships)):
                variant = {
                    "evidence": path.evidence[hop] if hop < len(path.evidence) else "",
                    "page": path.pages[hop] if hop < len(path.pages) else 0,
                    "year": path.years[hop] if hop < len(path.years) else 0,
                    "evidence_id": path.evidence_ids[hop] if hop < len(path.evidence_ids) else "",
                    "filing": path.filings[hop] if hop < len(path.filings) else "",
                }
                if variant not in representative.evidence_variants[hop]:
                    representative.evidence_variants[hop].append(variant)

        unique_paths = sorted(grouped.values(), key=cls._path_sort_key)
        for index, path in enumerate(unique_paths):
            path.path_id = f"path_{index:03d}"
        return unique_paths

    @staticmethod
    def _path_sort_key(path: CausalPath) -> Tuple:
        """Stable ranking key: score first, then deterministic path identity."""
        return (
            -round(path.aggregate_score, 4),
            path.total_hops,
            tuple(path.nodes),
            tuple(path.relationships),
            tuple(path.years),
            tuple(path.evidence_ids),
        )

    def _fallback_response(
        self,
        query: str,
        anchors: List[str],
        temporal_context: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Generate a response when no paths are found."""
        temporal_status = None
        if temporal_context and temporal_context["requested"]:
            temporal_status = self._evaluate_temporal_coverage([], temporal_context)
            answer = (
                "[INSUFFICIENT TEMPORAL EVIDENCE] No verified causal path was "
                "retrieved for the requested time window. No cross-year trend "
                "or change conclusion was generated."
            )
        else:
            answer = (
                "[INSUFFICIENT EVIDENCE] The knowledge graph does not contain "
                "sufficient causal pathways to answer this query. The graph may "
                "need additional SEC filings or a broader set of financial documents "
                "to build the necessary relationship network.\n\n"
                f"Searched for: {', '.join(anchors) if anchors else 'all entities'}"
            )
        return {
            "query": query,
            "intent": "FALLBACK",
            "intent_display": "No Results",
            "answer": answer,
            "structured_report": {
                "format": "evidence_claim_v1",
                "status": "INSUFFICIENT_EVIDENCE",
                "executive_summary": answer,
                "claims": [],
                "evidence_quality": "No verified causal path was retrieved.",
                "limitations": "The current filing scope cannot support a conclusion.",
            },
            "paths": [],
            "evidence_sentences": [],
            "metadata": {
                "total_candidates": 0,
                "top_paths": 0,
                "anchors_used": anchors,
                "temporal": temporal_status,
                "llm": self._llm_route_metadata(),
            },
        }

    def _serialize_path(self, path: CausalPath) -> Dict:
        """Serialize a CausalPath to a JSON-safe dict."""
        return {
            "path_id": path.path_id,
            "fingerprint": path.fingerprint(),
            "semantic_fingerprint": path.semantic_fingerprint(),
            "nodes": path.nodes,
            "node_labels": path.node_labels,
            "relationships": path.relationships,
            "causal_strengths": path.causal_strengths,
            "evidence": path.evidence,
            "pages": path.pages,
            "years": path.years,
            "evidence_ids": path.evidence_ids,
            "filings": path.filings,
            "causal_forms": path.causal_forms,
            "total_hops": path.total_hops,
            "score": path.aggregate_score,
            "score_breakdown": path.score_breakdown,
            "duplicate_count": path.duplicate_count,
            "evidence_variants": path.evidence_variants,
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
