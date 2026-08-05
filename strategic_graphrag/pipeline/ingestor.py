# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Neo4j Graph Ingestor
========================================
Writes extracted (source, relation, target, evidence) triples
into Neo4j using the 6-layer temporal causal financial schema.

Key features:
  - Native Neo4j relationship types (no generic RELATION)
  - Evidence nodes (Document, Sentence) with provenance
  - Temporal anchoring (Year nodes, OBSERVED_IN/OCCURS_DURING)
  - Deduplication within and across filings
  - Batch write with transaction management
"""

import os
import hashlib
import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired
from dotenv import load_dotenv

from ..ontology.entity_registry import norm_id, is_banned
from ..ontology.relation_inference import VALID_RELATIONS, CAUSAL_STRENGTHS, ENTITY_CATEGORIES

load_dotenv()
logger = logging.getLogger("GraphIngestor")


class GraphIngestor:
    """
    Handles all Neo4j write operations for the Strategic-GraphRAG pipeline.
    Writes entities, relationships, evidence nodes, and temporal anchors.
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None

        # Track seen keys to deduplicate within a batch
        self._seen_keys: Set[Tuple[str, str, str, int, int]] = set()
        # Track entity write counts for stats
        self._entity_counts: Dict[str, int] = defaultdict(int)
        self._relation_counts: Dict[str, int] = defaultdict(int)

    def connect(self) -> bool:
        """Connect to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password),
                max_connection_lifetime=1800,
                keep_alive=True,
                connection_acquisition_timeout=15,
            )
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j: {self.uri}")
            return True
        except ServiceUnavailable as e:
            logger.error(f"Cannot connect to Neo4j: {e}")
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def _driver_alive(self) -> bool:
        """Check if driver connection is still alive."""
        if not self.driver:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self):
        if self.driver:
            self.driver.close()

    def reset_batch_state(self):
        """Reset deduplication state for a new filing."""
        self._seen_keys.clear()
        self._entity_counts.clear()
        self._relation_counts.clear()

    # ── Core Ingest Methods ──

    def ingest_triple(
        self,
        triple: Dict,
        filename: str,
        page: int,
        year: int,
        section: str = "",
    ) -> bool:
        """
        Write a single extracted triple to Neo4j as a native relationship.

        Creates/evaluates:
          - Source & Target nodes (with correct label)
          - Native Neo4j relationship (e.g., :CAUSES)
          - EvidenceSentence node with SUPPORTS edge
          - Document node with BELONGS_TO edge
          - Year node with OBSERVED_IN edge

        Returns True if successfully written.
        """
        s_name = str(triple.get("source", "")).strip()
        t_name = str(triple.get("target", "")).strip()
        rel_type = str(triple.get("relation", "")).strip().upper()
        s_cat = str(triple.get("source_category", "")).strip()
        t_cat = str(triple.get("target_category", "")).strip()
        cs = str(triple.get("causal_strength", "DISCLOSED_EXPOSURE")).upper()
        evidence = str(triple.get("evidence_sentence", ""))[:500]

        # Pre-checks
        if not s_name or not t_name or not rel_type:
            return False
        if rel_type not in VALID_RELATIONS:
            return False
        if s_cat not in ENTITY_CATEGORIES or t_cat not in ENTITY_CATEGORIES:
            return False
        if is_banned(s_name) or is_banned(t_name):
            return False

        # Dedup key
        s_id = norm_id(s_name)
        t_id = norm_id(t_name)
        dedup_key = (s_id, rel_type, t_id, year, page)
        if dedup_key in self._seen_keys:
            return False
        self._seen_keys.add(dedup_key)

        # Evidence ID
        eid = hashlib.md5(
            f"{s_id}|{rel_type}|{t_id}|{year}|p{page}".encode()
        ).hexdigest()[:16]
        es_id = eid + "_es"
        claim_id = eid + "_claim"

        # Cypher: Create nodes + native relationship + evidence chain
        cypher = f"""
        // 1. Create/merge source and target nodes
        MERGE (s:{s_cat} {{id: $s_id}})
        ON CREATE SET s.name = $s_name
        MERGE (t:{t_cat} {{id: $t_id}})
        ON CREATE SET t.name = $t_name

        // 2. Create native relationship
        MERGE (s)-[r:{rel_type} {{id: $eid}}]->(t)
        ON CREATE SET
            r.causal_strength = $cs,
            r.confidence = $conf,
            r.evidence_sentence = $ev,
            r.year = $yr,
            r.page = $pg,
            r.filing = $file,
            r.section = $sec,
            r.extraction_method = $method,
            r.source_category = $sc,
            r.target_category = $tc

        SET r.causal_strength = $cs,
            r.confidence = $conf,
            r.evidence_sentence = $ev,
            r.year = $yr,
            r.page = $pg,
            r.filing = $file,
            r.section = $sec,
            r.extraction_method = $method,
            r.source_category = $sc,
            r.target_category = $tc,
            r.evidence_id = $claim_id,
            r.source_filing = $file,
            r.source_page = $pg

        // 3. Temporal anchors
        MERGE (y:Year {{year: $yr}})
        MERGE (s)-[:OBSERVED_IN]->(y)

        // 4. Document evidence
        MERGE (d:Document {{doc_id: $doc_id}})
        ON CREATE SET
            d.filename = $file,
            d.fiscal_year = $yr
        MERGE (d)-[:REPORTS]->(y)

        // 5. Sentence and claim-level provenance.  Neo4j relationships cannot
        // have outgoing edges, so an EvidenceClaim node represents the exact
        // evidence supporting this specific graph edge.
        MERGE (es:Sentence {{id: $es_id}})
        ON CREATE SET
            es.text = $ev,
            es.page = $pg,
            es.section = $sec,
            es.doc_id = $doc_id
        MERGE (es)-[:SUPPORTS]->(s)
        MERGE (es)-[:BELONGS_TO]->(d)

        MERGE (claim:EvidenceClaim {{id: $claim_id}})
        SET claim.text = $ev,
            claim.page = $pg,
            claim.section = $sec,
            claim.doc_id = $doc_id,
            claim.fiscal_year = $yr,
            claim.relation_id = $eid,
            claim.relation_type = $rel_type,
            claim.source_id = $s_id,
            claim.target_id = $t_id,
            claim.verification_status = 'VERBATIM'
        MERGE (claim)-[:SUPPORTED_BY]->(es)
        MERGE (claim)-[:ABOUT_SOURCE]->(s)
        MERGE (claim)-[:ABOUT_TARGET]->(t)

        // Store evidence reference on the relationship
        SET r.evidence_id = $claim_id,
            r.source_filing = $file,
            r.source_page = $pg
        """
        try:
            with self.driver.session() as session:
                session.run(
                    cypher,
                    s_id=s_id, s_name=s_name,
                    t_id=t_id, t_name=t_name,
                    eid=eid, es_id=es_id, claim_id=claim_id,
                    rel_type=rel_type,
                    cs=cs, ev=evidence,
                    yr=year, pg=page, file=filename,
                    sec=section, doc_id=filename.replace(".pdf", ""),
                    sc=s_cat, tc=t_cat,
                    conf=self._calibrate_confidence(cs, "HYBRID", len(evidence), rel_type),
                    method="HYBRID",
                )
            self._entity_counts[s_cat] += 1
            self._entity_counts[t_cat] += 1
            self._relation_counts[rel_type] += 1
            return True
        except Neo4jError as e:
            logger.warning(f"Ingest error: {e}")
            return False

    def ingest_batch(
        self,
        triples: List[Dict],
        filename: str,
        pages: List[int],
        year: int,
        sections: List[str] = None,
    ) -> int:
        """
        Write a batch of triples from one filing to Neo4j using UNWIND for
        true batch insertion (single transaction, single network round-trip).

        Returns number of triples successfully ingested.
        """
        if not triples:
            return 0

        doc_id = filename.replace(".pdf", "")
        page0 = pages[0] if pages else 1
        sec0 = sections[0] if sections else ""

        # Build parameterized list — filter invalid triples upfront
        batch_params = []
        for i, triple in enumerate(triples):
            s_name = str(triple.get("source", "")).strip()
            t_name = str(triple.get("target", "")).strip()
            rel_type = str(triple.get("relation", "")).strip().upper()
            s_cat = str(triple.get("source_category", "")).strip()
            t_cat = str(triple.get("target_category", "")).strip()
            cs = str(triple.get("causal_strength", "DISCLOSED_EXPOSURE")).upper()
            evidence = str(triple.get("evidence_sentence", ""))[:500]

            if not s_name or not t_name or not rel_type:
                continue
            if rel_type not in VALID_RELATIONS:
                continue
            if s_cat not in ENTITY_CATEGORIES or t_cat not in ENTITY_CATEGORIES:
                continue
            if is_banned(s_name) or is_banned(t_name):
                continue

            s_id = norm_id(s_name)
            t_id = norm_id(t_name)
            pg = pages[min(i, len(pages) - 1)] if pages else page0
            sec = sections[min(i, len(sections) - 1)] if sections else sec0
            dedup_key = (s_id, rel_type, t_id, year, pg)
            if dedup_key in self._seen_keys:
                continue
            self._seen_keys.add(dedup_key)

            eid = hashlib.md5(
                f"{s_id}|{rel_type}|{t_id}|{year}|p{pg}".encode()
            ).hexdigest()[:16]
            es_id = eid + "_es"
            claim_id = eid + "_claim"

            conf = self._calibrate_confidence(cs, "HYBRID", len(evidence), rel_type)

            batch_params.append({
                "s_id": s_id, "s_name": s_name, "s_cat": s_cat,
                "t_id": t_id, "t_name": t_name, "t_cat": t_cat,
                "eid": eid, "es_id": es_id, "rel_type": rel_type,
                "claim_id": claim_id,
                "cs": cs, "ev": evidence, "conf": conf,
                "yr": year, "pg": pg, "file": filename,
                "sec": sec, "doc_id": doc_id,
            })

        if not batch_params:
            return 0

        # Single UNWIND batch — all triples in one transaction
        # Neo4j doesn't support dynamic labels in parameterized queries.
        # We build the Cypher with literal labels per triple category,
        # grouped by source_category/target_category pairs.
        from collections import defaultdict
        grouped = defaultdict(list)
        for bp in batch_params:
            key = (bp["s_cat"], bp["t_cat"], bp["rel_type"])
            grouped[key].append(bp)

        total_ingested = 0
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Reconnect driver if defunct (AuraDB free tier drops idle connections)
                if attempt > 0 or not self._driver_alive():
                    self.connect()
                with self.driver.session() as session:
                    for (s_cat, t_cat, rel_type), group in grouped.items():
                        cypher = f"""
                        UNWIND $batch AS row

                        MERGE (s:{s_cat} {{id: row.s_id}})
                        ON CREATE SET s.name = row.s_name

                        MERGE (t:{t_cat} {{id: row.t_id}})
                        ON CREATE SET t.name = row.t_name

                        MERGE (s)-[r:{rel_type} {{id: row.eid}}]->(t)
                        ON CREATE SET
                            r.causal_strength = row.cs,
                            r.confidence = row.conf,
                            r.evidence_sentence = row.ev,
                            r.year = row.yr,
                            r.page = row.pg,
                            r.filing = row.file,
                            r.section = row.sec,
                            r.extraction_method = 'HYBRID',
                            r.source_category = row.s_cat,
                            r.target_category = row.t_cat,
                            r.evidence_id = row.claim_id,
                            r.source_filing = row.file,
                            r.source_page = row.pg

                        MERGE (y:Year {{year: row.yr}})
                        MERGE (s)-[:OBSERVED_IN]->(y)

                        MERGE (d:Document {{doc_id: row.doc_id}})
                        ON CREATE SET
                            d.filename = row.file,
                            d.fiscal_year = row.yr
                        MERGE (d)-[:REPORTS]->(y)

                        MERGE (es:Sentence {{id: row.es_id}})
                        ON CREATE SET
                            es.text = row.ev,
                            es.page = row.pg,
                            es.section = row.sec,
                            es.doc_id = row.doc_id
                        MERGE (es)-[:SUPPORTS]->(s)
                        MERGE (es)-[:BELONGS_TO]->(d)

                        MERGE (claim:EvidenceClaim {{id: row.claim_id}})
                        SET claim.text = row.ev,
                            claim.page = row.pg,
                            claim.section = row.sec,
                            claim.doc_id = row.doc_id,
                            claim.fiscal_year = row.yr,
                            claim.relation_id = row.eid,
                            claim.relation_type = row.rel_type,
                            claim.source_id = row.s_id,
                            claim.target_id = row.t_id,
                            claim.verification_status = 'VERBATIM'
                        MERGE (claim)-[:SUPPORTED_BY]->(es)
                        MERGE (claim)-[:ABOUT_SOURCE]->(s)
                        MERGE (claim)-[:ABOUT_TARGET]->(t)

                        SET r.source_filing = row.file,
                            r.source_page = row.pg,
                            r.causal_strength = row.cs,
                            r.confidence = row.conf,
                            r.evidence_sentence = row.ev,
                            r.year = row.yr,
                            r.page = row.pg,
                            r.filing = row.file,
                            r.section = row.sec,
                            r.extraction_method = 'HYBRID',
                            r.source_category = row.s_cat,
                            r.target_category = row.t_cat,
                            r.evidence_id = row.claim_id
                        """
                        session.run(cypher, batch=group)
                        total_ingested += len(group)
                        self._entity_counts[s_cat] += len(group)
                        self._entity_counts[t_cat] += len(group)
                        self._relation_counts[rel_type] += len(group)
                break  # Success — exit retry loop

            except (Neo4jError, ServiceUnavailable, SessionExpired, OSError, Exception) as e:
                logger.warning(f"Batch ingest attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
                    try:
                        self.driver.verify_connectivity()
                    except Exception:
                        self.connect()
                else:
                    logger.error(f"Batch ingest failed after {max_retries} attempts")
                    return total_ingested  # Return whatever was ingested before failure

        return total_ingested

    def _calibrate_confidence(self, causal_strength: str, extraction_method: str,
    evidence_len: int, relation_type: str) -> float:
        """
        Calibrate per-triple confidence based on extraction quality signals.

        Weights reflect empirical reliability of extraction pathways:
        - LLM extractions get higher baseline (semantic understanding)
        - Rule extractions depend on causal_strength clarity
        - Direct causality from either engine = higher confidence
        - Short/vague evidence = penalty
        """
        base = 0.65  # Prior — slightly above neutral, below current hardcoded 0.7

        # Extraction method signal
        if extraction_method == "LLM_EXTRACTION":
            base += 0.10  # LLM semantic extraction more reliable
        elif extraction_method == "RULE_EXTRACTION":
            base += 0.00  # Rule extraction is pattern-based, lower baseline

        # Causal strength signal — explicit > implicit > speculative
        cs_boost = {
            "CONFIRMED_CAUSAL": 0.15,
            "STRONG_ASSOCIATION": 0.05,
            "WEAK_ASSOCIATION": 0.00,
            "DISCLOSED_ONLY": -0.05,
            "INFERRED": -0.10,
            "DIRECT_CAUSALITY": 0.15,
            "INDIRECT_CAUSALITY": 0.05,
            "RISK_ASSOCIATION": 0.00,
            "DISCLOSED_EXPOSURE": 0.02,
            "SPECULATIVE_RELATION": -0.10,
        }
        base += cs_boost.get(causal_strength, 0.0)

        # Evidence quality — too short means weak evidence
        if evidence_len < 50:
            base -= 0.08
        elif evidence_len > 150:
            base += 0.05  # Longer evidence generally more specific

        # Structural relationships (PRODUCES, OPERATES_IN) are less uncertain
        # than causal ones — but also less informative
        if relation_type in ("PRODUCES", "OPERATES_IN", "OBSERVED_IN"):
            base += 0.05  # Easy to verify, hard to get wrong

        # Clamp to [0.2, 0.95]
        return round(max(0.20, min(0.95, base)), 2)

    def get_stats(self) -> Dict:
        """Get current batch statistics."""
        return {
            "entities": dict(self._entity_counts),
            "relations": dict(self._relation_counts),
            "total_entities": sum(self._entity_counts.values()),
            "total_relations": sum(self._relation_counts.values()),
            "unique_triples": len(self._seen_keys),
        }

    def create_document_node(self, filename: str, doc_type: str = "10-K",
                              filing_date: str = "", fiscal_year: int = None,
                              total_pages: int = 0) -> bool:
        """Create/update a Document node for a source filing."""
        doc_id = filename.replace(".pdf", "")
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.filename = $fn,
                        d.doc_type = $dt,
                        d.filing_date = $fd,
                        d.fiscal_year = $fy,
                        d.total_pages = $tp
                    MERGE (y:Year {year: $fy})
                    MERGE (d)-[:REPORTS]->(y)
                """, doc_id=doc_id, fn=filename, dt=doc_type,
                   fd=filing_date, fy=fiscal_year, tp=total_pages)
            return True
        except Neo4jError as e:
            logger.warning(f"Document node error: {e}")
            return False

    # ── Maintenance ──

    def deduplicate_relations(self):
        """Remove duplicate relationships (same source, relation, target)."""
        cypher = """
        MATCH (s)-[r]->(t)
        WITH s, type(r) AS rel_type, t,
             coalesce(r.year, -1) AS year,
             coalesce(r.page, -1) AS page,
             coalesce(r.evidence_id, '') AS evidence_id,
             collect(r) AS rels
        WHERE size(rels) > 1
        UNWIND tail(rels) AS duplicate
        DELETE duplicate
        RETURN count(duplicate) AS removed
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher).single()
                if result:
                    logger.info(f"Removed {result['removed']} duplicate relationships")
        except Neo4jError as e:
            logger.warning(f"Dedup error: {e}")

    def enforce_hubness(self, max_out_edges: int = 30):
        """Prune only ungrounded extracted edges, never evidence-backed edges."""
        cypher = f"""
        MATCH (n)-[r]->()
        WHERE r.confidence IS NOT NULL AND r.evidence_id IS NULL
        WITH n, count(r) AS degree
        WHERE degree > {max_out_edges}
        MATCH (n)-[r]->(m)
        WHERE r.confidence IS NOT NULL AND r.evidence_id IS NULL
        WITH n, degree, r, m
        ORDER BY r.confidence ASC
        WITH n, degree, collect(r)[0..toInteger(degree - {max_out_edges})] AS to_delete
        UNWIND to_delete AS r
        DELETE r
        RETURN count(r) AS pruned
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher).single()
                if result:
                    logger.info(f"Hubness pruning: {result['pruned']} low-confidence edges removed")
        except Neo4jError as e:
            logger.warning(f"Hubness pruning error: {e}")

    @staticmethod
    def from_env() -> "GraphIngestor":
        """Factory method: create from .env configuration."""
        return GraphIngestor(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )
