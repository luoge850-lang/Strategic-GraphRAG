# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Neo4j Schema Manager
Complete 6-Layer Temporal Causal Financial Knowledge Graph

Usage:
    python -m strategic_graphrag.schema.manager --init
    python -m strategic_graphrag.schema.manager --reset
    python -m strategic_graphrag.schema.manager --verify
    python -m strategic_graphrag.schema.manager --stats
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SchemaManager")

# Canonical node labels for the current evidence-grounded schema.
# EvidenceClaim is an explicit node because Neo4j relationships cannot
# themselves own outgoing provenance edges.
NODE_LABELS = [
    "Company", "Product", "Market", "Region", "Regulation",  # Layer 1
    "RiskFactor",                                             # Layer 2
    "Strategy",                                               # Layer 3
    "FinancialMetric",                                        # Layer 4
    "Year", "Quarter", "Event",                              # Layer 5
    "Document", "Sentence", "EvidenceClaim",                # Layer 6
    "Mechanism",                                              # Bridge
]

# 21 Strict Financial Relationship Types
# v2.0: Evidence-Grounded Causal + Downgrade Relations
RELATIONSHIP_TYPES = {
    # Causal (require explicit evidence)
    "CAUSES": "Direct causal link (explicit verb required)",
    "TRIGGERS": "Triggers a mechanism",
    "AMPLIFIES": "Risk amplification",
    "INCREASES": "Increases a financial metric",
    "DECREASES": "Decreases a financial metric",
    "IMPLEMENTS": "Company implements strategy",
    "MITIGATES": "Strategy mitigates risk",
    "CONSTRAINS": "Regulation constrains entity",
    "EXPOSED_TO": "Entity exposed to risk",
    # Multi-hop intermediate (v2.0)
    "AFFECTS_SEGMENT": "Risk driver affects specific business segment",
    "CONSTRAINS_MARKET": "Restriction constrains market access",
    "EXPOSED_THROUGH": "Company exposed via business segment",
    "IMPACTS": "Driver impacts financial metric",
    "EXECUTES": "Company executes mitigation action",
    "ADDRESSES": "Mitigation action addresses risk driver",
    # Structural
    "OPERATES_IN": "Company operates in region/market",
    "PRODUCES": "Company produces product",
    "COMPETES_WITH": "Competitive relationship",
    "DEPENDS_ON": "Dependency relationship",
    "REGULATED_BY": "Regulated by a regulation",
    "SUPPLIES_TO": "Supplies to market/region",
    # Temporal
    "OCCURS_DURING": "Occurs during a time period",
    "PRECEDES": "Temporal ordering",
    "REPORTED_IN": "Reported in fiscal year",
    # Evidence
    "HAS_EVIDENCE": "Has sentence-level evidence",
    "BELONGS_TO": "Sentence belongs to document",
    "SUPPORTS": "Evidence supports entity",
    "REPORTS": "Document reports a fiscal year",
    "SUPPORTED_BY": "Evidence claim is supported by a sentence",
    "ABOUT_SOURCE": "Evidence claim identifies the source entity",
    "ABOUT_TARGET": "Evidence claim identifies the target entity",
    # Downgrade relations (v2.0: weak/uncertain signals)
    "DISCLOSES": "Document discloses entity (no causal claim)",
    "MENTIONS": "Document mentions entity",
    "POSSIBLE_RELATION": "Possible but unverified relationship",
}

CAUSAL_STRENGTHS = [
    "DIRECT_CAUSALITY", "INDIRECT_CAUSALITY",
    "RISK_ASSOCIATION", "SPECULATIVE_RELATION", "DISCLOSED_EXPOSURE",
]

FINANCIAL_METRICS_REGISTRY = {
    "REVENUE": ("REVENUE", "POSITIVE"),
    "GROSS_MARGIN": ("PROFIT", "POSITIVE"),
    "OPERATING_MARGIN": ("PROFIT", "POSITIVE"),
    "NET_INCOME": ("PROFIT", "POSITIVE"),
    "CASH_FLOW": ("CASH_FLOW", "POSITIVE"),
    "FREE_CASH_FLOW": ("CASH_FLOW", "POSITIVE"),
    "EARNINGS_PER_SHARE": ("VALUATION", "POSITIVE"),
    "MARKET_VALUE": ("VALUATION", "POSITIVE"),
    "OPERATING_COST": ("COST", "NEGATIVE"),
    "COST_OF_REVENUE": ("COST", "NEGATIVE"),
    "R_AND_D_EXPENSE": ("COST", "NEGATIVE"),
    "CAPEX": ("COST", "NEGATIVE"),
}


class SchemaManager:
    """Programmatic Neo4j schema lifecycle manager."""

    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None

    def connect(self) -> bool:
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j: {self.uri}")
            return True
        except ServiceUnavailable as e:
            logger.error(f"Cannot connect to Neo4j: {e}")
            return False

    def close(self):
        if self.driver:
            self.driver.close()

    def _run(self, cypher, **params):
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, **params)]

    def _exec_many(self, statements: List[str]) -> int:
        count = 0
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                self._run(stmt)
                count += 1
            except Neo4jError as e:
                if "already exists" in str(e) or "EquivalentSchemaRule" in str(e):
                    count += 1
                else:
                    logger.warning(f"  Statement error: {e.message[:100] if hasattr(e, 'message') else str(e)[:100]}")
        return count

    # ── INIT ──
    def init_constraints(self) -> int:
        logger.info("Creating node uniqueness constraints...")
        stmts = [
            "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT market_id IF NOT EXISTS FOR (n:Market) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT region_id IF NOT EXISTS FOR (n:Region) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT regulation_id IF NOT EXISTS FOR (n:Regulation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT risk_id IF NOT EXISTS FOR (n:RiskFactor) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT strategy_id IF NOT EXISTS FOR (n:Strategy) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT metric_id IF NOT EXISTS FOR (n:FinancialMetric) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT year_id IF NOT EXISTS FOR (n:Year) REQUIRE n.year IS UNIQUE",
            "CREATE CONSTRAINT quarter_id IF NOT EXISTS FOR (n:Quarter) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.doc_id IS UNIQUE",
            "CREATE CONSTRAINT sentence_id IF NOT EXISTS FOR (n:Sentence) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT mechanism_id IF NOT EXISTS FOR (n:Mechanism) REQUIRE n.id IS UNIQUE",
            # v2.0 new node types
            "CREATE CONSTRAINT regulation_change_id IF NOT EXISTS FOR (n:RegulationChange) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT risk_driver_id IF NOT EXISTS FOR (n:RiskDriver) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT risk_event_id IF NOT EXISTS FOR (n:RiskEvent) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT business_segment_id IF NOT EXISTS FOR (n:BusinessSegment) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT mitigation_action_id IF NOT EXISTS FOR (n:MitigationAction) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT evidence_claim_id IF NOT EXISTS FOR (n:EvidenceClaim) REQUIRE n.id IS UNIQUE",
            # Relationship property constraints (Neo4j 5.7+)
            "CREATE CONSTRAINT causes_cs IF NOT EXISTS FOR ()-[r:CAUSES]-() REQUIRE r.causal_strength IS NOT NULL",
            "CREATE CONSTRAINT triggers_cs IF NOT EXISTS FOR ()-[r:TRIGGERS]-() REQUIRE r.causal_strength IS NOT NULL",
            "CREATE CONSTRAINT amplifies_cs IF NOT EXISTS FOR ()-[r:AMPLIFIES]-() REQUIRE r.causal_strength IS NOT NULL",
            "CREATE CONSTRAINT decreases_cs IF NOT EXISTS FOR ()-[r:DECREASES]-() REQUIRE r.causal_strength IS NOT NULL",
            "CREATE CONSTRAINT increases_cs IF NOT EXISTS FOR ()-[r:INCREASES]-() REQUIRE r.causal_strength IS NOT NULL",
            "CREATE CONSTRAINT mitigates_cs IF NOT EXISTS FOR ()-[r:MITIGATES]-() REQUIRE r.causal_strength IS NOT NULL",
            "CREATE CONSTRAINT exposed_cs IF NOT EXISTS FOR ()-[r:EXPOSED_TO]-() REQUIRE r.causal_strength IS NOT NULL",
        ]
        return self._exec_many(stmts)

    def init_indexes(self) -> int:
        logger.info("Creating indexes...")
        stmts = [
            "CREATE INDEX company_name IF NOT EXISTS FOR (n:Company) ON (n.name)",
            "CREATE INDEX product_name IF NOT EXISTS FOR (n:Product) ON (n.name)",
            "CREATE INDEX market_name IF NOT EXISTS FOR (n:Market) ON (n.name)",
            "CREATE INDEX region_name IF NOT EXISTS FOR (n:Region) ON (n.name)",
            "CREATE INDEX risk_name IF NOT EXISTS FOR (n:RiskFactor) ON (n.name)",
            "CREATE INDEX strategy_name IF NOT EXISTS FOR (n:Strategy) ON (n.name)",
            "CREATE INDEX metric_name IF NOT EXISTS FOR (n:FinancialMetric) ON (n.name)",
            "CREATE INDEX mechanism_name IF NOT EXISTS FOR (n:Mechanism) ON (n.name)",
            "CREATE INDEX doc_filename IF NOT EXISTS FOR (n:Document) ON (n.filename)",
            "CREATE INDEX sentence_page IF NOT EXISTS FOR (n:Sentence) ON (n.page)",
            "CREATE INDEX sentence_section IF NOT EXISTS FOR (n:Sentence) ON (n.section)",
            "CREATE INDEX evidence_claim_page IF NOT EXISTS FOR (n:EvidenceClaim) ON (n.page)",
        ]
        return self._exec_many(stmts)

    def init_seed_data(self) -> int:
        logger.info("Seeding core entities (Company, Product, Region, Market, Metric, Year, Regulation, Event)...")
        count = 0

        # Company: NVIDIA
        self._run("MERGE (n:Company {id:'NVIDIA_CORPORATION'}) SET n.name='NVIDIA Corporation', n.ticker='NVDA', n.sector='Semiconductors', n.headquarters='Santa Clara, California'")
        count += 1

        # Competitors
        competitors = [
            ("ADVANCED_MICRO_DEVICES", "AMD"), ("INTEL_CORPORATION", "INTC"),
            ("TSMC", "TSM"), ("SAMSUNG_ELECTRONICS", None), ("BROADCOM_INC", "AVGO"),
            ("QUALCOMM_INC", "QCOM"), ("MICRON_TECHNOLOGY", "MU"), ("SK_HYNIX", None),
        ]
        for cid, ticker in competitors:
            self._run("MERGE (c:Company {id:$id}) SET c.name=$name, c.sector='Semiconductors'",
                      id=cid, name=cid.replace("_", " ").title())
            if ticker:
                self._run("MATCH (c:Company {id:$id}) SET c.ticker=$t", id=cid, t=ticker)
            self._run("MATCH (a:Company {id:'NVIDIA_CORPORATION'}), (b:Company {id:$id}) MERGE (a)-[:COMPETES_WITH]->(b)", id=cid)
            count += 1

        # Regions
        regions = [
            ("UNITED_STATES", "United States", "COUNTRY"), ("CHINA", "China", "COUNTRY"),
            ("TAIWAN", "Taiwan", "COUNTRY"), ("EUROPE", "Europe", "ECONOMIC_BLOCK"),
            ("ASIA_PACIFIC", "Asia Pacific", "CONTINENT"), ("JAPAN", "Japan", "COUNTRY"),
            ("SOUTH_KOREA", "South Korea", "COUNTRY"), ("SINGAPORE", "Singapore", "COUNTRY"),
            ("ISRAEL", "Israel", "COUNTRY"), ("INDIA", "India", "COUNTRY"),
            ("GLOBAL", "Global", "ECONOMIC_BLOCK"),
        ]
        for rid, rname, rtype in regions:
            self._run("MERGE (r:Region {id:$id}) SET r.name=$n, r.region_type=$t", id=rid, n=rname, t=rtype)
            self._run("MATCH (n:Company {id:'NVIDIA_CORPORATION'}), (r:Region {id:$rid}) MERGE (n)-[:OPERATES_IN]->(r)", rid=rid)
            count += 1

        # Products
        products = [
            ("H100_TENSOR_CORE_GPU", "H100 Tensor Core GPU", "Data Center GPU", "Hopper"),
            ("A100_TENSOR_CORE_GPU", "A100 Tensor Core GPU", "Data Center GPU", "Ampere"),
            ("B200_BLACKWELL_GPU", "B200 Blackwell GPU", "Data Center GPU", "Blackwell"),
            ("GEFORCE_RTX_4090", "GeForce RTX 4090", "Consumer GPU", "Ada Lovelace"),
            ("CUDA_PLATFORM", "CUDA Platform", "Software Platform", None),
            ("DRIVE_PLATFORM", "DRIVE Platform", "Automotive Platform", None),
            ("DGX_SYSTEM", "DGX System", "AI Supercomputer", None),
            ("OMNIVERSE_PLATFORM", "Omniverse Platform", "Digital Twin Platform", None),
            ("MELLANOX_NETWORKING", "Mellanox Networking", "Networking", None),
        ]
        for pid, pname, pcat, pgen in products:
            self._run("MERGE (p:Product {id:$id}) SET p.name=$n, p.category=$c", id=pid, n=pname, c=pcat)
            if pgen:
                self._run("MATCH (p:Product {id:$id}) SET p.generation=$g", id=pid, g=pgen)
            self._run("MATCH (n:Company {id:'NVIDIA_CORPORATION'}), (p:Product {id:$pid}) MERGE (n)-[:PRODUCES]->(p)", pid=pid)
            count += 1

        # Markets
        markets = [
            ("GPU_MARKET", "GPU Market", "PRODUCT_CATEGORY"),
            ("DATA_CENTER_MARKET", "Data Center Market", "SECTOR"),
            ("AI_CHIP_MARKET", "AI Chip Market", "SECTOR"),
            ("AUTOMOTIVE_MARKET", "Automotive Market", "SECTOR"),
            ("GAMING_MARKET", "Gaming Market", "SECTOR"),
            ("CHINA_MARKET", "China Market", "GEOGRAPHIC"),
            ("CLOUD_COMPUTING_MARKET", "Cloud Computing Market", "SECTOR"),
        ]
        for mid, mname, mtype in markets:
            self._run("MERGE (m:Market {id:$id}) SET m.name=$n, m.market_type=$t", id=mid, n=mname, t=mtype)
            count += 1

        # Financial Metrics (canonical)
        for mid, (mtype, direction) in FINANCIAL_METRICS_REGISTRY.items():
            self._run("MERGE (fm:FinancialMetric {id:$id}) SET fm.name=$n, fm.metric_type=$t, fm.direction=$d",
                      id=mid, n=mid.replace("_", " ").title(), t=mtype, d=direction)
            count += 1

        # Years
        for y in range(2019, 2027):
            fiscal_end = {2019: "2019-01-27", 2020: "2020-01-26", 2021: "2021-01-31",
                          2022: "2022-01-30", 2023: "2023-01-29", 2024: "2024-01-28",
                          2025: "2025-01-26", 2026: "2026-01-25"}
            self._run("MERGE (y:Year {year:$y}) SET y.fiscal_year_end=$f", y=y, f=fiscal_end.get(y, f"{y}-01-31"))
            count += 1

        # Regulations
        regulations = [
            ("US_CHIP_EXPORT_CONTROLS_2022", "US Chip Export Controls (Oct 2022)", "EXPORT_CONTROL", "United States", "2022-10-07"),
            ("US_CHIP_EXPORT_CONTROLS_2023", "US Chip Export Controls (Oct 2023)", "EXPORT_CONTROL", "United States", "2023-10-17"),
            ("EU_AI_ACT", "EU AI Act", "AI_SAFETY", "European Union", "2024-08-01"),
            ("US_CHIPS_ACT", "US CHIPS and Science Act", "FINANCIAL_REPORTING", "United States", "2022-08-09"),
            ("BIS_ENTITY_LIST", "BIS Entity List Restrictions", "EXPORT_CONTROL", "United States", None),
        ]
        for rid, rname, rtype, juris, eff in regulations:
            self._run("MERGE (r:Regulation {id:$id}) SET r.name=$n, r.regulation_type=$t, r.jurisdiction=$j",
                      id=rid, n=rname, t=rtype, j=juris)
            if eff:
                self._run("MATCH (r:Regulation {id:$id}) SET r.effective_date=$d", id=rid, d=eff)
            count += 1

        # Events
        events = [
            ("COVID_19", "COVID-19 Pandemic", "PANDEMIC", "2020-03-11"),
            ("CHATGPT_LAUNCH", "ChatGPT Launch", "TECHNOLOGY_BREAKTHROUGH", "2022-11-30"),
            ("NVIDIA_ARM_TERMINATED", "NVIDIA-ARM Acquisition Terminated", "MERGER_ACQUISITION", "2022-02-08"),
            ("BLACKWELL_ANNOUNCED", "Blackwell Architecture Announced", "PRODUCT_LAUNCH", "2024-03-18"),
        ]
        for eid, ename, etype, edate in events:
            self._run("MERGE (e:Event {id:$id}) SET e.name=$n, e.event_type=$t, e.event_date=$d",
                      id=eid, n=ename, t=etype, d=edate)
            year = int(edate[:4])
            self._run("MATCH (e:Event {id:$id}), (y:Year {year:$y}) MERGE (e)-[:OCCURS_DURING]->(y)", id=eid, y=year)
            count += 1

        logger.info(f"Seeded {count} core entities")
        return count

    def init_all(self) -> Dict[str, int]:
        logger.info("=" * 60)
        logger.info("INITIALIZING Strategic-GraphRAG Neo4j Schema v1.0")
        logger.info("=" * 60)
        results = {
            "constraints": self.init_constraints(),
            "indexes": self.init_indexes(),
            "seed_entities": self.init_seed_data(),
        }
        logger.info("=" * 60)
        logger.info(f"COMPLETE — Constraints:{results['constraints']} Indexes:{results['indexes']} Seeds:{results['seed_entities']}")
        logger.info("=" * 60)
        return results

    # ── RESET ──
    def reset_all(self) -> int:
        logger.warning("DROPPING ALL DATA AND SCHEMA")
        count = 0
        for c in self._run("SHOW CONSTRAINTS"):
            try:
                self._run(f"DROP CONSTRAINT {c['name']}")
                count += 1
            except Neo4jError:
                pass
        for idx in self._run("SHOW INDEXES"):
            try:
                self._run(f"DROP INDEX {idx['name']}")
                count += 1
            except Neo4jError:
                pass
        self._run("MATCH (n) DETACH DELETE n")
        logger.info(f"Reset complete. Dropped {count} constraints/indexes.")
        return count

    # ── VERIFY ──
    def verify(self) -> Dict:
        results = {"node_counts": {}, "rel_counts": {}}
        for label in NODE_LABELS:
            r = self._run(f"MATCH (n:{label}) RETURN count(n) AS c")
            results["node_counts"][label] = r[0]["c"] if r else 0
        for rel_type in RELATIONSHIP_TYPES:
            r = self._run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c")
            results["rel_counts"][rel_type] = r[0]["c"] if r else 0
        return results

    def stats(self) -> Dict:
        r = self._run("MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS rels")
        stats = {"total_nodes": r[0]["nodes"], "total_rels": r[0]["rels"]} if r else {}
        stats["by_label"] = {}
        for label in NODE_LABELS:
            r = self._run(f"MATCH (n:{label}) RETURN count(n) AS c")
            if r and r[0]["c"] > 0:
                stats["by_label"][label] = r[0]["c"]
        stats["by_relationship"] = {}
        for rel_type in RELATIONSHIP_TYPES:
            r = self._run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c")
            if r and r[0]["c"] > 0:
                stats["by_relationship"][rel_type] = r[0]["c"]
        return stats


# ── CLI ──
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Strategic-GraphRAG Schema Manager")
    p.add_argument("--init", action="store_true", help="Initialize full schema")
    p.add_argument("--reset", action="store_true", help="Drop all data & schema")
    p.add_argument("--verify", action="store_true", help="Verify schema")
    p.add_argument("--stats", action="store_true", help="Graph statistics")
    p.add_argument("--uri", type=str)
    p.add_argument("--user", type=str)
    p.add_argument("--password", type=str)
    args = p.parse_args()

    mgr = SchemaManager(uri=args.uri, user=args.user, password=args.password)
    if not mgr.connect():
        sys.exit(1)
    try:
        if args.reset:
            c = input("Type 'YES' to confirm destructive reset: ")
            if c == "YES":
                mgr.reset_all()
                mgr.init_all()
        elif args.init:
            mgr.init_all()
        elif args.verify:
            print(json.dumps(mgr.verify(), indent=2, default=str))
        elif args.stats:
            print(json.dumps(mgr.stats(), indent=2))
        else:
            p.print_help()
    finally:
        mgr.close()
