// =============================================================================
// Strategic-GraphRAG: Temporal Causal Knowledge Graph Schema v1.0
// Neo4j Schema Initialization — Complete 6-Layer Financial Ontology
// =============================================================================
// Architecture:
//   Layer 1: ENTITY      — Company, Product, Market, Region, Regulation
//   Layer 2: RISK        — Fine-grained Financial Risk Factors
//   Layer 3: STRATEGY    — Business Risk Mitigation Strategies
//   Layer 4: METRIC      — Financial Metrics (Revenue, Margin, etc.)
//   Layer 5: TEMPORAL    — Year, Quarter, Event Timeline
//   Layer 6: EVIDENCE    — Document, Page, Sentence (Provenance)
//
// Relationship Semantics (Strict Financial):
//   EXPOSED_TO, CAUSES, MITIGATES, AMPLIFIES, INCREASES, DECREASES,
//   IMPLEMENTS, TRIGGERS, PRECEDES, OCCURS_DURING, REPORTED_IN,
//   HAS_EVIDENCE, OPERATES_IN, PRODUCES, REGULATED_BY, CONSTRAINS
// =============================================================================


// =============================================================================
// PART 1: CONSTRAINTS — Uniqueness & Entity Integrity
// =============================================================================

// ---- Layer 1: ENTITY Nodes ----
CREATE CONSTRAINT company_id_unique      IF NOT EXISTS FOR (n:Company)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT product_id_unique      IF NOT EXISTS FOR (n:Product)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT market_id_unique       IF NOT EXISTS FOR (n:Market)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT region_id_unique       IF NOT EXISTS FOR (n:Region)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT regulation_id_unique   IF NOT EXISTS FOR (n:Regulation)  REQUIRE n.id IS UNIQUE;

// ---- Layer 2: RISK Nodes ----
CREATE CONSTRAINT risk_factor_id_unique  IF NOT EXISTS FOR (n:RiskFactor)  REQUIRE n.id IS UNIQUE;

// ---- Layer 3: STRATEGY Nodes ----
CREATE CONSTRAINT strategy_id_unique     IF NOT EXISTS FOR (n:Strategy)    REQUIRE n.id IS UNIQUE;

// ---- Layer 4: METRIC Nodes ----
CREATE CONSTRAINT metric_id_unique       IF NOT EXISTS FOR (n:FinancialMetric) REQUIRE n.id IS UNIQUE;

// ---- Layer 5: TEMPORAL Nodes ----
CREATE CONSTRAINT year_id_unique         IF NOT EXISTS FOR (n:Year)        REQUIRE n.year IS UNIQUE;
CREATE CONSTRAINT quarter_id_unique      IF NOT EXISTS FOR (n:Quarter)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT event_id_unique        IF NOT EXISTS FOR (n:Event)       REQUIRE n.id IS UNIQUE;

// ---- Layer 6: EVIDENCE Nodes ----
CREATE CONSTRAINT document_id_unique     IF NOT EXISTS FOR (n:Document)    REQUIRE n.doc_id IS UNIQUE;
CREATE CONSTRAINT sentence_id_unique     IF NOT EXISTS FOR (n:Sentence)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT evidence_claim_id_unique IF NOT EXISTS FOR (n:EvidenceClaim) REQUIRE n.id IS UNIQUE;

// ---- Causal Mechanism Nodes (Bridge Layer) ----
CREATE CONSTRAINT mechanism_id_unique    IF NOT EXISTS FOR (n:Mechanism)   REQUIRE n.id IS UNIQUE;


// =============================================================================
// PART 2: INDEXES — Query Performance
// =============================================================================

// Name lookups
CREATE INDEX company_name_idx      IF NOT EXISTS FOR (n:Company)     ON (n.name);
CREATE INDEX product_name_idx      IF NOT EXISTS FOR (n:Product)     ON (n.name);
CREATE INDEX market_name_idx       IF NOT EXISTS FOR (n:Market)      ON (n.name);
CREATE INDEX region_name_idx       IF NOT EXISTS FOR (n:Region)      ON (n.name);
CREATE INDEX regulation_name_idx   IF NOT EXISTS FOR (n:Regulation)  ON (n.name);
CREATE INDEX risk_name_idx         IF NOT EXISTS FOR (n:RiskFactor)  ON (n.name);
CREATE INDEX strategy_name_idx     IF NOT EXISTS FOR (n:Strategy)    ON (n.name);
CREATE INDEX metric_name_idx       IF NOT EXISTS FOR (n:FinancialMetric) ON (n.name);
CREATE INDEX mechanism_name_idx    IF NOT EXISTS FOR (n:Mechanism)   ON (n.name);

// Temporal indexes
CREATE INDEX year_value_idx        IF NOT EXISTS FOR (n:Year)        ON (n.year);

// Evidence indexes
CREATE INDEX document_name_idx     IF NOT EXISTS FOR (n:Document)    ON (n.filename);
CREATE INDEX sentence_page_idx     IF NOT EXISTS FOR (n:Sentence)    ON (n.page);
CREATE INDEX evidence_claim_page_idx IF NOT EXISTS FOR (n:EvidenceClaim) ON (n.page);

// Full-text indexes for semantic search
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:Company|Product|Market|Region|Regulation|RiskFactor|Strategy|FinancialMetric|Event|Mechanism)
ON EACH [n.name, n.description];

CREATE FULLTEXT INDEX evidence_fulltext IF NOT EXISTS
FOR (n:Sentence)
ON EACH [n.text];


// =============================================================================
// PART 3: RELATIONSHIP TYPE REGISTRATION
// =============================================================================
// (Documentation only — Neo4j creates relationship types on first use.
//  This serves as the canonical registry of allowed financial semantics.)

// ┌─────────────────────┬──────────────────────────────────────────────────┐
// │ RELATIONSHIP        │ SEMANTIC MEANING                                  │
// ├─────────────────────┼──────────────────────────────────────────────────┤
// │ EXPOSED_TO           │ Entity is exposed to a specific risk             │
// │ CAUSES               │ Direct causal link (A causes B)                  │
// │ MITIGATES            │ Strategy reduces/controls a risk                 │
// │ AMPLIFIES            │ Risk A makes Risk B worse                        │
// │ INCREASES            │ A increases a financial metric                   │
// │ DECREASES            │ A decreases a financial metric                   │
// │ IMPLEMENTS           │ Company implements a strategy                    │
// │ TRIGGERS             │ Event/Risk triggers a mechanism                  │
// │ PRECEDES             │ Temporal ordering (A before B)                   │
// │ OCCURS_DURING        │ Event occurs during a time period                │
// │ REPORTED_IN          │ Finding is reported in a document                │
// │ HAS_EVIDENCE          │ Relationship has supporting sentence evidence   │
// │ OPERATES_IN          │ Company operates in a region/market              │
// │ PRODUCES             │ Company produces a product                       │
// │ REGULATED_BY         │ Market/Product is regulated by a regulation      │
// │ CONSTRAINS           │ Regulation constrains a market/product           │
// │ AGGRAVATES           │ Mechanism makes a risk worse                     │
// │ COMPETES_WITH        │ Company competes with another entity             │
// │ DEPENDS_ON           │ Entity depends on another entity                 │
// │ SUPPLIES_TO           │ Company supplies to a market/region             │
// │ REPORTS               │ Document reports a fiscal year                 │
// │ SUPPORTED_BY          │ Evidence claim is supported by a sentence      │
// │ ABOUT_SOURCE          │ Evidence claim identifies source entity        │
// │ ABOUT_TARGET          │ Evidence claim identifies target entity        │
// └─────────────────────┴──────────────────────────────────────────────────┘


// =============================================================================
// PART 4: RELATIONSHIP PROPERTY SCHEMA
// =============================================================================
// All causal relationships carry these standard properties:
//
//   causal_strength:  "DIRECT_CAUSALITY" | "INDIRECT_CAUSALITY" |
//                     "RISK_ASSOCIATION" | "SPECULATIVE_RELATION" |
//                     "DISCLOSED_EXPOSURE"
//
//   confidence:       0.0 – 1.0  (LLM extraction confidence)
//   support_count:    INTEGER     (number of evidence sentences supporting)
//   year:             INTEGER     (fiscal year of the relationship)
//   extraction_method:"LLM" | "RULE" | "HYBRID"
//   created_at:       DATETIME


// =============================================================================
// PART 5: NODE PROPERTY SCHEMAS
// =============================================================================

// --- Company ---
//   id:              STRING (unique, e.g. "NVIDIA_CORPORATION")
//   name:            STRING (display name, e.g. "NVIDIA Corporation")
//   ticker:          STRING (e.g. "NVDA")
//   sector:          STRING (e.g. "Semiconductors")
//   headquarters:    STRING
//   description:     STRING

// --- Product ---
//   id, name, category, generation, description

// --- Market ---
//   id, name, market_type (GEOGRAPHIC|SECTOR|PRODUCT_CATEGORY),
//   size_estimate, growth_rate, description

// --- Region ---
//   id, name, region_type (COUNTRY|CONTINENT|ECONOMIC_BLOCK),
//   gdp_rank, political_stability, description

// --- Regulation ---
//   id, name, jurisdiction, regulation_type (EXPORT_CONTROL|TARIFF|
//     ANTITRUST|ENVIRONMENTAL|DATA_PRIVACY|FINANCIAL_REPORTING),
//   effective_date, expiry_date, description

// --- RiskFactor ---
//   id, name, risk_category (SUPPLY_CHAIN|GEOPOLITICAL|REGULATORY|
//     MARKET|OPERATIONAL|FINANCIAL|TECHNOLOGY|REPUTATION|CYBER|
//     MACROECONOMIC|LEGAL|ENVIRONMENTAL),
//   severity (CRITICAL|HIGH|MEDIUM|LOW), likelihood, description

// --- Strategy ---
//   id, name, strategy_type (DIVERSIFICATION|COST_OPTIMIZATION|
//     R_AND_D_INVESTMENT|M_AND_A|MARKET_EXPANSION|TALENT_MANAGEMENT|
//     TECHNOLOGY_INVESTMENT|SUPPLY_CHAIN_RESILIENCE|REGULATORY_COMPLIANCE),
//   implementation_stage, effectiveness, description

// --- FinancialMetric ---
//   id, name, metric_type (REVENUE|COST|PROFIT|CASH_FLOW|VALUATION|
//     EFFICIENCY|GROWTH|LEVERAGE),
//   unit, value, fiscal_year, trend_direction, description

// --- Mechanism ---
//   id, name, mechanism_type (TRANSMISSION|AMPLIFICATION|ABSORPTION|
//     FEEDBACK|THRESHOLD),
//   description

// --- Year ---
//   year:           INTEGER (e.g. 2024)
//   fiscal_year_end: STRING (e.g. "2024-01-28")

// --- Quarter ---
//   id:             STRING (e.g. "FY2025_Q3")
//   quarter:        INTEGER (1-4)
//   fiscal_year:    INTEGER

// --- Event ---
//   id, name, event_type (EARNINGS_CALL|PRODUCT_LAUNCH|REGULATORY_ACTION|
//     NATURAL_DISASTER|MERGER_ACQUISITION|LEADERSHIP_CHANGE|
//     MARKET_CRASH|TRADE_POLICY_CHANGE),
//   event_date, description

// --- Document ---
//   doc_id:          STRING (e.g. "NVIDIA_10K_FY2025")
//   filename:        STRING
//   doc_type:        STRING (10-K|10-Q|8-K|ANNUAL_REPORT|PRESS_RELEASE)
//   filing_date:     STRING
//   fiscal_year:     INTEGER
//   total_pages:     INTEGER

// --- Sentence ---
//   id:              STRING
//   text:            STRING (the exact sentence text)
//   page:            INTEGER
//   paragraph:       INTEGER
//   section:         STRING (Item 1A|Item 7|Item 8)
//   doc_id:          STRING (reference to Document)

// --- EvidenceClaim ---
//   id:                STRING (stable claim/evidence identifier)
//   text:              STRING (verbatim evidence excerpt)
//   relation_id:       STRING (stable native relationship identifier)
//   relation_type:     STRING
//   page:              INTEGER
//   fiscal_year:       INTEGER
//   verification_status: STRING (VERBATIM|REVIEW_REQUIRED|REJECTED)


// =============================================================================
// PART 6: SEED DATA — Core Entity Nodes
// =============================================================================

// --- NVIDIA Corporation (Primary Company) ---
MERGE (nvidia:Company {id: "NVIDIA_CORPORATION"})
SET nvidia.name = "NVIDIA Corporation",
    nvidia.ticker = "NVDA",
    nvidia.sector = "Semiconductors",
    nvidia.headquarters = "Santa Clara, California, USA",
    nvidia.description = "NVIDIA is a multinational technology company that designs graphics processing units (GPUs), application programming interfaces (APIs) for data science and high-performance computing, and system-on-a-chip units (SoCs) for mobile computing and automotive markets.";

// --- Key Competitors ---
MERGE (amd:Company {id: "ADVANCED_MICRO_DEVICES"})
SET amd.name = "Advanced Micro Devices, Inc.", amd.ticker = "AMD",
    amd.sector = "Semiconductors";

MERGE (intel:Company {id: "INTEL_CORPORATION"})
SET intel.name = "Intel Corporation", intel.ticker = "INTC",
    intel.sector = "Semiconductors";

MERGE (tsmc:Company {id: "TSMC"})
SET tsmc.name = "Taiwan Semiconductor Manufacturing Company Limited",
    tsmc.ticker = "TSM", tsmc.sector = "Semiconductor Foundry";

// --- Key Regions ---
MERGE (us:Region {id: "UNITED_STATES"})
SET us.name = "United States", us.region_type = "COUNTRY";

MERGE (china:Region {id: "CHINA"})
SET china.name = "China", china.region_type = "COUNTRY";

MERGE (taiwan:Region {id: "TAIWAN"})
SET taiwan.name = "Taiwan", taiwan.region_type = "COUNTRY";

MERGE (europe:Region {id: "EUROPE"})
SET europe.name = "Europe", europe.region_type = "ECONOMIC_BLOCK";

MERGE (asia_pacific:Region {id: "ASIA_PACIFIC"})
SET asia_pacific.name = "Asia Pacific", asia_pacific.region_type = "CONTINENT";

// --- Key Markets ---
MERGE (gpu_mkt:Market {id: "GPU_MARKET"})
SET gpu_mkt.name = "GPU Market", gpu_mkt.market_type = "PRODUCT_CATEGORY";

MERGE (dc_mkt:Market {id: "DATA_CENTER_MARKET"})
SET dc_mkt.name = "Data Center Market", dc_mkt.market_type = "SECTOR";

MERGE (ai_mkt:Market {id: "AI_CHIP_MARKET"})
SET ai_mkt.name = "AI Chip Market", ai_mkt.market_type = "SECTOR";

MERGE (auto_mkt:Market {id: "AUTOMOTIVE_MARKET"})
SET auto_mkt.name = "Automotive Market", auto_mkt.market_type = "SECTOR";

MERGE (gaming_mkt:Market {id: "GAMING_MARKET"})
SET gaming_mkt.name = "Gaming Market", gaming_mkt.market_type = "SECTOR";

MERGE (china_mkt:Market {id: "CHINA_MARKET"})
SET china_mkt.name = "China Market", china_mkt.market_type = "GEOGRAPHIC";

// --- SEED RELATIONSHIPS: Company —[OPERATES_IN]—> Region ---
MERGE (nvidia)-[:OPERATES_IN {confidence: 1.0, extraction_method: "SEED"}]->(us);
MERGE (nvidia)-[:OPERATES_IN {confidence: 1.0, extraction_method: "SEED"}]->(china);
MERGE (nvidia)-[:OPERATES_IN {confidence: 1.0, extraction_method: "SEED"}]->(taiwan);
MERGE (nvidia)-[:OPERATES_IN {confidence: 1.0, extraction_method: "SEED"}]->(europe);
MERGE (nvidia)-[:OPERATES_IN {confidence: 1.0, extraction_method: "SEED"}]->(asia_pacific);

// --- SEED RELATIONSHIPS: Company —[PRODUCES]—> Product ---
MERGE (gpu:Product {id: "GPU"})
SET gpu.name = "Graphics Processing Unit", gpu.category = "Processor",
    gpu.description = "General-purpose GPU for gaming, professional visualization, and data center";

MERGE (h100:Product {id: "H100_TENSOR_CORE_GPU"})
SET h100.name = "H100 Tensor Core GPU", h100.category = "Data Center GPU",
    h100.generation = "Hopper";

MERGE (a100:Product {id: "A100_TENSOR_CORE_GPU"})
SET a100.name = "A100 Tensor Core GPU", a100.category = "Data Center GPU",
    a100.generation = "Ampere";

MERGE (b200:Product {id: "B200_BLACKWELL_GPU"})
SET b200.name = "B200 Blackwell GPU", b200.category = "Data Center GPU",
    b200.generation = "Blackwell";

MERGE (cuda:Product {id: "CUDA_PLATFORM"})
SET cuda.name = "CUDA Platform", cuda.category = "Software Platform";

MERGE (drive:Product {id: "DRIVE_PLATFORM"})
SET drive.name = "DRIVE Platform", drive.category = "Automotive Platform";

MERGE (dgx:Product {id: "DGX_SYSTEM"})
SET dgx.name = "DGX System", dgx.category = "AI Supercomputer";

MERGE (nvidia)-[:PRODUCES {confidence: 1.0, extraction_method: "SEED"}]->(gpu);
MERGE (nvidia)-[:PRODUCES {confidence: 1.0, extraction_method: "SEED"}]->(h100);
MERGE (nvidia)-[:PRODUCES {confidence: 1.0, extraction_method: "SEED"}]->(a100);
MERGE (nvidia)-[:PRODUCES {confidence: 1.0, extraction_method: "SEED"}]->(b200);
MERGE (nvidia)-[:PRODUCES {confidence: 1.0, extraction_method: "SEED"}]->(cuda);
MERGE (nvidia)-[:PRODUCES {confidence: 1.0, extraction_method: "SEED"}]->(drive);
MERGE (nvidia)-[:PRODUCES {confidence: 1.0, extraction_method: "SEED"}]->(dgx);

// --- SEED RELATIONSHIPS: Year nodes ---
FOREACH (y IN RANGE(2019, 2026) |
  MERGE (year:Year {year: y})
  SET year.fiscal_year_end = CASE y
    WHEN 2019 THEN "2019-01-27"
    WHEN 2020 THEN "2020-01-26"
    WHEN 2021 THEN "2021-01-31"
    WHEN 2022 THEN "2022-01-30"
    WHEN 2023 THEN "2023-01-29"
    WHEN 2024 THEN "2024-01-28"
    WHEN 2025 THEN "2025-01-26"
    WHEN 2026 THEN "2026-01-25"
    ELSE toString(y) + "-01-31"
  END
);

RETURN "Schema v1.0 initialized successfully: 6-layer temporal causal financial knowledge graph ready." AS status;
