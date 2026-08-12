# Strategic-GraphRAG: Evidence-Grounded Temporal Causal KG

> **Implementation contract: single-PDF stabilization (2026-08-11)**
>
> The current runnable baseline materializes `Company`, `Product`, `Market`,
> `Region`, `RiskFactor`, `FinancialMetric`, `Event`, `Document`, `Sentence`,
> `EvidenceClaim`, and `Year` nodes for one SEC filing. Its verified causal
> relation families are `CAUSES`, `DECREASES`, `INCREASES`, `EXPOSED_TO`,
> `OPERATES_IN`, and `PRODUCES`.
>
> `Mechanism`, `BusinessSegment`, `RiskDriver`, `RegulationChange`, and
> `MitigationAction` are extension targets. They must not be described as
> populated layers until extraction, ingestion, validation, and evaluation
> contain them.
>
> Current edge/evidence properties are `year`, `page`, `source_filing`,
> `evidence_sentence`, `confidence`, and `extraction_method`. Claims are
> connected through `SUPPORTED_BY`, `ABOUT_SOURCE`, and `ABOUT_TARGET`; the
> current graph does not yet create `Document-[:DISCLOSES]->EvidenceClaim`.

## Architecture: From Entity-Relation to Causal-Provenance Graph

### Research target versus current baseline
v1.0 allowed `EXPORT_CONTROL → DECREASES → REVENUE` — a single-hop edge that collapses
the entire causal mechanism into one relationship. This is scientifically invalid:
regulations do not "decrease" revenue; they *constrain market access*, which *exposes*
business segments to *revenue concentration risk*, which *may decrease* reported revenue.

The paragraph above describes the research target. In the current single-PDF
baseline, a direct risk-to-metric edge is retained only when its filing
sentence passes the verbatim evidence and ontology validators. It should be
reported as a direct disclosed impact, not as a completed mechanism model.

### Target v2.0 Multi-Hop Causal Ontology (extension roadmap)

```
Layer 1: ENTITY           Company, Product, Market, Region
Layer 2: EXTERNAL_DRIVER  RegulationChange, MacroEvent, GeopoliticalEvent, TechnologyShift
Layer 3: RISK_DRIVER      ExportRestriction, SupplyConstraint, DemandShift, CompetitiveThreat
Layer 4: RISK_EXPOSURE    RiskFactor (specific, granular), RiskEvent (temporal instance)
Layer 5: TRANSMISSION     Mechanism, BusinessSegment (how risk flows to financials)
Layer 6: IMPACT           FinancialMetric (Revenue, Margin, Cost, CashFlow), MarketPosition
Layer 7: MITIGATION       Strategy, MitigationAction (concrete action + evidence of effectiveness)
Layer 8: EVIDENCE         Document → EvidenceClaim → Sentence (provenance chain)
```

### Key New Node Types

| Label | Description | Example |
|---|---|---|
| RegulationChange | Specific regulatory event with date | US_CHIP_EXPORT_CONTROLS_2022, US_CHIP_EXPORT_CONTROLS_2023 |
| RiskDriver | Causal antecedent to risk exposure | EXPORT_RESTRICTION, HBM_SUPPLY_CONSTRAINT |
| BusinessSegment | Revenue/cost center affected | DATA_CENTER_SEGMENT, GAMING_SEGMENT, AUTOMOTIVE_SEGMENT |
| MitigationAction | Concrete action with effectiveness evidence | SUPPLIER_MULTI_SOURCING, PRODUCT_REDESIGN, LOBBYING_EFFORT |
| EvidenceClaim | Atomic claim extracted from document with verification status | (individual evidence sentences) |

### Key New Relation Types

| Relation | Domain → Range | Semantics |
|---|---|---|
| CAUSES | ExternalDriver → RiskDriver | Regulation causes restriction |
| EXPOSED_THROUGH | Company → BusinessSegment → RiskDriver | Company is exposed via segment |
| CONSTRAINS_MARKET | RiskDriver → Market | Restriction constrains market access |
| AFFECTS_SEGMENT | RiskDriver → BusinessSegment | Constraint affects specific segment |
| IMPACTS | RiskDriver → FinancialMetric | Driver impacts financial metric |
| DISCLOSES | Document → EvidenceClaim | Document contains claim |
| MENTIONS | EvidenceClaim → Entity | Claim mentions entity |
| POSSIBLE_RELATION | Entity → Entity | Weak signal, requires verification |
| EXECUTES | Company → MitigationAction | Company takes action |
| ADDRESSES | MitigationAction → RiskDriver | Action targets risk driver |

### Causal Strength Tiers (Judea Pearl-inspired)

| Tier | Label | Condition |
|---|---|---|
| 1 | CONFIRMED_CAUSAL | Evidence contains explicit causal verb + both entities in same sentence |
| 2 | STRONG_ASSOCIATION | Evidence strongly implies causation (e.g., "as a result of") |
| 3 | WEAK_ASSOCIATION | Entities co-occur in same risk disclosure section |
| 4 | DISCLOSED_ONLY | Entity mentioned but no causal language |
| 5 | INFERRED | System-inferred relationship (lowest confidence) |

### Temporal Binding (TKG-style)
Every entity and relationship MUST carry:
- `fiscal_year`: INTEGER (or year range)
- `effective_date`: STRING (approximate date of effect)
- `source_filing`: STRING (document ID)
- `page_number`: INTEGER
- `evidence_sentence`: STRING (verbatim quote)
- `confidence_score`: FLOAT (0.0–1.0)
- `source_type`: STRING (LLM_EXTRACTION | RULE_EXTRACTION | MANUAL_CURATION)

### Claim-Evidence Model
- Every causal edge MUST be backed by at least one EvidenceClaim node
- EvidenceClaim contains: claim_text, document_id, page, paragraph, verification_status
- LLM synthesis MUST cite EvidenceClaim IDs, not just page numbers
- Graph-constrained generation: LLM receives (Path + EvidenceClaims) → generates narrative
  with inline citations to specific claims
