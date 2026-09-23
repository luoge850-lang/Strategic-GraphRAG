# Strategic-GraphRAG: Evidence-Grounded Temporal Evidence Graph

> **Implementation contract: three-filing development snapshot (2026-09-22)**
>
> The current runnable baseline materializes `Company`, `Product`, `Market`,
> `Region`, `RiskFactor`, `FinancialMetric`, `Event`, `Document`, `Sentence`,
> The current development snapshot covers three NVIDIA 10-K filings and
> stores `EvidenceClaim` records with filing, page, quote, entity, relation,
> and claim-ID provenance. Relation families include disclosed business,
> structural, metric, and causal-language relations. A stored relation is not
> automatically a causal identification result.
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

## Architecture: From Entity-Relation to Evidence-Provenance Graph

### Research target versus current baseline
v1.0 allowed `EXPORT_CONTROL → DECREASES → REVENUE` — a single-hop edge that collapses
the entire causal mechanism into one relationship. This is too coarse for a
multi-hop evidence model: regulations do not automatically imply a direct
revenue change. A more cautious representation is that they *constrain market
access*, which may expose a business segment to a documented risk, which may
be associated with a reported revenue change.

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

### Disclosure-language strength labels

These labels describe the wording and extraction policy of a filing evidence
item. They are not Pearl-style causal effects, counterfactual claims, or
investment predictions. The legacy enum names remain for compatibility with
existing records; new documentation must state their operational meaning.

| Tier | Label | Condition |
|---|---|---|
| 1 | `CONFIRMED_CAUSAL` (legacy) | Evidence contains explicit causal wording and both entities in the bounded text span; call this “explicit causal language,” not identified causality. |
| 2 | `STRONG_ASSOCIATION` | Evidence uses consequential or implied causal wording without a causal identification design. |
| 3 | `WEAK_ASSOCIATION` | Entities co-occur in a disclosure context without a sufficient direct predicate. |
| 4 | `DISCLOSED_ONLY` | The filing discloses the entity or metric without causal wording. |
| 5 | `INFERRED` | System-derived relation; it is the weakest evidence tier and requires separate validation. |

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
- Every active business edge MUST be backed by at least one EvidenceClaim node
- EvidenceClaim contains: claim_text, document_id, page, paragraph, verification_status
- LLM synthesis MUST cite EvidenceClaim IDs, not just page numbers
- Graph-constrained generation: the LLM receives a bounded Path plus
  EvidenceClaims and must cite the specific claim IDs. Generation cannot turn
  a path into a counterfactual causal claim.
