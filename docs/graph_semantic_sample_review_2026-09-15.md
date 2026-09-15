# Graph Semantic Sample Review — 2026-09-15

## Scope and boundary

The read-only semantic audit found 69 normalized-triple groups supported by
different evidence records. This review uses the reproducible 20-group sample
stored in `reports/graph_semantic_consistency_2026-09-15.json`. The sample is a
machine-assisted semantic screening by the main review thread, not an
independent human annotation set and not a claim-level accuracy estimate.

No Neo4j write, PDF change, claim-ID change, or deletion of duplicate evidence
was performed. Repeated evidence can be useful: it may represent the same
disclosure repeated across sections or years. The question is whether each
individual evidence passage supports the exact source, relation type, and
target.

## Decision rule

- **EVIDENCE_ALIGNED**: the passage directly states or unambiguously expresses
  the endpoint and the relation direction.
- **WEAK_OR_OVERGENERALIZED**: the passage mentions the entity or a related
  product/market, but the exact relation verb is inferred or too broad.
- **LIKELY_MISLABELED**: the endpoint is absent, the direction conflicts with
  the passage, or the relation is materially stronger/different than the
  wording supports.

These labels are triage labels. They must not be converted into formal
precision/recall or used to rewrite the graph without a separately approved
repair plan.

## Sample findings

| # | Normalized triple | Representative pages | Screening result | Reason |
|---:|---|---|---|---|
| 1 | `NVIDIA_CORPORATION --OPERATES_IN--> GAMING_MARKET` | 2023 p.4, 2023 p.36, 2025 p.36 | WEAK_OR_OVERGENERALIZED | The passages discuss demand and platforms related to gaming, but do not consistently state that NVIDIA operates in the market. |
| 2 | `NVIDIA_CORPORATION --PRODUCES--> GPU` | 2023 p.4, p.5, p.6 | WEAK_OR_OVERGENERALIZED | The passages mention NVIDIA GPUs and their use; “produces” is stronger than the wording in several passages. |
| 3 | `NVIDIA_CORPORATION --OPERATES_IN--> DATA_CENTER_MARKET` | 2023 p.5, p.6, 2024 p.5 | WEAK_OR_OVERGENERALIZED | Data-center platforms and services are described, but market participation is partly inferred from product descriptions. |
| 4 | `NVIDIA_CORPORATION --PRODUCES--> GEFORCE_RTX_4090` | 2023 p.5, p.8, 2024 p.36 | LIKELY_MISLABELED | The representative passages mention generic GeForce/RTX products or RTX 4060/4070, not the RTX 4090 endpoint. |
| 5 | `NVIDIA_CORPORATION --PRODUCES--> JETSON_PLATFORM` | 2023 p.5, 2024 p.5 | WEAK_OR_OVERGENERALIZED | Jetson is listed as a platform, but the excerpts do not consistently contain an explicit production statement. |
| 6 | `NVIDIA_CORPORATION --PRODUCES--> OMNIVERSE_PLATFORM` | 2023 p.5, p.7, 2024 p.4 | WEAK_OR_OVERGENERALIZED | The excerpts describe Omniverse as software/platform offerings; “produces” is an abstraction rather than a repeated explicit verb. |
| 7 | `NVIDIA_CORPORATION --PRODUCES--> TENSOR_CORE_GPU` | 2023 p.7, p.38 | WEAK_OR_OVERGENERALIZED | The passages mention Tensor Cores and a Tensor Core GPU, but the production relation is not explicit in both passages. |
| 8 | `NVIDIA_CORPORATION --PRODUCES--> DRIVE_PLATFORM` | 2023 p.7, p.8, 2025 p.7 | WEAK_OR_OVERGENERALIZED | NVIDIA offers and uses Drive platforms/software; the exact “produces” predicate is not uniformly stated. |
| 9 | `NVIDIA_CORPORATION --OPERATES_IN--> PROFESSIONAL_VIZ_MARKET` | 2023 p.7, 2024 p.6, 2025 p.6 | EVIDENCE_ALIGNED | The filing explicitly says NVIDIA serves the professional visualization market. |
| 10 | `ADVANCED_MICRO_DEVICES --PRODUCES--> GPU` | 2023 p.10, 2024 p.9, 2025 p.9 | WEAK_OR_OVERGENERALIZED | AMD is described as a supplier/licensor of GPU solutions; supplier is not necessarily identical to producer. |
| 11 | `UNITED_STATES --EXPOSED_TO--> GOVERNMENT_REGULATION_RISK` | 2023 p.11, 2024 p.10, 2025 p.10 | LIKELY_MISLABELED | The text says NVIDIA’s worldwide business is subject to US and foreign rules; the source endpoint is likely the company, not the United States. |
| 12 | `CYBERATTACKS --DECREASES--> FINANCIAL_CONDITION` | 2023 p.16, 2024 p.14, 2025 p.14 | EVIDENCE_ALIGNED | The filing explicitly says cyber-attacks could adversely affect financial condition; the modal “could” must be preserved. |
| 13 | `PRODUCT_TRANSITIONS --DECREASES--> REVENUE` | 2023 p.19, 2025 p.17 | EVIDENCE_ALIGNED | The filing explicitly says product transitions negatively impact revenue. |
| 14 | `PRODUCT_TRANSITION_RISK --DECREASES--> REVENUE` | 2023 p.19, 2025 p.17 | EVIDENCE_ALIGNED | The passages state that qualification and channel effects can reduce or create volatility in revenue. |
| 15 | `SUPPLY_CHAIN_DISRUPTION --DECREASES--> REVENUE` | 2023 p.20, 2024 p.18 | EVIDENCE_ALIGNED | The text explicitly describes adverse effects on revenue and financial results. |
| 16 | `COVID_19 --CAUSES--> SUPPLY_CHAIN_DISRUPTION` | 2023 p.20, p.27, 2023 p.37 | EVIDENCE_ALIGNED | The passages directly connect COVID-related disruption or pandemic constraints with supply-chain/logistics constraints. |
| 17 | `NATURAL_DISASTER --CAUSES--> SUPPLY_CHAIN_DISRUPTION` | 2023 p.20, 2024 p.17, 2025 p.17 | EVIDENCE_ALIGNED | The text explicitly lists natural disasters as causes of supply constraints. |
| 18 | `MACROECONOMIC_CONDITIONS --INCREASES--> CAPEX` | 2023 p.22, 2024 p.19, 2025 p.19 | LIKELY_MISLABELED | The representative text discusses lower capital expenditures as an adverse effect, not an increase in capex. |
| 19 | `FOREIGN_EXCHANGE_RISK --INCREASES--> CAPEX` | 2023 p.22, 2024 p.19, 2025 p.19 | LIKELY_MISLABELED | Currency fluctuations are listed among uncertainties; the passage does not support an increase-in-capex direction. |
| 20 | `CYBERATTACKS --DECREASES--> REVENUE` | 2023 p.23, 2024 p.20, 2025 p.20 | EVIDENCE_ALIGNED | The filing explicitly says cyber-attacks could reduce expected revenue; the modal “could” must remain in any answer. |

## Interpretation

The sample contains 8 evidence-aligned groups, 8 weak/overgeneralized groups,
and 4 likely mislabeled groups. This is not a graph-wide error rate because
the 20 groups are an audit sample and many groups contain multiple claims.
The main risk is concentrated in relation semantics and endpoint grounding,
especially broad `PRODUCES` mappings and direction/type mistakes. It is not a
PDF quote-traceability failure: the machine audit found zero normalized quote
mismatches and zero obvious ontology direction/type conflicts in its structural
checks, while this review identifies semantic risks that structural checks
cannot prove or disprove.

## Required follow-up

1. Keep all evidence variants under the current freeze; do not deduplicate by
   deleting claims.
2. Use rows 4, 11, 18, and 19 as high-priority candidates for a future
   extraction/ontology repair experiment, but obtain approval before changing
   the graph or regenerating derived temporal artifacts.
3. Use rows 1–3, 5–8, and 10 as boundary cases when constructing the minimum
   30-row human Golden QA set. The reviewer should decide whether the exact
   predicate is supported, not whether the company is financially good or bad.
4. Treat rows 9 and 12–17, 20 as examples of clearly supported relations, while
   preserving the filing’s uncertainty words such as `could` and `may`.
