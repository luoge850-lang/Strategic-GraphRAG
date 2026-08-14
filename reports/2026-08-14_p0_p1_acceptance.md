# P0/P1 Runtime and Research Acceptance Audit

Date: 2026-08-14  
Scope: NVIDIA fiscal 2023, 2024, and 2025 10-K only

## Decision

The local three-filing candidate now passes the requested P0 runtime gates and
implements a defensible first P1 baseline. It is suitable for a live portfolio
demo and for beginning supervised evaluation, but it is not yet a publishable
research result because semantic extraction precision/recall and Golden QA
metrics remain unlabeled.

## P0 acceptance

| Gate | Before | Accepted result | Status |
|---|---|---|---|
| Process liveness | `/health` waited on Aura and LLM setup | `/health/live` 5-22 ms; no external dependency calls | PASS |
| Dependency readiness | first request timed out and duplicated model loading | background Hybrid warm-up + initialization locks; warm readiness 576 ms | PASS |
| All-filing statistics | 15-30 s timeout observed | 722 ms cold, 25 ms cached | PASS |
| 500-node subgraph | query formed a nodes x relationships Cartesian product | 1.04 s cold, 12 ms cached | PASS |
| Demo graph | blank when either graph request stalled | 130 rendered nodes / 383 strict edges; explicit loading and error state | PASS |
| Hybrid runtime | silently fell back because `sentence_transformers` was absent | vector status `OK`, five hits, four graph paths page-matched | PASS |
| Frontend production build | not verified in repaired environment | TypeScript/Vite build succeeded, 471 modules | PASS |

The runtime is now Python 3.12.13. Core dependencies are separated from the
Hybrid profile (`requirements-hybrid.txt`). The active vector collection was
rebuilt from only the three allowlisted PDFs and contains 1,686 chunks.

## Hybrid query acceptance

Question: `How do US export controls impact NVIDIA revenue?`

The browser rendered a grounded answer with export-control evidence rather
than unrelated revenue-only paths. The leading paths included:

- FY2024 `CHIP_EXPORT_RESTRICTION -> DECREASES -> REVENUE`, EvidenceClaim
  `claim_v2_722660fd0f5d3d33717b7c07`, page 26.
- FY2025 `CHIP_EXPORT_RESTRICTION -> DECREASES -> REVENUE`, EvidenceClaim
  `claim_v2_ff037d9774c3dc183cc569a5`, page 28.

The answer correctly distinguished disclosed forward-looking exposure from a
realized, quantified revenue loss. An uncached explicit Hybrid run took 16.07 s:
29.68 ms vector retrieval, 2.28 s anchor resolution, 1.55 s graph path search,
and 8.39 s DeepSeek Flash synthesis. LLM synthesis is the dominant latency.

## P1 extraction-quality baseline

The automated audit covered all 383 active `VERBATIM` EvidenceClaims:

- declared-page verbatim match: 383/383 (100%);
- required provenance completeness: 383/383 (100%);
- exactly one linked business edge: 383/383 (100%);
- exact duplicate excess claims: 0;
- distribution: 126 FY2023, 129 FY2024, 128 FY2025;
- methods: 169 rules, 92 LLM, 122 table extraction.

These are provenance and schema metrics, not semantic accuracy. The audit
deliberately reports entity precision, relation precision, evidence-support
precision, and relation recall as `NOT_MEASURED`. A deterministic 60-claim
sample stratified by filing, section, and extraction method is stored at
`evaluation/annotation/extraction_sample_v1.jsonl`. Independent labeling is
the next required step; self-labeling would inflate the result.

## P1 temporal model

`NEXT_DISCLOSURE` remains an ordering relation. The new
`observed_change_v1` layer adds 99 `TemporalChange` nodes, each linking an
earlier and later `VERBATIM` EvidenceClaim and storing transaction years,
valid-period strings, model version, and comparison semantics.

Current classes:

- 58 consecutive narrative disclosures;
- 6 non-consecutive recurring disclosures;
- 19 observed metric increases;
- 3 observed metric decreases;
- 13 metric pairs rejected as non-comparable.

The comparability guard requires matching units and matching currency/percent
evidence signatures. This was added after an audit found that two claims named
`cost_of_revenue` came from incompatible table contexts despite having the same
normalized target. The model never infers resolution from absence and never
assigns narrative increase/decrease without comparable numeric evidence.

## Remaining risks and priority

1. **P1 semantic labels:** label the prepared 60-claim sample and a negative
   page sample to obtain extraction precision, recall, F1, and error classes.
2. **P1 temporal labels:** create an independent benchmark for new,
   intensified, mitigated, and resolved narrative states. The current model is
   observed numeric change plus recurrence, not full event-time reasoning.
3. **P1 retrieval evaluation:** create 30-50 human Golden QA items, including
   unanswerable questions, then report EvidenceClaim Recall@K/Precision@K,
   faithfulness, answer relevance, abstention accuracy, p50/p95 latency.
4. **P2 latency:** DeepSeek synthesis and LLM anchor extraction dominate the
   request. Add anchor caching, bounded provider timeouts, and streaming before
   public deployment.
5. **P2 deployment:** enable API authentication, restricted CORS,
   observability, cost limits, container health probes, and secret management.

## External design references used

- Microsoft GraphRAG indexing/dataflow: explicit Documents, TextUnits,
  Entities, Relationships and optional Claims; versioned outputs and prompt
  tuning are part of the indexing contract.
  <https://microsoft.github.io/graphrag/index/overview/>
- Microsoft GraphRAG local search: combine graph structures with source text
  rather than treating vector snippets as independently citable graph facts.
  <https://microsoft.github.io/graphrag/query/overview/>
- Neo4j Python performance guidance: specify the database, use read routing,
  reduce transactions, set query timeouts, and profile/index hot queries.
  <https://neo4j.com/docs/python-manual/current/performance/>
- Ragas context recall: retrieval recall requires reference answers or exact
  reference context IDs; it cannot be inferred from page coverage.
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/>
- Temporal KG survey: temporal facts require timestamps/intervals and evolving
  structure; omission alone does not prove state resolution.
  <https://arxiv.org/abs/2201.08236>
