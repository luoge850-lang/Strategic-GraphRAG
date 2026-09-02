# Strategic-GraphRAG v3.1

Evidence-grounded GraphRAG for NVIDIA's fiscal 2023, 2024, and 2025 10-K
filings. The project turns SEC PDFs into a strict Neo4j evidence graph,
combines graph traversal with filing-scoped vector retrieval, and returns
structured answers whose citations can be joined back to verbatim PDF text.

> Research status as of 2026-09-01: the engineering baseline is usable, and
> the 30-row 2025 post-repair AI-assisted working set is labeled. The historical
> `post_repair_v2` file is a prefilled candidate audit, not independent human
> gold. Recall/F1 and the retrieval/answer benchmark remain blocked until a
> blind human relation inventory and a human-reviewed Golden QA set are available.

![Strategic-GraphRAG dashboard](docs/demo-dashboard-v3.png)

## Verified scope

| Filing | Pages | Strict EvidenceClaims | Evidence pages | Vector chunks |
|---|---:|---:|---:|---:|
| 2023 10-K | 169 | 126 | 44 | 678 |
| 2024 10-K | 96 | 129 | 41 | 425 |
| 2025 10-K | 130 | 128 | 46 | 583 |
| **Total** | **395** | **383** | **131 filing-page pairs** | **1,686** |

The active graph has 383 strict business edges, each linked to a `VERBATIM`
EvidenceClaim with filing, page, section, chunk, source entity, target entity,
and relation metadata. All 383 claims use content-derived `claim_v2_*` IDs.
The post-migration audit found 49 valid same-filing two-hop paths and zero
invalid strict paths.

The historical `extraction_sample_2025_post_repair_v2.jsonl` is a prefilled
candidate audit. The current `extraction_sample_2025_post_repair_human_v1.jsonl`
contains 30 rows labeled with GPT-5.6/Sol assistance: source entity 29/30,
target entity 29/30, relation 22/30, and evidence support 22/30. The relation
and evidence fields each contain 7 `uncertain` labels and 1 explicit `false`
label. These are precision-like sample estimates from a different, 2025-only
sample than the original 60-row three-filing baseline; they are not Recall/F1
and should not be reported as a statistically significant before/after result.
Neither file is an independent human Golden QA set and neither should be
reported as one.
The machine-readable readiness audit is
`reports/2026-08-28_research_readiness.json` and is intentionally fail-closed.
The historical 30-row machine-readable annotation audit is
`reports/extraction_annotation_audit_2025_post_repair_v2.json`; the newer
`human_v1` file is an AI-assisted working set rather than a human gold set. The older
`extraction_quality_2025_post_repair_v2.json` file is historical and must not
be cited as the current result.

Legacy storage is now physically isolated: the post-clean check found zero
out-of-scope business edges, zero old evidence nodes, and zero old Chroma
collections. A complete local recovery archive was created before deletion but
is intentionally not committed because it contains embeddings and extracted
filing text.

## Architecture

```text
Three allowlisted 10-K PDFs
  -> page parsing and SEC section detection
  -> overlapping text chunks plus financial-table rows
  -> rules + DeepSeek Flash extraction
  -> ontology, quote/span, and entity validation
  -> Neo4j business edge + EvidenceClaim + Sentence provenance
  -> Chroma semantic chunks
  -> query router -> Vector / Graph / Hybrid / Hybrid+Temporal
  -> directed path search + personalized PageRank (PPR)
  -> grounded structured synthesis
  -> FastAPI + React/Vite evidence UI
```

Key implementation decisions:

- Exact metric-only questions route to `REPORTS_METRIC` facts. Causal questions
  that mention a metric remain Hybrid so vector evidence pages can expand graph
  anchors before path search.
- Every synthesized citation is checked against the returned path evidence.
- Financial-table claims are normalized into 237 period-specific
  `FinancialObservation` nodes linked to company, metric, filing, fiscal year,
  and the exact supporting `EvidenceClaim`. Percentage-of-revenue denominator
  rows are excluded from amount retrieval; genuine percentage rows are mapped
  to margin/ratio metrics.
- All 383 strict claims are represented as `TemporalFact` versions using
  separate valid-time and recorded-time fields. The current
  `bitemporal_fact_v2` migration marks 124 versions `ACTIVE_CURRENT` and 259
  `SUPERSEDED_DISCLOSURE`; supersession means a later disclosure version exists,
  not that the earlier real-world assertion became false.
- The 99 `TemporalChange` nodes now link fact versions and supporting claims:
  58 continued, 6 recurred, 19 metric increases, 3 metric decreases, and 13
  non-comparable metric changes. The model never infers resolution from silence.
- `QueryRouter` exposes four reproducible modes: `vector`, `graph`, `hybrid`,
  and `hybrid_temporal`. Hybrid modes use vector-to-graph anchor expansion and
  PPR; Hybrid+Temporal additionally scores bitemporal fact matches.
- Identical successful API requests can use a bounded TTL cache. Responses
  expose `cache.hit`, selected retrieval mode, and per-stage latency so cached
  and uncached performance are not mixed.
- An incremental planner compares PDF SHA-256 values before rebuilding. The
  current plan reports all three PDFs unchanged and `requires_rebuild=[]`.

## Demonstrated query

`Compare revenue in 2023, 2024, and 2025`

The current engine retrieves three `REPORTS_METRIC` claims and reports:

- FY2023: $26,974 million (`p.86`)
- FY2024: $60,922 million (`p.79`)
- FY2025: $130,497 million (`p.80`)

The response was grounding-verified and used one stable EvidenceClaim ID per
filing. It does not infer the causes of revenue growth from those accounting
facts alone.

## Run locally

Python 3.11 or 3.12 is recommended. Python 3.14 can emit compatibility warnings
from some LangChain/Pydantic dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-hybrid.txt
Copy-Item .env.example .env
uvicorn strategic_graphrag.api.server:app --host 127.0.0.1 --port 8000
```

Build the frontend:

```powershell
cd frontend
npm install
npm run build
```

FastAPI serves `frontend/dist` at `http://127.0.0.1:8000/`. Configure Neo4j,
DeepSeek, the active vector collection, CORS, optional API authentication, rate
limits, cache TTL, and Cross-Encoder behavior in `.env`; never commit `.env`.

After building the frontend, the repeatable Windows launcher is:

```powershell
.\scripts\start_demo.ps1
# Controlled restart:
.\scripts\start_demo.ps1 -Restart
```

The launcher waits for `/health/live`; `/health/ready` can still report
`degraded` when Neo4j or an external LLM is unavailable.

## Reproducibility and checks

```powershell
python -m pytest -q
python scripts/check_runtime.py
python -m compileall -q strategic_graphrag scripts tests
python scripts/plan_incremental_update.py `
  --manifest reports/2026-08-14_corpus_manifest.json `
  --output reports/incremental_plan.json
python scripts/audit_strict_chains.py --output reports/strict_chains.json
python scripts/audit_research_readiness.py
python scripts/migrate_financial_observations.py --apply
python scripts/build_temporal_change_model.py --apply
python scripts/run_retrieval_baselines.py `
  --question "How did NVIDIA revenue change between 2023 and 2025?" `
  --cross-filing `
  --output reports/retrieval_baselines_smoke.json
cd frontend
npm run build
```

The current release passed 48 Python tests, including 43 focused Python
contracts plus reproducibility checks, Python compilation,
frontend TypeScript/Vite production build, Neo4j/Chroma post-clean checks, stable
ID consistency, strict path validation, API health, and browser rendering. The
largest JavaScript chunk is about 422 kB after splitting React, Motion,
vis-data, and vis-network.

In the latest local check, the all-filing statistics endpoint took 4.07 s cold
and 5-19 ms cached; the visualization subgraph took 1.44 s cold and 25-32 ms
cached. A retrieval-only cross-filing smoke test returned three strict revenue
paths for 2023-2025 in Graph, Hybrid, and Hybrid+Temporal modes. Single-run
latencies were 10.12 s, 3.91 s, and 2.41 s respectively; Vector retrieval took
35.87 ms. Aura cold starts and cache effects make these development
observations, not benchmark guarantees.

## Research status and honest limitations

This is a strong engineering candidate, not yet a completed research result:

- Automated provenance checks passed for all 383 claims: 100% declared-page
  verbatim match, 100% required provenance completeness, one linked business
  edge per claim, and zero exact duplicates. The legacy 60-claim stratified
  sample has status fields marked `LABELED`, but lacks verifiable human metadata
  such as annotator identity and notes. It is therefore a legacy/prefilled
  diagnostic set, not a completed independent human annotation pass. Under the
  declared protocol (`uncertain` counted as incorrect), relation correctness
  and evidence support are both 32/60 (53.33%), and exact-triple correctness is
  32/60 (53.33%) for this historical prefilled set. These are precision-like
  diagnostic estimates, not recall or F1: a complete gold relation inventory
  and an independent adjudication pass are still required.
  The duplicate-triple audit found 8 repeated logical triples across 21 rows;
  7 groups have different evidence, 1 group repeats the same evidence, and
  duplicate-triple label agreement is 0.75. See
  `reports/extraction_annotation_audit_v1.json` for the machine-readable audit.
- The existing 38-item auto-generated QA file is stale after the evidence-ID
  migration and is not a valid Golden QA benchmark. There is no independent
  human Golden QA in the current checkout; the current evaluation artifact is
  only an AI-assisted working set (`human_v1`).
- A real DeepSeek Flash Hybrid query and the corresponding browser flow were
  tested across all three filings. This is a smoke test, not a Golden QA score.
- Filing disclosures support attributed relationships; they do not prove
  counterfactual causality, effect size, probability, or investment outcomes.
- The current `human_v1` extraction labels are AI-assisted by GPT-5.6/Sol and
  carry an explicit AI annotator identity, but have no independent human
  adjudication. Duplicate-triple agreement is a diagnostic only and must not
  be reported as inter-annotator agreement.
- The 30-row `human_v1` AI-assisted working set is complete as an AI-assisted
  artifact, but its relation/evidence scores of 22/30 are not comparable
  evidence of improvement because its sampling frame differs from the original
  60-row baseline. An independent human blind pass and matched sampling are
  required before claiming Recall, F1, or improvement.
- Extraction runs record `LLM_EXTRACTION_TEMPERATURE` in their statistics.
  Fixing it at 0.0 reduces sampling randomness for comparisons but does not
  guarantee identical responses from an external model service.
- `bitemporal_fact_v2` separates valid and recorded time and supports explicit
  invalidation/supersession links. Migrated records use a labeled migration
  timestamp because the historical database-write time is unknown. Narrative
  intensified/mitigated/resolved labels and an independently labeled temporal
  benchmark remain open.
- The four retrieval modes are implemented and smoke-tested, but they are not
  yet accuracy baselines: a labeled QA/evidence set is still required for
  Recall@K, Precision@K, faithfulness, answer relevance, and significance tests.
- API authentication is configurable but disabled in the local demo. It must be
  enabled with restricted CORS before public deployment.
- DeepSeek Flash is an external processor. Production use needs documented data
  governance, consent, retention, and provider-failure behavior.

See [the P0/P1 acceptance audit](reports/2026-08-14_p0_p1_acceptance.md) and the
[machine-readable corpus manifest](reports/2026-08-14_corpus_manifest.json).
The current frozen baseline is documented in the
[v3.1 release notes](reports/2026-08-17_v3.1_release_notes.md) and
[v3.1 freeze manifest](reports/2026-08-17_v3.1_freeze_manifest.json).
The v3.1 manifest is a historical freeze from 2026-08-17; the post-repair
2025-only audit is a subsequent working state. The full evaluation contract is
documented in [research_evaluation_protocol.md](docs/research_evaluation_protocol.md).
The delivery cleanup and retained/deleted-file decisions are recorded in
[delivery_cleanup_manifest.md](docs/delivery_cleanup_manifest.md).

## Next research milestones

1. First create a blind human annotation set with a complete gold relation
   inventory and an independent adjudication pass; only then report
   entity/relation precision, recall, F1, and error categories.
2. Build a 30-50 question independently human-reviewed Golden QA set with stable evidence IDs and
   unanswerable cases; report retrieval Recall@K, Precision@K, faithfulness,
   answer relevance, abstention accuracy, and latency distributions.
3. Extend observed numeric changes with independently labeled narrative states
   such as new, intensified, mitigated, and resolved; evaluate them separately.
4. Evaluate the four implemented retrieval baselines, then add reranker and
   evidence-guard ablations only after the main-model benchmark is stable.
5. Containerize and deploy behind authentication, restricted CORS, observability,
   request timeouts, and cost controls.

## License and data

Code is intended for academic and portfolio use. SEC filings, model APIs, and
third-party libraries retain their own licenses and terms. PDFs, vector stores,
credentials, and large local audit archives are excluded from Git.
