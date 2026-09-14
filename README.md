# Strategic-GraphRAG v3.1

Evidence-grounded GraphRAG for NVIDIA's fiscal 2023, 2024, and 2025 10-K
filings. The project turns SEC PDFs into a strict Neo4j evidence graph,
combines graph traversal with filing-scoped vector retrieval, and returns
structured answers whose citations can be joined back to verbatim PDF text.

> Research status as of 2026-09-14: the engineering baseline is usable, the
> strict graph audit passes, and an automatic 37-question Silver retrieval
> regression is available for all four modes. This is not human gold. The
> fail-closed readiness audit remains `NOT_READY` because independent human
> Golden QA is incomplete and a repeated external-LLM extraction produced 126
> versus 131 accepted claims at temperature 0.0; neither issue is hidden behind
> a fabricated score.

![Strategic-GraphRAG dashboard](docs/demo-dashboard-v3.png)

## Verified scope

| Filing | Pages | Strict EvidenceClaims | Evidence pages | Vector chunks |
|---|---:|---:|---:|---:|
| 2023 10-K | 169 | 126 | 44 | 678 |
| 2024 10-K | 96 | 129 | 41 | 425 |
| 2025 10-K | 130 | 126 | 43 | 583 |
| **Total** | **395** | **381** | **128 filing-page pairs** | **1,686** |

The active graph has 381 strict business edges, each linked to a `VERBATIM`
EvidenceClaim with filing, page, section, chunk, source entity, target entity,
and relation metadata. All 381 active claims use content-derived `claim_v2_*` IDs.
The post-rebuild strict-chain audit found 9 valid same-filing two-hop paths and
zero invalid strict paths.

The historical `extraction_sample_2025_post_repair_v2.jsonl` is a prefilled
candidate audit. The latest structural annotation audit reports source entity
29/30, target entity 29/30, relation 27/30, and evidence support 27/30 when
`uncertain` is counted as incorrect. Two sampled IDs no longer exist after the
2025 replacement and must be remapped before this sample is used as a current
benchmark. The separate
`extraction_sample_2025_post_repair_human_v1.jsonl` contains 30 rows labeled
with GPT-5.6/Sol assistance: relation 22/30 and evidence support 22/30. The relation
and evidence fields each contain 7 `uncertain` labels and 1 explicit `false`
label. These are precision-like sample estimates from a different, 2025-only
sample than the original 60-row three-filing baseline; they are not Recall/F1
and should not be reported as a statistically significant before/after result.
Neither file is an independent human Golden QA set and neither should be
reported as one.
The machine-readable readiness audit is
`reports/research_readiness_current.json` and is intentionally fail-closed.
The latest 30-row machine-readable annotation audit is
`reports/extraction_annotation_audit_2025_post_rebuild_2026-09-03.json`; the
historical audit is
`reports/extraction_annotation_audit_2025_post_repair_v2.json`; the newer
`human_v1` file is an AI-assisted working set rather than a human gold set. The older
`extraction_quality_2025_post_repair_v2.json` file is historical, retained under
`archive/cleanup-2026-09-14/historical-reports/`, and must not be cited as the
current result.

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
- The graph also contains bitemporal and temporal-change models. Because the
  2025 filing was subsequently replaced with a stricter 126-claim set, these
  derived artifacts must be regenerated before temporal counts or temporal
  accuracy are reported. The post-rebuild snapshot explicitly records this
  dependency instead of treating the older 383-fact materialization as current.
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

If `/health/live` is `alive` but `/health/ready` is `503`, the frontend process
is running and the missing graph is an external dependency problem. Re-copy the
current Neo4j Aura connection URI and database name from the Aura Connect panel
into `.env`; do not infer a new URI from an old database ID. Then run
`.\scripts\start_demo.ps1 -Restart` and reload the page.

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

The current working tree passed 84 Python tests, including focused Python
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

- Automated provenance checks passed for all 381 active claims: 100% declared-page
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
  `archive/cleanup-2026-09-14/historical-reports/extraction_annotation_audit_v1.json`
  for the historical machine-readable audit.
- The existing 38-item auto-generated QA file is stale after the evidence-ID
  migration and is not a valid Golden QA benchmark. There is no independent
  human Golden QA in the current checkout; the current extraction-annotation
  artifact is only an AI-assisted working set (`human_v1`). The separate
  retrieval artifact is an auto-generated Silver regression, not human gold.
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
- The four retrieval modes are implemented and evaluated on the automatic
  Silver set in `reports/retrieval_benchmark_silver_2026-09-14.json`. On its
  common page-level unit and 32 answerable questions, the observed macro
  Recall@5/MRR are Vector 0.2188/0.0828, Graph 0.7009/0.7083, Hybrid
  0.5134/0.4740, and Hybrid Temporal 0.5446/0.5000. Graph is higher than
  Vector on this self-generated Silver regression, but this is not evidence
  that GraphRAG is generally superior to Vector RAG: expected pages are
  derived from the graph under test, so the benchmark has self-test bias and
  no independent human Golden QA. Hybrid Temporal did not exceed Graph on this
  snapshot; with only two temporal-metric questions, that does not establish
  that the temporal module is ineffective. The report now includes
  question-type strata, question-ID paired win/tie/loss counts, and descriptive
  question-level bootstrap intervals. A human-reviewed QA/evidence set remains
  required for paper-level Recall@K, Precision@K, faithfulness, answer
  relevance, and significance tests.
- API authentication is configurable but disabled in the local demo. It must be
  enabled with restricted CORS before public deployment.
- DeepSeek Flash is an external processor. Production use needs documented data
  governance, consent, retention, and provider-failure behavior.

See [the P0/P1 acceptance audit](reports/2026-08-14_p0_p1_acceptance.md) and the
[machine-readable corpus manifest](reports/2026-08-14_corpus_manifest.json).
The current frozen baseline is documented in the
[v3.1 release notes](reports/2026-08-17_v3.1_release_notes.md) and
[v3.1 freeze manifest](reports/2026-08-17_v3.1_freeze_manifest.json).
The v3.1 manifest is a historical freeze from 2026-08-17; the post-rebuild
2025-only audit is a subsequent working state. The current graph snapshot is
`reports/neo4j_snapshot_post_rebuild_derived_2026-09-05.json`, and the
full evaluation contract is
documented in [research_evaluation_protocol.md](docs/research_evaluation_protocol.md).
The delivery cleanup and retained/deleted-file decisions are recorded in
[delivery_cleanup_manifest.md](docs/delivery_cleanup_manifest.md).
The graduation-project and application-material positioning is summarized in
[graduation_application_brief.md](docs/graduation_application_brief.md).

## Next research milestones

1. **Completed:** keep the automatic Silver benchmark as an engineering
   regression and implement the versioned extraction response cache. The 2025
   record/replay gate passed with 126 accepted claims in both runs, 170 unique
   keys, 170 record network calls, and 0 replay network calls. This demonstrates
   deterministic replay of the frozen response artifact only; it is not fresh
   external-model repeatability, semantic correctness, or human gold.
2. **Deferred:** create a blind human annotation set with a complete gold
   relation inventory and an independent adjudication pass; only then report
   entity/relation precision, recall, F1, and error categories.
3. **Deferred:** build a 30-50 question independently human-reviewed Golden QA
   set with stable evidence IDs and unanswerable cases; report retrieval
   Recall@K, Precision@K, faithfulness, answer relevance, abstention accuracy,
   and latency distributions.
4. Extend observed numeric changes with independently labeled narrative states
   such as new, intensified, mitigated, and resolved; evaluate them separately.
5. Run a targeted, human-reviewed temporal ablation: add cross-filing questions
   with independently verified valid-time/recorded-time evidence, hold the
   candidate budget fixed, compare Hybrid with Hybrid Temporal by temporal and
   non-temporal strata, and report temporal-field/path-hit diagnostics. Refresh
   derived temporal artifacts from the current graph and freeze/cache external
   extraction responses before interpreting the result.
6. Evaluate the four implemented retrieval baselines, then add reranker and
   evidence-guard ablations only after the main-model benchmark is stable.
7. Containerize and deploy behind authentication, restricted CORS, observability,
   request timeouts, and cost controls.

### 2025 extraction cache dry-run

Use PowerShell environment variables for the cache mode and path; these commands
do not modify `.env`:

```powershell
$env:LLM_RESPONSE_CACHE_PATH = "evaluation/cache/llm_extraction_v1.jsonl"
$env:LLM_RESPONSE_CACHE_MODE = "record"
python -m strategic_graphrag.pipeline.pipeline `
  --pdf data/pdfs/2025-10-K.pdf --require_llm --dry_run `
  --output_stats reports/rebuild_2025_cache_record_2026-09-14.json

$env:LLM_RESPONSE_CACHE_MODE = "replay"
python -m strategic_graphrag.pipeline.pipeline `
  --pdf data/pdfs/2025-10-K.pdf --require_llm --dry_run `
  --output_stats reports/rebuild_2025_cache_replay_2026-09-14.json
```

Reports record cache path, schema, mode, key count, hits/misses/writes, and
network-call count. The cache is auditable JSONL and stores structured responses
plus prompt digests, not a separate raw PDF prompt payload or API keys; its
structured evidence fields remain available for deterministic replay. It is not
a human Golden QA dataset.

## License and data

Code is intended for academic and portfolio use. SEC filings, model APIs, and
third-party libraries retain their own licenses and terms. PDFs, vector stores,
credentials, and large local audit archives are excluded from Git.
