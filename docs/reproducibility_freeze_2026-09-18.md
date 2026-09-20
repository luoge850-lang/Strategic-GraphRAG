# Reproducibility freeze — 2026-09-18

## Frozen experiment boundary

- Corpus: the three allowlisted NVIDIA 10-K PDFs for fiscal 2023, 2024, and
  2025.
- Active claim-ID schema: `claim_v2_*`.
- Active 2025 filing: `2025-10-K.pdf`.
- Graph baseline: 381 `VERBATIM` EvidenceClaims across 126/129/126 claims per
  filing; no graph rebuild is part of this freeze.
- Vector baseline: Chroma collection `nvidia_sec_filings_active`, 1,686 chunks,
  `all-MiniLM-L6-v2`.
- Extraction model: DeepSeek V4 Flash, extraction temperature `0.0`, prompt
  version `v2-evidence-claim-1`.
- Answer synthesis: DeepSeek V4 Flash, current report contract, temperature
  `0.3`.
- Answer-level judge: DeepSeek V4 Flash, prompt version
  `answer-level-judge-v1`, temperature `0.0`.

## What was fixed

Fresh external calls were not exactly repeatable: two frozen 2025 dry runs
produced 126 and 131 accepted claims at the same nominal temperature. This is
provider-side variation, not evidence that the Neo4j graph was rebuilt or that
the PDFs changed. The operational fix is to use the versioned response cache
for reproducible extraction experiments:

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

The frozen record/replay pair has 126 claims in both runs, 170 unique cache
keys, 170 record network calls, and zero replay network calls. Replay fails
closed on a cache miss. This makes the engineering experiment reproducible;
it does not make a fresh external model deterministic and must not be reported
as fresh-model exact repeatability.

## Evaluation commands

Use the canonical environment and commands below:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/audit_research_readiness.py `
  --output reports/research_readiness_2026-09-16.json
.\.venv\Scripts\python.exe scripts/run_golden_evaluation.py `
  --dataset evaluation/golden_qa_human_v2.jsonl `
  --base-url http://127.0.0.1:8000 --judge `
  --synthesize `
  --output reports/golden_qa_human_v2_answer_level_2026-09-18_ranking_v2.json
```

The canonical retrieval-only Silver run uses
`scripts/evaluate_retrieval_benchmark.py`. It disables answer synthesis,
remote LLM anchor expansion, and the optional cross-encoder so that the four
retrieval modes are compared under a fixed retrieval contract. The current
post-ranking report is
`reports/retrieval_benchmark_silver_2026-09-18_ranking_v2_metrics.json`.
The raw four-mode smoke trace is retained separately at
`reports/retrieval_benchmark_silver_2026-09-18_ranking_v2.json`.

The current ranking contract is deliberately two-stage: deduplicate semantic
paths while retaining evidence variants, score directness and evidence role,
then select from a bounded candidate pool. An `ANSWER_CRITICAL` path cannot be
discarded merely because a background path had a higher pre-policy aggregate
score. Temporal selection additionally reserves the best available path for
each requested fiscal year.

### Engineering-only latency hardening — 2026-09-20

The frozen PDFs, Neo4j claim IDs, vector collection, and benchmark questions
were not changed. The following read-only changes are now part of the current
runtime contract:

- PPR results use a bounded five-minute process-local cache keyed by anchors,
  filing, year range, and limit. Cache entries are copied on read and are
  never written to Neo4j.
- Temporal fact fusion resolves the bounded claim-ID list through the indexed
  `EvidenceClaim` lookup before traversing `SUPPORTED_BY_CLAIM` edges.
- API telemetry reports `anchor_resolution_ms` and `ppr_ms` separately, so a
  future latency report can distinguish entity resolution from propagation.
- Structural evidence ranking remains fail-closed: endpoint alignment and a
  supported direct predicate are required for explicit `PRODUCES`/`OPERATES_IN`
  answer-critical paths; `includes`, `runs on`, and `based on` variants remain
  background evidence.

These changes may reduce repeated Graph/Hybrid latency, but they do not prove a
new retrieval-quality improvement until the four-mode Silver benchmark and the
answer-level audit are rerun under the same frozen protocol. The PPR cache is
also not a substitute for a cold-start measurement.

The Demo is intentionally not configured for boot-time auto-start. To start
the local service after entering Codex, run:

```powershell
.\scripts\start_demo.ps1 -Restart
```

The readiness endpoint must report `ready`, with Neo4j, Chroma, and the LLM
available, before a benchmark is started.

## Remaining limitation

The readiness audit remains `NOT_READY` only because the fresh external-model
repeatability gate still observes 126 versus 131. The cache replay gate passes.
Historical report artifacts are classified in
`docs/report_archive_manifest_2026-09-18.md`; no report was irreversibly
deleted. Neither the fresh-model limitation nor the single-reviewer Golden QA
limitation is hidden by the Silver benchmark.

The candidate-conditioned answer-level evaluation is an engineering audit over
30 rows. Because the same question text is intentionally repeated for some
evidence candidates, the safer answer-level view is the derived
`reports/golden_qa_human_v2_question_level_answer_2026-09-18_semantic_gate.json`
report: 20 unique questions, 5 answerable and 15 abstention-required. It keeps
three conflicting candidate-label groups visible rather than silently deleting
them. Both views are suitable for regression analysis, but neither substitutes
for a second independent reviewer or a larger finance-domain benchmark.

The extraction code now rejects future structural `PRODUCES` candidates whose
evidence only expresses membership, runtime, or composition. A replayed 2025
dry run retained 112 strict candidates versus 126 in the current graph. No
Neo4j write was made in this optimization pass: a complete rebuild would need
equivalent frozen extraction responses for all three filings before it could be
called reproducible.
