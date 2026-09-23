# Public results summary — development branch

This is a compact, source-linked summary for the public repository. It is not
a replacement for raw reports. All values below are tied to the current
development evidence boundary and must not be copied to the `stable` release
without a new run.

## Snapshot identity

| Item | Value |
|---|---|
| Repository | `luoge850-lang/Strategic-GraphRAG` |
| Branch | `codex/v3-three-filing-evidence-graphrag` |
| Corpus | NVIDIA fiscal 2023/2024/2025 10-K; 395 physical pages |
| Active graph inventory | 381 strict EvidenceClaims: 126 / 129 / 126 by filing |
| Vector inventory | 1,686 chunks; `all-MiniLM-L6-v2` |
| Silver set | 37 auto-generated questions: 32 answerable, 5 abstention-required |
| Golden QA | 30 rows, one reviewer, engineering-only checkpoint |
| Readiness | `NOT_READY`: fresh external-model extraction repeatability gate |

Source: `docs/canonical_project_state_2026-09-19.md`,
`docs/reproducibility_freeze_2026-09-18.md`, and the local machine-readable
reports named in those documents.

## Retrieval regression

The four modes use the same Silver questions and page-level evaluation contract.
These numbers are retrieval observations, not end-to-end answer accuracy.

| Mode | Precision@5 | Recall@5 | nDCG@5 | MRR | Answerable n |
|---|---:|---:|---:|---:|---:|
| Vector | 0.0500 | 0.2188 | 0.1103 | 0.0828 | 32 |
| Graph | 0.2313 | 0.8259 | 0.8079 | 0.8073 | 32 |
| Hybrid | 0.2313 | 0.8259 | 0.7733 | 0.7604 | 32 |
| Hybrid+Temporal | 0.2250 | 0.7946 | 0.7577 | 0.7500 | 32 |

Source report: local `reports/retrieval_benchmark_silver_2026-09-18_ranking_v2_metrics.json`.
The report is intentionally not treated as independent gold: expected evidence
is derived from the graph under test. The defensible statement is “Graph scored
higher than Vector on this graph-derived Silver regression,” not “GraphRAG is
generally superior to Vector RAG.”

## Runtime smoke profile

The 2026-09-20 retrieval-only smoke run used four Silver questions, concurrency
1, synthesis disabled, and separate API cache-miss/cache-hit-or-fill phases.
`cache_miss` is not a process-level cold start.

| Mode / phase | Mean wall ms | P50 ms | P95 ms | Errors | Interpretation |
|---|---:|---:|---:|---:|---|
| Vector / cache miss | 27.39 | 21.12 | 45.44 | 0/4 | Fast local retrieval baseline |
| Graph / cache miss | 3,340.57 | 2,011.51 | 7,468.01 | 0/4 | Neo4j path search dominates |
| Hybrid / cache miss | 4,004.22 | 3,808.48 | 5,985.00 | 0/4 | PPR/path/vector orchestration dominates |
| Hybrid+Temporal / cache miss | 4,059.21 | 4,027.24 | 5,057.73 | 0/4 | Temporal scoring remains variable |

The warm phases in this smoke run were API-cache fills rather than confirmed
API cache hits. They are retained in the raw report but are not presented as a
guaranteed latency improvement. The new PPR cache is covered by unit tests; the
Silver smoke questions did not exercise a nonzero `ppr_ms` path in every mode.

Source report: local `reports/runtime_performance_2026-09-20_p1_smoke.json`.

## Quality and readiness boundaries

- Structural provenance checks are necessary but do not measure semantic
  extraction precision, recall, or F1.
- The 30-row Golden QA is single-reviewer and has repeated question text; it is
  useful for engineering regression but not an independently adjudicated gold
  benchmark.
- The 60-row table-quality candidate queue remains P2 work in progress.
- Fresh external extraction has recorded different accepted-output counts under
  the same nominal settings. Frozen response replay is deterministic, but the
  fresh-model gate remains open.
- No public result in this document claims production readiness, counterfactual
  causal identification, investment performance, or universal model
  superiority.
