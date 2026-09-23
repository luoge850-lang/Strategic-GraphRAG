# Strategic-GraphRAG

[![CI](https://github.com/luoge850-lang/Strategic-GraphRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/luoge850-lang/Strategic-GraphRAG/actions/workflows/ci.yml)

Strategic-GraphRAG is a research-oriented retrieval system for tracing
financial disclosures in NVIDIA fiscal 2023–2025 SEC 10-K filings. It combines
a strict evidence graph, filing-scoped vector retrieval, temporal filters, and
an evidence-aware answer contract. It is an engineering/research prototype,
not an investment adviser, causal-identification system, or production service.

## Status at a glance

The GitHub default branch is `stable`. This checkout is the development branch
`codex/v3-three-filing-evidence-graphrag`; its metrics must not be mixed with
the stable release. The current development acceptance boundary is:

| Gate | Status | Meaning |
|---|---|---|
| Trusted paper experiment | `BLOCKED` | Independent second review, adjudication, fresh-model repeatability, and a claim-matched benchmark remain incomplete. |
| Engineering stable | `BLOCKED` | Local contracts pass, but clean-install, isolated-store, recovery, and load acceptance remain incomplete. |
| Production candidate | `NOT_RUN` | Security, monitoring, backup/rollback, SLO, cost, and production failure/load tests are not accepted. |

See the [acceptance ledger](docs/acceptance_ledger_2026-09-21.md), the
[public claim ledger](docs/claim_ledger.md), and the
[version strategy](docs/version_strategy_2026-09-22.md). The working-tree
decisions are recorded in the [delivery cleanup manifest](docs/delivery_cleanup_manifest.md).

## Verified development snapshot

| Filing | Physical pages | Strict EvidenceClaims | Vector chunks |
|---|---:|---:|---:|
| 2023 10-K | 169 | 126 | 678 |
| 2024 10-K | 96 | 129 | 425 |
| 2025 10-K | 130 | 126 | 583 |
| **Total** | **395** | **381** | **1,686** |

These numbers describe the current development snapshot only. The public
`stable` branch contains a different historical snapshot; do not combine its
383-claim inventory with this table. Raw PDFs, Chroma files, Neo4j data, and
external-model response caches are local/generated assets and are not shipped
in this repository. Their acquisition and hash boundary are described in
[`data/README.md`](data/README.md) and the
[reproducibility freeze](docs/reproducibility_freeze_2026-09-18.md).

## What is implemented

- Canonical page, text-block, table, and cell records with page-conservation
  and fail-closed parse statuses.
- Validated `EvidenceClaim` records with quote, page, filing, entity, relation,
  and claim-ID provenance.
- Neo4j graph retrieval, Chroma vector retrieval, Hybrid fusion, and
  Hybrid+Temporal retrieval.
- Filing scope, fact-period parsing, directed path search, bounded PPR anchor
  expansion, temporal fact filters, and structured response contracts.
- Evidence variants are merged by semantic path before ranking. Explicit
  structural relations require endpoint alignment and direct predicate support;
  `includes`, `runs on`, and `based on` remain background evidence.
- A bounded process-local PPR cache and separate `anchor_resolution_ms` /
  `ppr_ms` telemetry for repeated Graph/Hybrid requests.
- A local React/Vite Demo and a click-to-run Windows launcher. The launcher
  waits for `/health/ready` and does **not** configure boot-time auto-start.

The editable architecture and publication diagrams are:

- [`docs/diagrams/architecture.mmd`](docs/diagrams/architecture.mmd)
- [`docs/diagrams/publication_pipeline.mmd`](docs/diagrams/publication_pipeline.mmd)
- Generated SVGs in the same directory after running the documented diagram
  command.

The live runtime verification record is [`docs/visual_verification_2026-09-23.md`](docs/visual_verification_2026-09-23.md).
It records the actual graph state, an evidence-trace query that failed closed,
and a reasonable abstention. The stale `0 NODES` image is intentionally not
used as the project screenshot. Because the graph and external services are
not shipped in Git, a static bitmap is not treated as reproducible evidence;
release screenshots must carry a commit, build ID, data scope, and live/replay
label.

## Evidence-traceable example

Question: `Compare revenue in 2023, 2024, and 2025`.

The current development runtime is expected to return three
`REPORTS_METRIC` evidence paths, one per filing, with PDF page references. The
answer contract treats accounting disclosure as disclosure; it does not infer
why revenue changed or claim a counterfactual causal effect. Use the live Demo
to inspect the returned claim IDs and open the cited source page.

## Retrieval results

The current Silver regression contains 37 automatically generated questions,
with 32 answerable and 5 abstention-required. The page-level development
snapshot is summarized in [public results](docs/public_results_2026-09-22.md):

| Mode | Precision@5 | Recall@5 | nDCG@5 | MRR |
|---|---:|---:|---:|---:|
| Vector | 0.0500 | 0.2188 | 0.1103 | 0.0828 |
| Graph | 0.2313 | 0.8259 | 0.8079 | 0.8073 |
| Hybrid | 0.2313 | 0.8259 | 0.7733 | 0.7604 |
| Hybrid+Temporal | 0.2250 | 0.7946 | 0.7577 | 0.7500 |

These are retrieval metrics on graph-derived Silver labels, not answer
accuracy and not independent human gold. They support only the narrow
observation that Graph scored higher than Vector on this snapshot. They do not
prove that GraphRAG is generally superior to Vector RAG, and Hybrid+Temporal
is not the global winner in this run.

## Quick start

### Offline checks without external services

Use the canonical Python environment for tests and static checks:

```powershell
\.venv\Scripts\python.exe -m pytest -q
\.venv\Scripts\python.exe -m compileall -q strategic_graphrag scripts tests
git diff --check
```

The current local environment is expected to be Python 3.12 with versions in
[`requirements-lock-2026-09-19.txt`](requirements-lock-2026-09-19.txt). A clean
install is a separate acceptance condition and is not implied by a local test
pass.

### Real Neo4j/Chroma Demo

1. Obtain the three public 10-K PDFs from SEC EDGAR or the official company
   filing pages. Do not commit them to this repository.
2. Create `.env` from [`.env.example`](.env.example). Configure the current
   Neo4j Aura URI/database, DeepSeek credentials/model, and the active Chroma
   collection. Never commit `.env`.
3. Prepare the vector and graph stores using the explicit migration scripts;
   migrations write to external stores and are not part of ordinary checks.
4. Build the frontend:

   ```powershell
   cd frontend
   npm install
   npm run build
   cd ..
   ```

5. After entering Codex or opening a terminal, double-click `open_demo.cmd`,
   or run:

   ```powershell
   .\scripts\open_demo.ps1
   ```

   The script starts the local API only when needed, waits for Neo4j, Chroma,
   and the configured LLM to report ready, and then opens
   `http://127.0.0.1:8000/`. It does not install a Windows startup task.
6. For a controlled restart only:

   ```powershell
   .\scripts\open_demo.ps1 -Restart
   ```

If `/health/live` is available but `/health/ready` is not, the API process is
running but an external dependency is not ready. Check that the Aura database
is running and copy its current Connect URI/database name into `.env`; the
database instance ID alone is not a connection URI.

### Optional model generation

Retrieval-only evaluation disables synthesis and external LLM anchor expansion
for comparability. Answer synthesis is optional and must be reported in a
separate run with model, prompt, temperature, cache mode, and failure counts.

## Evaluation and reproduction

The evaluation protocol defines units and denominators for page-level,
sentence-level, EvidenceClaim-level, primary-evidence, answer, citation, and
abstention metrics:
[`docs/research_evaluation_protocol.md`](docs/research_evaluation_protocol.md).

```powershell
# Readiness audit; fail-closed and read-only
\.venv\Scripts\python.exe scripts/audit_research_readiness.py

# Retrieval-only four-mode evaluation
\.venv\Scripts\python.exe scripts/evaluate_retrieval_benchmark.py --help

# Runtime smoke profiling; cache miss is not process cold start
\.venv\Scripts\python.exe scripts/benchmark_runtime_performance.py `
  --base-url http://127.0.0.1:8000 --limit 4 --concurrency 1

# Table-quality Gold candidate annotation
\.venv\Scripts\python.exe scripts/export_table_annotation_queue.py --help
\.venv\Scripts\python.exe scripts/evaluate_table_quality.py --help
```

Do not run commands with `--apply` against the active graph during ordinary
review. Active-store writes require an isolated database, a new immutable
build identity, completeness checks, and a rollback plan.

## Evaluation boundaries

- `VERBATIM` means the stored quote matches the declared source location. It
  is not a semantic truth label.
- The 30-row answer-level Golden QA is complete as a one-reviewer engineering
  checkpoint, not as independent two-reviewer paper gold.
- The 60-row table-quality queue is separate and remains a candidate set until
  independent annotation and adjudication are complete.
- The current Silver expected evidence is derived from the graph under test;
  its scores are regression proxies with self-test bias.
- Fresh external-model extraction has produced different accepted-output
  counts under nominally fixed settings. Cached record/replay is deterministic
  for the recorded responses, but the readiness gate remains `NOT_READY`.
- Disclosed relationships are attributed to the filing. Graph paths do not
  prove counterfactual causality, effect sizes, investment outcomes, or future
  performance.
- Docling equivalence, clean-install deployment, isolated store publication,
  live fault recovery, production load, cost, monitoring, and security remain
  unaccepted.

## Repository map

| Path | Role |
|---|---|
| `strategic_graphrag/` | Runtime, contracts, extraction, retrieval, schema |
| `frontend/` | React/Vite Demo |
| `evaluation/` | Versioned Silver, Golden QA, and table candidate inputs |
| `scripts/` | Build, audit, benchmark, launcher, and preparation commands |
| `docs/` | Protocols, ledgers, current status, diagrams, and historical records |
| `tests/` | Offline regression and contract tests |
| `archive/` | Local recovery material; ignored and not a public source artifact |

## License and data

No code license has been selected in this repository. Do not infer an open
source license from GitHub visibility. The SEC filings, model weights, APIs,
and third-party datasets have their own terms. See [data/README.md](data/README.md)
and [data/external/README.md](data/external/README.md) before redistribution.
