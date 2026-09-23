# Reconstruction and acceptance ledger — 2026-09-21

This is the current evidence record for the three active NVIDIA 10-K filings.
It is deliberately narrower than a research paper result or a production
readiness declaration. A status below applies only to the stated scope; it
does not inherit a stronger meaning from another check.

## Status summary

| Scope | Status | What is actually evidenced | Remaining condition |
|---|---|---|---|
| A — trusted paper experiment | `BLOCKED` | Frozen three-filing corpus identity, claim-level contracts, current Silver/engineering Gold artifacts, and independent code regression | Independent second reviewer plus adjudication, fresh extraction repeatability, claim-matched benchmark, and complete metric run are still absent |
| B — engineering stable | `BLOCKED` | Canonical document-layer read, response/grounding contracts, QueryPlan unit contract, table conservation dry-run, and local regression pass | Clean install, live dependency/model fault injection, isolated graph/vector/numeric round trip, recovery, load, and deployment acceptance are not complete |
| C — production candidate | `NOT_RUN` | No production acceptance result is asserted | Security/permissions, monitoring, backup/rollback, SLO/RPO/RTO, cost, multi-instance, and production failure/load tests have not run |

The permitted status vocabulary is `PASS`, `FAIL`, `BLOCKED`, `NOT_RUN`, and
`NOT_APPLICABLE`. `BLOCKED` means that a required evidence condition is
currently unavailable or incomplete. It is not a claim that the code is broken.

## Reproducibility identity

- Build ID: `build_53eca4570a230b5c`
- Git commit observed locally: `50f696a75b03f89fda8be0090b6f938282256f22`
- Corpus: `2023-10-K.pdf`, `2024-10-K.pdf`, `2025-10-K.pdf`
- Parser: `pdfplumber-document-layer/v1`
- Corpus page total: `395`
- Frontend production build: `PASS` (`tsc -b && vite build`, 474 modules transformed)
- No active Neo4j or Chroma store was relabeled or overwritten.
- No Git push, paid model call, OCR call, or active-store write was performed
  for this record.

## M1 — response, provenance, numeric, and evaluator contracts

| Check | Status | Evidence |
|---|---|---|
| Missing or conflicting `build_id` is fail-closed | `PASS` | `require_same_build_id` rejects missing/non-mapping IDs and conflicting IDs; reconstruction tests cover both paths |
| Unbound evidence cannot be relabeled by a request-side build label | `PASS` | `EvidenceBundle` derives identity from stored evidence metadata and exposes `UNBOUND`; strict binding rejects it |
| NaN/Infinity numeric values are rejected | `PASS` | Financial numeric parser rejects string and floating-point non-finite values |
| Refusal and model-error states are orthogonal to answer status | `PASS` | Refusal-with-year/metric and generation-error regression cases are covered |
| Exact-hop evidence, number, and negation grounding | `PASS` | Regression cases reject borrowed-hop text, forged numeric values, and polarity mismatch |
| Hit Rate@K versus Recall@K | `PASS` | Retrieval evaluator now emits both distinct metrics with separate confidence intervals and registry specifications |
| Full local regression | `PASS` | `.venv\\Scripts\\python.exe -m pytest -q` -> `158 passed, 2 warnings, 7 subtests passed` |

This is a code-contract result. It does not prove semantic extraction accuracy
on a fresh corpus or live-model behavior.

## M2 — canonical document layer and parser pilot

| Check | Status | Evidence |
|---|---|---|
| Canonical page conservation | `PASS` | Read-only audit retained 169/169, 96/96, and 130/130 pages; 0 failed and 0 `OCR_REQUIRED` pages |
| Explicit text/table/OCR/layout statuses | `PASS` | Page records now retain separate status fields and table errors instead of silently dropping failures |
| Stable document/page/table/cell identity | `PASS` | IDs include the document SHA identity; the regression suite checks cross-document non-collision |
| Deterministic difficult-page candidate selection | `PASS` | Parser pilot selected 12 table/layout-risk pages from 108 table-bearing pages |
| Docling equivalence pilot | `NOT_RUN` | `scripts/run_parser_pilot.py` reports `docling.available=false`: package is not installed; this audit did not install it |
| Isolated parser/store publication | `BLOCKED` | No isolated Neo4j/Chroma/numeric materialization or atomic cutover was executed |

The parser pilot command is:

```powershell
.venv\\Scripts\\python.exe scripts/run_parser_pilot.py `
  --build-id build_53eca4570a230b5c `
  --output reports/parser_pilot_2026-09-21.json
```

The pilot output is intentionally `NOT_RUN` while its pdfplumber baseline is
`PASS`; page coverage is not a claim of table or footnote equivalence.

## M3 — QueryPlan execution

| Check | Status | Evidence |
|---|---|---|
| Parse filing scope and fact period separately | `PASS` | Regression covers “2025 annual report for FY2024”; document scope and fact period are distinct |
| Pass document scope into vector retrieval | `PASS` | Fake-storage contract test observes the parsed source filing filter |
| Pass temporal constraints into graph retrieval | `PASS` | QueryPlan carries exact fact years and relation filters in the unit contract |
| Live disclosure/as-of and restatement semantics | `BLOCKED` | No fresh live graph/numeric integration run established filing timestamps, restated-fact choice, or cross-filing precedence |

## M4 — source/version/license review and baselines

Official design references were reviewed, but they are not project benchmark
results: [Docling](https://github.com/docling-project/docling),
[Microsoft GraphRAG query overview](https://microsoft.github.io/graphrag/query/overview/),
[Sentence Transformers retrieve/rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html),
[Anthropic contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval),
and [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

| Check | Status | Evidence |
|---|---|---|
| Source URLs and intended roles recorded | `PASS` | Links above are primary/official references for the parser, retrieval, graph query, and filing API designs |
| Exact external dependency commit/tag/license freeze | `BLOCKED` | Docling is not installed and no external parser commit is pinned in this checkout; model licenses remain separate from code licenses |
| B0–B5 baseline and ablation matrix executed | `NOT_RUN` | No fresh isolated baseline run was performed against the active corpus |
| GraphRAG > Vector claim | `NOT_APPLICABLE` | No such claim is made until a claim-matched, independently reviewed benchmark exists |

## M5 — independent evaluation

| Check | Status | Evidence |
|---|---|---|
| Formal metric implementation | `PASS` | Metric registry, Hit Rate/Recall distinction, question-level bootstrap intervals, and Wilson execution intervals are wired into evaluators |
| Current engineering Gold | `PASS` | Existing 30-row human-reviewed artifact remains explicitly single-reviewer engineering Gold |
| Independent second labeler and adjudication | `BLOCKED` | No second blind label set or adjudication log is present |
| Fresh claim-matched benchmark | `BLOCKED` | The current Silver/Gold artifacts do not establish the requested independent paper result |
| Full fresh evaluation run | `NOT_RUN` | Model/dependency and isolated-store prerequisites are not all available |

## M6 — clean install, faults, staging, and deployment

| Check | Status | Evidence |
|---|---|---|
| No-LLM 2025 dry-run | `PASS` | 38 rule triples extracted, 0 ingested, 51 table candidates, 51 pending, 0 accepted/rejected, conservation true, 51 queue rows; LLM calls/network calls were 0 |
| Clean-environment installation | `BLOCKED` | Only the existing `.venv` was exercised; no clean rebuild was accepted |
| Fault injection and recovery | `NOT_RUN` | Live model, Neo4j, Chroma, cache corruption, rollback, and restart drills were not run |
| Load/SLO/deployment profile | `NOT_RUN` | No target hardware, concurrency, SLO/RPO/RTO, cost, or multi-instance acceptance run exists |

## Exact evidence commands

```powershell
.venv\\Scripts\\python.exe -m pytest -q
.venv\\Scripts\\python.exe scripts/create_reconstruction_manifest.py `
  --output reports/reconstruction_manifest_2026-09-21.json
.venv\\Scripts\\python.exe scripts/build_document_layer.py `
  data/pdfs_other/2023-10-K.pdf data/pdfs_other/2024-10-K.pdf `
  data/pdfs/2025-10-K.pdf `
  --build-id build_53eca4570a230b5c `
  --output reports/document_layer_audit_2026-09-21.json
.venv\\Scripts\\python.exe scripts/run_parser_pilot.py `
  --build-id build_53eca4570a230b5c `
  --output reports/parser_pilot_2026-09-21.json
.venv\\Scripts\\python.exe -m strategic_graphrag.pipeline.pipeline `
  --pdf data/pdfs/2025-10-K.pdf --no_llm --dry_run `
  --build_id build_53eca4570a230b5c `
  --pending_table_queue reports/reconstruction_dry_run_table_queue_2026-09-21.jsonl `
  --output_stats reports/reconstruction_dry_run_stats_2026-09-21.json
```

The JSON outputs above are local ignored artifacts in this checkout. This
tracked ledger records their relevant results so a future GitHub revision does
not depend on an untracked report file.
