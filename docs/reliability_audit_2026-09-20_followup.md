# Reliability Audit Follow-up — 2026-09-20

This is a fresh follow-up ledger for the current dirty checkout. It records what was re-verified in this turn and what remains unproven. It does not upgrade the project to paper-ready or production-ready status.

## Fresh identity and asset probe

- Branch: `codex/v3-three-filing-evidence-graphrag`.
- HEAD at the start of this follow-up: `6c36fad29f07537cdaf3f3e6342e37e7207476ba`.
- The worktree was already dirty. Existing user changes and deleted historical reports were preserved. No reset, checkout, clean, graph rebuild, vector-index replacement, PDF download, model call, commit, or push was performed.
- Runtime: Python 3.12.14; FastAPI 0.141.1; Uvicorn 0.52.3; Neo4j driver 6.2.0; ChromaDB 1.5.9; ONNX Runtime 1.28.0; PyMuPDF 1.28.2; pdfplumber 0.11.10; Pydantic 2.13.4.
- Active raw PDF SHA256 checks matched the manifest for all three filings:
  - `2023-10-K.pdf`: `89981bbfcd91e20498c1060d7efb39022ac8f85e639f2841091695885aa9f8a8`
  - `2024-10-K.pdf`: `536f66d7f1c3413abbf643e0a02bd0aab65639116fe630225f3f93529244658b`
  - `2025-10-K.pdf`: `b67bd67a64488a54886de001c788bc5059a965fc6ef5b4d8b71624951e13df8e`
- Current source-tree hash after this patch: `2dea88dd95d3daf7512ec91cbb5ff620df3dd0f6e3722806709c396ce666a69a`. The older manifest's source hash is therefore stale for the current uncommitted checkout; the corpus file hashes and derived counts were checked separately.
- Read-only Neo4j/Chroma probe: 3 documents, 381 `EvidenceClaim`, 234 `FinancialObservation`, 381 `TemporalFact`, 195 `TemporalChange`, and 1,686 Chroma chunks. No write query was issued.

## Gate status

| Gate | Status | Basis and remaining boundary |
|---|---|---|
| A — trusted paper experiment | `BLOCKED` | Three raw filings and derived assets are identifiable, but fresh independent extraction repeatability is still blocked; human Gold remains 30 rows from one reviewer, with 23 abstention rows, and the 60-row table set is still unlabeled. No paper-grade benchmark claim is made. |
| B — engineering-stable version | `BLOCKED` | P0 response, grounding, API, evaluator, relation-binding, and table-queue contracts pass local regression plus a real 2025 PDF dry-run. Clean-environment, live dependency/model fault injection, load, and deployment acceptance remain unrun. |
| C — production candidate | `NOT_RUN` | No fixed hardware/cost/SLO run, multi-instance deployment rehearsal, rollback test, or production security review was performed. |

## Issue ledger

| ID | Priority | Location | Trigger / root cause | Change and regression evidence | Status / remaining action |
|---|---|---|---|---|---|
| P0-GROUNDING | P0 | `graph_rag_engine.py` grounding validator | Generated `numeric_fields`, wrong units/signs/periods, unrelated cited IDs, and extra summary sentences could be accepted through weak path overlap | Added ID-indexed support checks, exact numeric/sign/unit/period/modal diagnostics, citation mismatch audit, and summary coverage tests; full suite passes | `PASS` for deterministic contract cases; run corpus-wide live audit before research claims |
| P0-STATUS | P0 | `response_contract.py`, `api/server.py`, evaluators | Legacy text or conflicting HTTP/body status could turn failures into answers or abstentions | Added orthogonal status derivation, HTTP precedence, provenance, and execution-aware denominators; full suite passes | `PASS` locally; live model/dependency fault injection remains `BLOCKED` |
| P1-TABLE | P1 | `pipeline.py`, `financial_table_extractor.py`, table queue/API | Missing company identity could be silently dropped or guessed; candidate IDs were not guaranteed to round-trip | Added document registry, SHA256 binding, stable IDs, pending queue, Gold scaffold, and processed-candidate conservation; 2025 dry-run verified 51 pending and conservation | `PASS` for local dry-run; isolated database write/read round-trip remains `BLOCKED` |
| P1-CI | P1 | `.github/workflows/ci.yml` | `unittest discover` did not guarantee collection of function-style pytest tests | CI now installs project runtime dependencies and runs `python -m pytest -q` plus compile and whitespace checks | `PASS` for workflow configuration; GitHub run after push is `NOT_RUN` |
| P1-GOLD | P1 | `evaluation/golden_qa_human_v2.jsonl` and benchmark scripts | One reviewer and repeated questions cannot establish independent paper-grade quality | Evaluators retain annotation-level and question-level distinctions and do not fabricate a second reviewer | `BLOCKED`; obtain independent blind labels and adjudication |

For each `BLOCKED` item, the missing condition is stated in the final column and
the next action is an isolated or independently reviewed run. No blocked item
is treated as a pass merely because the local unit tests pass.

## Changes verified in this follow-up

- Added versioned orthogonal response state: `execution_status`, `answer_status`, `grounding_status`, with derived compatibility `outcome`. Empty/unknown responses are contract errors; HTTP 429/401/403/408/504/503/502/5xx precedence is explicit; retrieval-only is `NOT_REQUESTED` rather than an abstention.
- Hardened grounding so claim IDs must bind to the submitted page/year/filing and every cited claim ID must support the asserted statement. Numeric values use exact token matching; explicit sign, unit, period, negation, and modal mismatches are surfaced as diagnostics. Mixed factual text plus refusal remains substantive; executive-summary fragments must be covered by structured claims.
- Hardened `PRODUCES` source binding in both relation scoring and extraction. A sentence about another company producing a product no longer becomes a source-company production fact merely because the queried company appears elsewhere in the sentence.
- Updated API and evaluation paths to preserve raw model output, response status provenance, execution errors, and denominator separation. Execution failures are not scored as abstentions or successful retrievals.
- Added stable table candidate identity, filename+page+table/row/period round-trip fields, document SHA256, explicit `PENDING_COMPANY_REVIEW`, optional verified document registry, deduplicated JSONL pending queue, Gold scaffolding, and candidate conservation accounting. Pending rows are excluded before graph ingestion.

## Verification evidence

- Full Python regression: `144 passed, 2 warnings, 7 subtests passed`.
- Targeted grounding/pipeline/table regression: `42 passed, 2 warnings`.
- AST parse of modified Python modules/tests: `AST_OK`.
- Frontend production build: `npm run build` passed; Vite transformed 474 modules and completed production chunk rendering.
- Real local dry-run, `2025-10-K.pdf`, `--no_llm --dry_run`, no Neo4j connection/write: 38 triples extracted, 0 ingested, 51 table candidates, 0 accepted, 51 pending, 0 filter-rejected, `conservation_holds=true`, and 51 unique pending queue rows written to a temporary JSONL file.

The remaining blockers are evidence blockers, not merely missing documentation: independent review, clean repeatability, full benchmark execution, and production acceptance measurements are still required before changing any gate to complete.
