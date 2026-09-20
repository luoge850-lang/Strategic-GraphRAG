# Reliability Audit and Evidence Ledger — 2026-09-20

## Scope and decision rule

This is an incremental reliability patch and evidence ledger for the NVIDIA GraphRAG workspace. It is not a claim that the research or production acceptance program is complete. Status labels are deliberately scoped:

- `VERIFIED_COMPLETE`: the stated acceptance criterion is covered by code inspection and automated regression evidence in this checkout.
- `IMPLEMENTED_NOT_VALIDATED`: the implementation exists, but a fresh live, corpus-wide, or independent validation run is still missing.
- `BLOCKED`: the required evidence cannot be completed with the current assets, reviewers, or execution conditions.
- `NOT_STARTED`: no fresh implementation or validation was completed for the item.
- `SUPERSEDED`: retained only for historical traceability.

## Environment and asset identity

- Date: 2026-09-20, Asia/Shanghai.
- Branch: `codex/v3-three-filing-evidence-graphrag`.
- HEAD at audit start: `6c36fad29f07537cdaf3f3e6342e37e7207476ba`.
- The worktree was already dirty. Existing user changes and recoverable historical report deletions were preserved; no reset, checkout, clean, graph rebuild, or index replacement was performed.
- Project Python: 3.12.14. Runtime dependency probe reported ready for FastAPI 0.141.1, Uvicorn 0.52.3, Neo4j 6.2.0, ChromaDB 1.5.9, ONNX Runtime 1.28.0, PyMuPDF 1.28.2, pdfplumber 0.11.10, and Pydantic 2.13.4.
- Local raw-PDF inventory is incomplete: only `data/pdfs/2025-10-K.pdf` is present in this checkout. Read-only probes found three derived filings in the active Neo4j/Chroma assets: 2023-10-K, 2024-10-K, and 2025-10-K.
- Read-only derived-asset counts: 381 `EvidenceClaim`, 234 `FinancialObservation`, 381 `TemporalFact`, 195 `TemporalChange`; Chroma collection `nvidia_sec_filings_active` contains 1,686 chunks.
- Because the local raw corpus and the active derived assets are not independently rejoined by this audit, three-filing raw-corpus completeness is not asserted.

## Issue and acceptance ledger

| Area | Status | Evidence and remaining boundary |
|---|---|---|
| P0 grounding fail-closed behavior | `VERIFIED_COMPLETE` | Citation identity now preserves and audits submitted page/year/filing values, validates claim IDs, and rejects unsupported relation evidence. Covered by new grounding tests and the full regression suite. A complete live corpus audit is still separate. |
| P0 negative-claim guard | `VERIFIED_COMPLETE` | Retained substantive claims are grounded and validated; pure refusals remain non-grounding abstentions. Guard actions are recorded in metadata. Covered by regression tests. |
| P0 structured outcome contract | `VERIFIED_COMPLETE` | Shared outcomes are `ANSWERED`, `PARTIALLY_ANSWERED`, `ABSTAINED`, `DEPENDENCY_ERROR`, `MODEL_ERROR`, `TIMEOUT`, and `VALIDATION_ERROR`; legacy responses are parsed conservatively. Golden/retrieval evaluators now record outcome counts and execution-success rate. |
| P0 HTTP error contract | `IMPLEMENTED_NOT_VALIDATED` | Engine dependency failures now carry `DEPENDENCY_ERROR`; API maps dependency/model/timeout/validation failures to 503/502/504/422. Mocked API regression passes; a live failure-injection run was not performed. |
| Fair retrieval and answer evaluation | `IMPLEMENTED_NOT_VALIDATED` | Evaluators now separate abstention from execution failure and retain outcome labels. No fresh full benchmark was run in this turn, so answer-level and retrieval-level scores are not being claimed. |
| Evidence identity and filing normalization | `IMPLEMENTED_NOT_VALIDATED` | Page/year/filing identity and mismatch reasons are explicit. Unit coverage includes correct evidence ID with wrong page/file/year. Cross-filing normalization and page-unit reconciliation still need a corpus-wide run. |
| Relation semantics and source binding | `VERIFIED_COMPLETE` | Competitor-sentence false positives are rejected in extraction and evidence support scoring; enumeration/passive/modal cases have regression coverage. This is a code-contract result, not a precision/recall estimate. |
| Table percentage/change-column handling | `IMPLEMENTED_NOT_VALIDATED` | Percentage rows preserve both values and explicit change-column detection is supported; regression coverage passes. No fresh PDF-wide extraction quality report was generated. |
| Company/entity IDs in table extraction | `IMPLEMENTED_NOT_VALIDATED` | Verified `company_id` input is supported; missing identity is emitted as `PENDING_COMPANY_REVIEW` rather than guessed. No multi-company Gold set validates recall or disambiguation. |
| Canonical table schema | `IMPLEMENTED_NOT_VALIDATED` | Table triples now carry document/report-period, currency/scale/unit, sign, table/row/column identifiers, source span, and resolution fields. The active graph was not rebuilt, so derived-asset adoption is unvalidated. |
| Independent human Gold workflow | `BLOCKED` | Current evidence has 30 reviewed Golden QA rows from one reviewer, 23 requiring abstention, and 60 unlabeled table candidates. A second independent reviewer and adjudication are still required. |
| Fresh extraction repeatability | `BLOCKED` | Existing project state records fresh repeatability as blocked. This turn did not recover the missing raw PDFs or run a clean independent extraction. |
| External benchmark execution | `NOT_STARTED` | External benchmark protocols/inventory exist, but they are registered rather than executed against the active NVIDIA corpus. |
| Performance/deployment acceptance | `IMPLEMENTED_NOT_VALIDATED` | Existing project artifacts cover some latency/deployment work; this turn did not run a fresh load test, clean deployment rehearsal, or target-based acceptance measurement. |
| Research/paper acceptance | `BLOCKED` | Reliable claims remain limited by independent Gold, fresh repeatability, raw-corpus reconciliation, and corpus-specific benchmark gaps. |
| Production candidate | `NOT_STARTED` | No production-readiness claim is made. |

## Changes in this audit patch

- Added `strategic_graphrag/response_contract.py` as the shared outcome parser and contract.
- Hardened engine grounding, citation mismatch reporting, relation-support checks, negative-claim metadata, and dependency outcomes in `strategic_graphrag/engine/graph_rag_engine.py`.
- Added structured HTTP outcomes and status mapping in `strategic_graphrag/api/server.py`.
- Updated Golden, retrieval, standard-query, and end-to-end audit scripts to record structured outcomes and avoid scoring execution failures as abstentions.
- Hardened relation source binding in `strategic_graphrag/engine/evidence_quality.py` and `strategic_graphrag/pipeline/extractor.py`.
- Added conservative table percentage handling, explicit company-review state, and canonical table fields in the table pipeline.
- Added regression coverage in `tests/test_evidence_and_outcome_contract.py`.

## Verification performed

```text
138 passed, 2 warnings, 7 subtests passed
AST_OK 12 files
git diff --check: no whitespace errors
```

The two warnings are the pre-existing FastAPI `on_event` deprecation warnings. No live LLM request, graph write, vector-index rebuild, PDF download, benchmark run, or Git upload was performed by this audit patch.

## Safe handoff

The defensible current claim is: the main P0 response/grounding contracts and several extraction false-positive guards are implemented and regression-tested in this checkout. The defensible non-claim is: the project is not yet independently reproducible, benchmark-complete, or production-ready.
