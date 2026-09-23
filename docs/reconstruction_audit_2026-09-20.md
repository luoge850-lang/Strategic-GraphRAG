# System reconstruction audit — 2026-09-20

This record covers the reconstruction work added after the reliability follow-up. It is an evidence ledger, not a claim that the project is paper-ready or production-ready. Every gate uses only `PASS`, `FAIL`, `BLOCKED`, `NOT_RUN`, or `NOT_APPLICABLE`.

## Scope and preserved baseline

- Repository: `Nvidia-GraphRAG-Engine`, branch `codex/v3-three-filing-evidence-graphrag`.
- Active source PDFs were read from the existing workspace only. No PDF, annotation, active Neo4j database, Chroma collection, or external model response was replaced.
- Pre-existing historical-report deletions remain local and uncommitted; they were not used as current acceptance evidence.
- No live-store write, OCR call, paid model call, or Git push was performed in this reconstruction turn.

## Reconstruction decisions

| Area | Decision | Why the previous boundary was insufficient | Compatibility / rollback | Acceptance evidence | Status |
|---|---|---|---|---|---|
| PDF representation | `RETAIN_AND_ROUTE_THROUGH_CANONICAL_DOCUMENT_LAYER` | Pipeline and vector audit paths could independently extract page text; page/table identity was not one shared object | Existing pipeline return fields remain; reader is read-only and can be disabled by reverting the caller | 395/395 physical pages retained across three PDFs; page conservation true | `PASS` |
| OCR | `BLOCK_SILENT_INGEST` | Non-empty extracted text is not proof that a scanned page was parsed correctly | Pages with no text are `OCR_REQUIRED`; no downstream ingestion from those pages | Three active PDFs had 0 OCR-required and 0 failed pages in the current parser run | `PASS` |
| Query interpretation | `RETAIN_ENGINE_WITH_EXPLICIT_QUERY_PLAN` | Unknown questions previously defaulted to causal impact analysis | Historical `StructuredQuery` name remains an alias; `to_dict()` is versioned | Unknown-query and disclosure-period tests pass | `PASS` |
| Evidence transport | `ADD_COMMON_EVIDENCE_BUNDLE` | Graph/vector results exposed different provenance shapes | Existing `paths` and vector hits remain; bundle is additive | Bundle retains page, filing, evidence ID, method, and build ID; mixed IDs fail closed | `PASS` |
| Build identity | `BIND_NEW_ARTIFACTS_TO_BUILD_ID` | Corpus/model/prompt/index/cache mixing could be invisible | Existing active artifacts are not relabeled; new builds can opt into `--build-id` and old records remain readable | Secret-free reconstruction manifest generated; cache key includes build ID | `PASS` |
| Active graph/vector migration | `STAGING_REQUIRED` | In-place replacement cannot prove atomic publication or rollback | No active-store mutation; isolated build/export is required before cutover | No isolated Neo4j/Chroma round trip executed | `BLOCKED` |

The deliberate choice is incremental for the running system: the canonical
document layer and contracts are now executable, while active-store migration
is not claimed until an isolated staging build can be validated and rolled back.

## Current artifacts and fingerprints

The local reconstruction manifest is `reports/reconstruction_manifest_2026-09-20.json`.
It records the build identity, source fingerprint, dependency-lock hash, PDF
SHA256 values, parser configuration hash, model names, embedding configuration,
metric registry, and artifact status. The page-level review artifact is
`reports/document_layer_audit_2026-09-20.json`.

Current active PDF page ledger:

| Filing | Total | Parsed | OCR required | Failed | Conservation |
|---|---:|---:|---:|---:|---|
| 2023 10-K | 169 | 169 | 0 | 0 | `PASS` |
| 2024 10-K | 96 | 96 | 0 | 0 | `PASS` |
| 2025 10-K | 130 | 130 | 0 | 0 | `PASS` |
| **Total** | **395** | **395** | **0** | **0** | `PASS` |

The page parser records raw text, normalized text, physical page number,
printed-page candidate, dimensions, reading-order text blocks, table/cell
identities, parser version, configuration hash, and page status. Coordinates
are retained when exposed by the PDF backend; table coordinates are explicitly
unknown when the backend does not expose them.

## Verification

- Full Python regression after reconstruction: `152 passed`, 2 warnings, and 7 subtests passed.
- The targeted reconstruction/cache/pipeline slice: `44 passed`.
- Full 2025 no-LLM dry-run: `38` triples extracted, `0` ingested, `51` table candidates, `0` accepted, `51` pending, `0` rejected, conservation `true`, and `51` queue rows.
- The dry-run returned `build_id=build_fffe2c962a003570` and reported `130/130` pages parsed.
- The new manifest and document-layer audit are local derived artifacts; they do not prove graph/vector/numeric stores share that build ID because those isolated rebuilds were not run.

## Unified metric contract

`strategic_graphrag/evaluation/metric_spec.py` defines the metric unit,
numerator, denominator, dataset version, exclusions, aggregation, and interval
policy for PDF coverage, table binding, extraction precision, retrieval,
answers, abstention, execution success, and latency. Wilson intervals return
an interval below 1.0 for a one-success/one-trial sample; no all-success small
sample is presented as certain.

## Gate decision

| Target | Status | Evidence boundary |
|---|---|---|
| A — trusted paper experiment | `BLOCKED` | Independent blind human labels, adjudication, fresh external-model repeatability, and a claim-matched public benchmark remain incomplete. |
| B — engineering stable | `BLOCKED` | Canonical local extraction and contracts pass; clean install, live dependency fault injection, isolated staging, recovery, and load acceptance remain incomplete. |
| C — production candidate | `NOT_RUN` | Deployment profile, SLO/RPO/RTO, security review, monitoring, backup/rollback, cost, and multi-instance tests are not accepted. |

The correct next boundary is an isolated staging build with the recorded
`build_id`, followed by completeness checks, read-only query checks, simulated
interruption, rollback, and only then a controlled publication decision.
