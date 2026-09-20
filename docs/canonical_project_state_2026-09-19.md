# Canonical project state — 2026-09-19

This document is the working-tree handoff for the NVIDIA three-filing
evidence-grounded GraphRAG project. It records the canonical state after the
2026-09-19 audit and evaluation-preparation pass; it is not a claim that the
project is publication ready.

## Canonical implementation

- Code: `strategic_graphrag/`
- API and Demo: `strategic_graphrag/api/server.py`, served through
  `frontend/dist/`
- Frontend source: `frontend/src/`
- Evaluation scripts: `scripts/`
- Tests: `tests/`
- Runtime: `.venv\Scripts\python.exe`, Python 3.12.14
- Current branch: `codex/v3-three-filing-evidence-graphrag`
- Source commit before this state capture: `6c36fad`
- Working tree: dirty by design; user changes and recoverable report moves are
  preserved.

## Frozen corpus and derived assets

- PDFs: NVIDIA 2023, 2024 and 2025 10-K; SHA-256 and byte sizes are recorded in
  `reports/corpus_manifest_2026-09-19.json`.
- Active graph inventory: 126 / 129 / 126 strict claims by filing; 381 total.
- Derived temporal inventory: 381 TemporalFact, 195 temporal changes in the
  readiness audit.
- Vector collection: `nvidia_sec_filings_active`, 1,686 chunks,
  `all-MiniLM-L6-v2`.
- Claim ID version: `v2`.
- Extraction prompt: `v2-evidence-claim-1`.
- Provider/model currently configured: DeepSeek / `deepseek-v4-flash`.
- Dependency lock: `requirements-lock-2026-09-19.txt`.
- Source-tree hash after this pass: recorded in
  `reports/corpus_manifest_2026-09-19.json`.

## Change classification

### Retain as current user/project work

- Evidence-aware ranking, strict structural relation gating, question-level
  Golden evaluation, Neo4j reconnect handling, startup race handling, tests and
  the current documentation updates.
- `GraphRAG初稿.docx` is user-authored working material and is preserved.
- `evaluation/golden_qa_human_v2.jsonl` is current human-reviewed data and is
  preserved without rewriting labels.

### Intentional recoverable archive moves

Superseded reports and audit bundles were moved, not deleted, under
`archive/cleanup-2026-09-18/historical-reports/`. The complete list is in
`docs/report_archive_manifest_2026-09-18.md`.

### Do not restore blindly

The deleted report paths in Git are the tracked side of those intentional
archive moves. Restoring them would reintroduce duplicate and stale results;
deleting the archive would destroy audit history. No Neo4j graph rebuild is part
of this state capture.

### Additions completed in this pass

- Exact dependency lock, refreshed corpus manifest, and canonical state report.
- A 60-row table-quality annotation queue with empty gold fields. It is not
  counted as independent gold until a reviewer completes it.
- The table annotation workbench is available at the local `/table-qa` route;
  it keeps system predictions separate from reviewer Gold fields and opens the
  source PDF at the recorded page. The operational protocol is
  `docs/table_quality_annotation_guide_2026-09-20.md`.
- Independent registration and schema validation for FinanceBench, FinQA and
  TAT-QA. Their scores are not mixed into the NVIDIA benchmark.
- Runtime benchmark reports for all four modes, cache miss/hit phases, stage
  timings, tail latency and a bounded concurrency smoke run.
- Active report directory reduced to 40 files; superseded copies were moved to
  the recoverable archive manifest rather than deleted.

## Reproducibility status

- `.venv` is the validated development environment; `pytest` is installed
  there. System `py` is a different Python and is not the project runner.
- Cache record/replay is deterministic, but fresh external extraction remains
  non-repeatable (observed 126, 131 and 112 candidates under different runs).
- The current readiness report remains fail-closed `NOT_READY` for that one
  reason. This is a research limitation, not permission to silently replace
  the graph.

## Next controlled phases

1. Keep this state frozen and verify the lock/manifest hashes before any graph
   replacement.
2. Add independent extraction labels and table-cell evaluation.
3. Add separated external FinanceBench/FinQA/TAT-QA adapters without mixing
   their metrics with the NVIDIA benchmark.
4. Re-run four retrieval modes with a common protocol and latency breakdown.
5. Expand the NVIDIA QA set to two independent reviewers plus arbitration.
6. Only after those gates pass, evaluate a read-only Agent.
