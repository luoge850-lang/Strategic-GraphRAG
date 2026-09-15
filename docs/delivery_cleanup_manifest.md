# Delivery cleanup manifest — 2026-08-28

This manifest records the delivery cleanup performed on 2026-08-28. The
canonical implementation remains `strategic_graphrag/`; the built frontend is
served from `frontend/dist/` by the FastAPI entry point.

## Moved to the local recovery archive

The following items were removed from the delivery root and moved to
`archive/cleanup-2026-08-28/`. The archive is ignored by Git and is not part of
the source delivery package.

- `src/` and its eight legacy step/dashboard files: no current code or
  documentation references remained, and `strategic_graphrag/` is the active
  implementation.
- Eight unreferenced machine-specific helper scripts: the two Ollama
  installers, two local-model setup scripts, three ad-hoc model tests, and
  `clear_kg.py`.
- `evaluation/annotation/extraction_sample_2025_post_repair_v1.jsonl`: an
  unlabeled 30-row draft superseded by the labeled v2 sample.
- `docs/demo-dashboard.png`: an unused older screenshot superseded by
  `docs/demo-dashboard-v3.png`.
- Four byte-identical report duplicates; `final_kg_audit.json` and
  `final_page_coverage.json` were restored because
  `scripts/create_stable_provenance.py` still consumes those exact paths.

## Permanently removed generated artifacts

Only regenerable artifacts were deleted from the working directory:

- Python bytecode caches and `.pytest_cache/`.
- Historical runtime `.log` files under `reports/`.
- Frontend `pnpm` lock/workspace leftovers and `tsconfig.tsbuildinfo`.

The `reports/` directory was reduced from 113 files to a 19-file delivery
whitelist. The other 94 non-duplicate historical JSON/Markdown reports were
moved to `archive/cleanup-2026-08-28/historical-reports/`.

## Deliberately retained

- The three frozen PDFs, Chroma data, Neo4j snapshots, and all current audit
  reports.
- `reports/final_kg_audit.json` and `reports/final_page_coverage.json`, which
  are required inputs for the stable-provenance manifest script.
- The original 60-row annotation set and the labeled 30-row v2 set.
- `frontend/dist/`, `frontend/package-lock.json`, `.env`, and installed local
  runtimes needed to run the current demo.
- Non-duplicate JSON/Markdown reports that provide research provenance.

The retained report whitelist includes the corpus manifest, v3.1 freeze
artifacts, post-rebuild Neo4j snapshot and strict-chain audit, both repair-run
stats, the current extraction audits, the readiness audit, the retrieval smoke
report, and the stable-provenance inputs consumed by the existing scripts.

To recover a moved tracked file before committing the cleanup, use the archive
copy or restore it from Git; no remote branch or external message was created.

## Historical report cleanup — 2026-09-14

Sixteen superseded or failed-attempt reports were moved to
`archive/cleanup-2026-09-14/historical-reports/`: the old readiness audits,
failed-rebuild snapshots, pre-rebuild 2025 snapshot, obsolete repair-run
statistics, one-question retrieval probe, old chain/coverage audits, and the
historical extraction-quality audit. Current snapshots, per-filing audits,
stable-provenance inputs, the Silver benchmark, and the current readiness report
remain in `reports/`. This reduced the active report inventory to 34 files
without deleting evidence needed to explain earlier runs.

## Follow-up cleanup guardrails — 2026-09-02

- `.venv/` is the only canonical Python environment for this delivery and is
  expected to use Python 3.12.
- The root `venv/` was confirmed as a legacy Python 3.14 environment and moved
  to `archive/legacy-venv-2026-09-02/`; the canonical `.venv/` remains in place.
  It was archived rather than deleted so it can be recovered if an old local
  script unexpectedly depends on it.
- Git-history deletions and untracked files must not be restored or deleted by
  a one-click bulk action. Classify each item individually, record whether it
  is current, historical, generated, or user-owned, and then choose the
  recoverable action for that item.

## Temporary working artifacts — 2026-09-11

The root `tmp/` directory contained 26 untracked document-render folders,
proposal scripts, and intermediate architecture images. No current source or
runtime path referenced these files. They were moved, not destroyed, to
`archive/cleanup-2026-09-11/tmp/` so the delivery root stays focused while the
render history remains recoverable.

## Golden QA artifacts retained — 2026-09-02

- `data/evaluation/golden_qa_v2.jsonl` remains the generated candidate set. It
  is useful for reproducibility and regression comparison, but its labels are
  not treated as human truth.
- `evaluation/golden_qa_human_v2.jsonl` is the current 30-row, stratified
  claim-ID-v2 human-review work file. It starts blank and becomes a usable
  Golden QA set only row by row
  after a reviewer records the answerability decision, reference answer,
  evidence IDs, pages, and reviewer identity. The older v1 file is retained as
  historical evidence because its candidate IDs no longer resolve in the
  current graph.
- The original extraction samples, frozen filing artifacts, graph snapshots,
  and audit reports remain preserved because they are required to reproduce
  and explain earlier measurements. They are not disposable cache files.

## Follow-up cleanup — 2026-09-14

- Confirmed that the project-root `tmp/` directory was empty, including hidden
  entries, and removed only that empty directory.
- No files and no non-empty directories were removed.
- Retained the user-owned `GraphRAG初稿.docx`, `evaluation/cache/`,
  `evaluation/review_packets/`, `evaluation/silver_retrieval_v1.jsonl`, and
  `evaluation/annotation/`.
- Retained `scripts/`, `tests/`, `docs/`, `archive/`, `reports/`,
  `data/` and its PDFs, and Chroma artifacts.
- Did not touch `.env`, Neo4j, or Chroma contents, and performed no remote
  GitHub operation.
