# Repository Freeze Manifest — 2026-09-14

This is a local, reviewable freeze manifest. It does not stage, commit, tag,
push, or otherwise change the remote repository.

## Freeze boundary

The local Git freeze is intended to capture the reproducible implementation,
tests, documentation, and evaluation definitions/products that are safe to
version as project assets:

- `strategic_graphrag/`
- `scripts/`
- `tests/`
- `docs/`
- repository-level documentation such as `README.md` and `STRATEGY.md`
- `evaluation/annotation/`
- `evaluation/cache/`
- `evaluation/review_packets/`
- `evaluation/silver_retrieval_v1.jsonl`

The source/evaluation freeze is separate from the local runtime snapshot. The
Neo4j snapshot, PDFs, Chroma index, and generated reports remain on this
machine for reproducibility but are not staged into the Git commit because
they are large, generated, environment-specific, or already covered by the
existing archive/report manifests.

## Explicitly excluded but retained locally

- `GraphRAG初稿.docx` (user-owned draft; deliberately remains untracked)
- `.env` and all credentials/secrets
- `.venv/`, `venv/`, `frontend/node_modules/`, and other local environments
- `data/`, `reports/`, `archive/`, and Chroma runtime artifacts

Exclusion from the Git freeze does not authorize deletion or relocation.

## Unchanged external and runtime state

- `.env` was not touched.
- Neo4j was not touched.
- Chroma contents were not touched.
- No remote GitHub operation was performed; the remote repository remains
  unchanged.
- The only filesystem cleanup in this follow-up was removal of the already
  verified empty project-root `tmp/` directory.
