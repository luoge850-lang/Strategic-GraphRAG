# Delivery cleanup manifest

This manifest records the working-tree decisions made for the development
publication. It prevents an intentional archive from being confused with an
accidental data loss and prevents a local draft from entering the public repo.

## Keep and publish on the development branch

- Current runtime and evaluation changes under `strategic_graphrag/`,
  `scripts/`, and `tests/`.
- Current acceptance, reconstruction, claim-ledger, results, version, and
  diagram documents under `docs/`.
- `README.md`, `API-ANALYSIS.md`, `data/README.md`, and the current schema
  design.
- Mermaid sources and deterministic SVG outputs under `docs/diagrams/`.

## Preserve locally, do not publish

- `GraphRAG初稿.docx`: user-owned personal draft; it is explicitly ignored.
- `.env`, raw PDFs, extracted text, Chroma data, Neo4j data, response caches,
  and local reports: generated or sensitive runtime assets.
- Local archive/recovery material: useful for rollback, but not a public source
  artifact.

## Intentional historical report cleanup

The deleted tracked reports are superseded historical snapshots, including
old corpus manifests, older Silver/Golden results, and pre-freeze audits. They
are not restored into the public tree because doing so would encourage readers
to mix incompatible inventories and evaluation protocols. The current public
boundary is the checked-in protocol plus the compact public results summary;
the raw files remain recoverable from the local history/archive when needed.

## Reproducibility consequence

The development branch remains dirty until this publication change is
committed. No old CI success is inherited by the working tree. After review,
the exact pushed commit and its CI run must be added to the release record;
`stable` and existing tags remain unchanged.
