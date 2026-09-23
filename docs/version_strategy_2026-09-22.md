# Version and release strategy

## Current refs

| Ref | Role | Observed commit / state | Public policy |
|---|---|---|---|
| `stable` | GitHub default branch | `297379f` at audit time | Preserve as the frozen public release identity. Do not mix its metrics with development metrics. |
| `codex/v3-three-filing-evidence-graphrag` | Active development branch | `6c99910` after publication; CI run `35821223049` passed | Development snapshot is published here. It is not the default branch and must not be presented as stable. |
| `v3.2.0-research-freeze-2026-09-14` | Historical tag | Existing immutable tag | Keep historical claims unchanged. |
| Local HEAD | Working checkout | `6c99910`; clean after publication | Local tests were run before commit; GitHub CI passed for the exact pushed commit. |

The repository API currently reports `stable` as the default branch. The local
remote-tracking `origin/HEAD` may be stale and is not used as the authority for
this statement; refresh it with `git fetch origin --prune` and inspect the
repository API or GitHub branch selector.

The current development publication head is
[`6c99910`](https://github.com/luoge850-lang/Strategic-GraphRAG/commit/6c99910e18c5072e4c19ff49b3916a70a6d40a04).
Its GitHub Actions run [35821223049](https://github.com/luoge850-lang/Strategic-GraphRAG/actions/runs/35821223049)
completed successfully. The preceding content commit
[`02ce1bd`](https://github.com/luoge850-lang/Strategic-GraphRAG/commit/02ce1bd6f93504f1e72525bf77ff2064cd6535bd)
also passed CI; neither result applies to `stable`.

## CI contract

`.github/workflows/ci.yml` runs on pushes and pull requests. It currently
executes Python compilation, whitespace checks, the full Python test suite, and
the frontend TypeScript/Vite build. A green run is valid only for its exact
commit and branch. It does not prove Neo4j/Chroma availability, external LLM
repeatability, semantic extraction quality, production recovery, or load
capacity.

## Publication policy

1. Keep `stable` and existing tags unchanged.
2. Review and commit only files classified as current source, current tests,
   current public documentation, and reproducible visual sources.
3. Keep `.env`, raw PDFs, vector stores, extracted text, response-cache data,
   recovery archives, and personal drafts out of the public commit.
4. Push the reviewed commit to the development branch and wait for its CI.
5. Create a release only after the user explicitly accepts the new results and
   the Claim Ledger has no unresolved contradiction.
