# Version and release strategy

## Current refs

| Ref | Role | Observed commit / state | Public policy |
|---|---|---|---|
| `stable` | GitHub default branch | `297379f` at audit time | Preserve as the frozen public release identity. Do not mix its metrics with development metrics. |
| `codex/v3-three-filing-evidence-graphrag` | Active development branch | `50f696a` before this publication commit; local work was dirty | Publish new work here first. CI must pass for the resulting commit before any release decision. |
| `v3.2.0-research-freeze-2026-09-14` | Historical tag | Existing immutable tag | Keep historical claims unchanged. |
| Local HEAD | Working checkout | Dirty before publication work | A dirty checkout has no inherited CI status. Record the final commit after review. |

The repository API currently reports `stable` as the default branch. The local
remote-tracking `origin/HEAD` may be stale and is not used as the authority for
this statement; refresh it with `git fetch origin --prune` and inspect the
repository API or GitHub branch selector.

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
