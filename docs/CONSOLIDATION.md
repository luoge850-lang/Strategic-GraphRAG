# Consolidation receipt

## Active closure work

Branch `codex/quality-closure` preserves the prior uncommitted repairs. Accounting aliases, preceding unknown obligations, split header dates, Docker exclusions and per-question evaluation persistence were reviewed. Response validation now uses the existing API response model once; arrays/null/invalid objects become recorded invalid responses. Results use a context-managed exclusive file, and failures stay in the 25-question denominator. The latest full run before the final dependency-error classification adjustment was 440 passed, 7 subtests passed; pip check passed.

The API and database are stopped. No new LIVE_REPLAY result or final build is claimed. Exact deletion targets `.local/build` and `.local/neo4j` were resolved within this project, then their removal was rejected with `blocked by policy`. Neither directory was deleted. Their removal is the minimum manual action needed before the existing build/import flow can resume; no identity check was disabled and no parallel runtime was created. New period completeness and R03/R12 HTTP outcomes remain unmeasured. Historical results below retain their original identity.

Scope: current workflow and GitHub source delivery. Disk cleanup is blocked, not completed.

## Source and runtime

Started from PR #4 head `db2f323a01983331458978451f0bd6285c7664a9`, not the older desktop branch. Compared current engine, frontend, scripts, tests and dependencies with the dirty delivery worktree. Its only source differences from PR #4 were four historical deployment wrappers and the cleared example API key; no unique current engine/frontend implementation was omitted.

Canonical source commit: `90f8e928d0ea57487203f7393bfc28722f3215c8`. Current build: `build_3b918db3d1355d09`; full component/source/PDF fingerprints are in [build identity](../evaluation/current/build-identity.json). Later evidence/documentation commits do not alter runtime sources. Active API: loopback port 8000; Neo4j: 7687/7474. Runtime/configuration: `.local/`; active Python: `.venv/`.

New generic entry points directly build from the five canonical PDFs, import into an empty local database and serve the matching graph/vector/source registry. The 25-question evidence set and its scorer no longer import dated deployment scripts. Historical scripts are still physically present because deletion was rejected; the canonical flow does not call them. Their removal has not been represented as completed.

Three graph lookup methods now propagate external database failures instead of silently returning empty/default evidence. The API maps database connectivity/session failures to 503. Original PDF hash checks are cached per loaded registry identity, rather than repeated for every page image. Vector storage is explicitly configured; missing expected collections cannot be silently created. Financial period/currency/scope/operand checks remain intact.

## Actual verification

- Full existing test suite: 420 passed, 7 subtests passed, 2 warnings. No tests removed/skipped to fix CI.
- Both stale table assertions retain strict years, values, signs, column order and exclusion of change columns; source metadata has separate assertions.
- New dependency environment restored from offline wheels; both restored and active environments passed `pip check`. This is a real restore test, but the duplicate restored environment has not been removed.
- `npm ci --offline --no-audit --no-fund` and production build succeeded.
- Fresh PDF build: 815 physical pages; 1,914 vector chunks, 695 financial observations and 3 business dependency/risk records. Zero generation-model calls; generation not evaluated.
- Neo4j import: 304 evidence claims, 698 expanded fact edges; full database 1,398 nodes / 5,389 relationships including provenance auxiliaries.
- Stop/restart was exercised with the imported current database, then 25 API requests were replayed. Windows console mode requires stopping the owned process tree; no OS service was installed. Recovery succeeded. This is process termination/recovery, not graceful shutdown.
- Real browser: five cases in [browser check](../evaluation/current/browser-check.json); actual PDF page 52 clicked and inspected. Back navigation preserved the question/result. A build change cleared the previous build's persisted answer once the current inventory loaded.

| Current 25-question diagnostic | Numerator | Denominator | Not judged / NA |
|---|---:|---:|---:|
| Core semantic/numeric identity correctness | 23 | 25 | 0 |
| Exact labeled citation reference | 22 | 25 | 0 |
| Complete request/source/metadata joint | 9 | 25 | 0 |
| Incorrect refusal | 1 | 25 | 0 |
| Severe incorrect successful answer | 0 | 25 | 0 |

Both original recomputation and current live replay produce these results. R03 and R12 remain failures. Original labels were exposed after their historical run; this is a known-family regression, not independent unseen validation. Review: AI/PDF diagnostic; human Gold reviews: 0. Citation identity/location scoring does not establish exhaustive original-text semantic support. New graph semantic accuracy/recall and LLM generation were not measured in this maintenance task.

## Space and file operations

Snapshot excludes shared `.git` object internals, does not follow reparse points, and covers both project roots, two other registered worktrees and three identified project audit directories. Global caches/installations were not changed. Before free-space observation was recorded after initial source preparation, before environment restoration and builds.

| Measurement | Before | After snapshot |
|---|---:|---:|
| Registered worktrees | 4 | 4 |
| Files in scoped roots | 216,458 | 251,991 |
| Logical bytes in scoped roots | 13,632,372,161 | 14,677,100,077 |
| Disk free bytes | 49,007,378,432 | 47,299,416,064 |

Actual deleted files: **0**. Logical deleted bytes: **0**. Disk free-space change: **-1,707,962,368 bytes**; this is a decrease while restoring dependencies/building, not released capacity. Same-volume moves of the superseded preflight build/database were only namespace changes and are not counted as cleanup. Their directories still exist. Logs/current files continued changing after the timestamped snapshot.

Automatic execution review rejected (1) the combined generated source-deletion batch, and (2) a separately preflighted exact `reports` directory deletion after checking its absolute path, reparse points and running processes. Both returned only `blocked by policy`; neither command executed. No alternative tool was used to bypass these decisions. The source batch was not executed, so no old scripts, components or tests were deleted.

Still pending: obsolete `archive/`, `reports/`, `experiments/`, dated deployment chains, duplicate environments/runtime stores, the three old worktrees, project-specific audit copies, and individually classified old docs. User untracked docs/config/tmp remain present. Current tree is source-committed but not fully clean because these user documents remain untracked. No claim of one remaining worktree or released disk capacity is made.

## Current structure and inventory

```text
strategic_graphrag/   runtime, typed facts and financial rules
frontend/            current frontend
scripts/manage.py    canonical build/import/local service
scripts/evaluate.py  current evidence recomputation/replay
requirements*.txt    active runtime/dev + current constraints (legacy files pending deletion)
evaluation/current/  references, original and current responses, scorers, fingerprints
data/                corpus/source manifests; five original PDFs local only
.local/              ignored configuration/current graph/vector/Neo4j; pending preflight copies
docs/                current inventory, receipt and two browser evidence images
```

[FILE_INVENTORY.csv](FILE_INVENTORY.csv) enumerates current tracked/new public files, all five canonical PDFs with hashes, and grouped dependency/store directories. Pending-removal directories and legacy files are explicitly marked; this is not a claim to have enumerated every file in the blocked historical forests. All 4 worktrees remain registered; only the desktop tree is used by the new workflow. Old research/deployment branches are retired as current entry points, but their branch refs/history are retained while worktree cleanup is blocked.

## GitHub and reproducibility

[PR #4](https://github.com/luoge850-lang/Strategic-GraphRAG/pull/4) is the sole source-delivery PR for this work. [Its checks](https://github.com/luoge850-lang/Strategic-GraphRAG/pull/4/checks) are the authoritative remote CI status; local test success is not substituted for Actions results. Final merge SHA/status is reported after the actual remote operation.

README contains independent install/build/import/start/replay commands. External requirements are Python 3.12, Node, JDK21, Neo4j5.26, ONNX MiniLM resources and the five exact PDFs. No command in the canonical flow requires an old worktree or dated experiment directory. Original NVIDIA download provenance is incomplete; source byte identity is explicit in the corpus manifest.
