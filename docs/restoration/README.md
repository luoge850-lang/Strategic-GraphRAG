# Verified restoration workflow — 2026-10-27

This source-publication branch includes the latest local engine, pipeline, UI, tests and dated deployment entry points. It is not a new production candidate or a merge into stable. The frozen local runtime and its source fingerprint remain preserved.

## Identity boundary

The preserved local candidate is `build_8a5ce6d74be82a7c`, with source fingerprint `5ff28e24e8b7dfc91a3af2ab4c51bac2f22abc17e600f62fac45d9e57d3ee942`.

This public source has fingerprint `684c50fd106545572c7d99152431842731a9ea74ae4e937fe53ca1030df027ce`. Core Python package bytes are unchanged. Three historical deployment wrappers now resolve the original workspace through `GRAPHRAG_ORIGINAL_WORKSPACE` with a user-home fallback; another wrapper's documentation examples have private paths removed. The public template development API key field is also cleared. These source changes require their own build identity. **Do not pair this public source with the frozen local package and bypass the identity guard.** Rebuild into a new isolated artifact/database/collection to run this public revision against PDFs.

## What was actually tested

- Two independent frontend copies were installed from their respective package locks using `npm ci --offline --no-audit --no-fund`; both production builds passed. The current frozen frontend bundle was not overwritten.
- A new Python3.12 virtual environment was created without system site-packages. Runtime requirements were installed offline from existing pip-cache wheels, constrained to the original installed versions. `pip check` passed. Actual installed runtime/test versions are in [runtime-pins-2026-10-27.txt](runtime-pins-2026-10-27.txt).
- Full tests on restored Python: **418 passed,2 failed**,2 warnings and7 subtests passed. The two failures also reproduce in the original Python environment. The complete public source plus necessary fixtures has the same418/2 result.
- Existing test failures are `test_three_year_income_statement_preserves_oldest_value` and `test_sga_change_columns_are_not_metric_values`. Their exact-dictionary assertions expect older minimal observations while the extractor returns additional metadata. No production code or test assertion was changed to force a pass. This branch is therefore a review candidate, not a green-test release.
- The frozen local service was started with the restored environment, loading the original Neo4j state and1861-chunk collection. NVIDIA interest-income, segment and balance requests, a TSMC request and a financial ratio were exercised. The browser displayed the current build,695 observations, and the actual original PDF physical52page after a citation click. These are operational checks, not a new independent accuracy benchmark.

The latest experiment's23/25 source-first sample score is historical AI/PDF evidence with known-family and small-sample limits. Human review for that experiment is0. Earlier annotation fixtures are retained for historical tests and do not establish human Gold for this candidate.

## Reproduction levels

### Saved-score recomputation

From the repository root, Python's standard library is sufficient:

```powershell
python experiments/core-accuracy-20261018/recompute.py --output NEW-summary.json
```

The output path must not exist. The public records include both rounds' identity metadata, labels, raw responses, scorer and separate AI/PDF adjudications required by this command. It recomputes saved results, not live answers. Public test-log copies have private paths redacted; original local logs remain unchanged.

### Dependencies and frontend

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r docs/restoration/runtime-pins-2026-10-27.txt
.\.venv\Scripts\python.exe -m pip check
cd frontend
npm ci --no-audit --no-fund
npm run build
```

The recorded restoration was offline with an existing local wheel index and npm cache. Those caches and interpreter binaries are not uploaded; a new machine needs package-index access or a separately supplied verified wheelhouse. The pinned Windows/Python3.12 environment does not prove Python3.14 legacy environments or other platforms can be restored.

### Original-source rebuild and real services

Obtain the five original hash-matched reports, compatible Neo4j5.26/JDK21, the embedding model resources, and private local configuration. Use a new build output, empty isolated database and new collection. Original source registries, credentials, live databases, vector runtimes and PDF payloads are intentionally absent. Historical scripts may require earlier artifacts at their documented relative locations; they are retained for traceability, not advertised as a clean-machine one-click installer.

## Cleanup boundaries

Only explicitly verified rebuildable dependencies and task-created validation copies are eligible for this cleanup. User changes, original PDFs, historical failures, current/rollback packages and Python environments without a demonstrated restoration are retained. An exact local compressed backup of idle dependencies is additional recovery protection, not an independent off-disk backup. Moving or compressing on one disk is not by itself a disk-space release measurement.

Real model generation remains unconfigured and unmeasured. No runtime reliability or large-scale ingestion claim follows from these restoration checks.
