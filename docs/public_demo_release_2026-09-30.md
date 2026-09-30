# Protected public demo — delivery boundary

Release classification: **reproducible experimental candidate; public production
deployment remains incomplete**. No server/domain is available, and the live
legacy stores cannot be accepted as the isolated candidate.

## Verified artifacts and checks

- Candidate build: `build_7feb21b48e594a7a`.
- Runtime-source fingerprint:
  `ab2693cf127abd39a28c6d70f576ae996778302ee225e11dec8e574b389606d2`.
- Immutable package checks passed before/after inspection, with all three PDF
  hashes matching. Deployment/docs/workflow edits are **outside this existing
  runtime fingerprint's inputs**. They are identified by Git commits, not
  retroactively included in the old build identity.
- Linux GitHub CI for engineering commit `d0758c3` completed successfully:
  full Python suite 227 passed, 7 subtests passed, 2 lifecycle warnings;
  deployment/scoring checks 5 passed; frontend build passed.
  [Actual run](https://github.com/luoge850-lang/Strategic-GraphRAG/actions/runs/36703787994).
  These counts do not measure answer accuracy or deployment reliability.
- A new clean checkout at `d0758c3` and a newly created Python 3.12.14 virtual
  environment ran the minimal deployment/scoring tests: 5 passed.
  Offline response recomputation reproduced 5/5 HTTP execution and 1/5 scenario
  acceptance. Ten recorded artifact hashes matched after checkout.
  Installed minimal-test dependencies: pytest 9.1.1, colorama 0.4.6,
  iniconfig 2.3.0, packaging 26.3, pluggy 1.6.0, Pygments 2.21.0.
  This fresh-environment check **did not install/run the complete application**;
  it establishes only the minimal public raw-record reproduction path.

## Real service, not a mock

The local service used the configured remote Neo4j and existing Chroma.
The remote database was reachable; a system database inventory exposed one
distinct non-system database, not an already available separate candidate
database. No new database was created or old data overwritten.

Production inspection found 381 claims, 381 business relationships and 234
financial observations without the candidate build ID; active Chroma had
1,686 unbound records rather than 843 candidate chunks. All four numerical
smoke cases returned insufficient evidence. Only the out-of-corpus refusal
assertion passed. Readiness returned a dependency timeout.

The actual browser displayed the revenue evidence, and a citation click
opened the 2025 PDF at physical page 80. This proves that one local
query/display/citation path worked, **not** that the numerical question,
candidate import, public access, recovery or complete browser suite passed.
See [raw responses and genuine screenshots](../experiments/public-demo-delivery-2026-09-30/README.md).

## GitHub delivery and stable boundary

Engineering changes update [existing PR #1](https://github.com/luoge850-lang/Strategic-GraphRAG/pull/1)
on `codex/financial-evidence-qa-delivery-2026-09-29`.
The independent homepage-only [PR #3](https://github.com/luoge850-lang/Strategic-GraphRAG/pull/3)
targets `stable`, at `474b3c5`, and its four CI checks passed.
The README includes an evidence journey graphic, a source-row example,
collapsible experimental details, actual browser captures, and explicit
failure/production limits. The screenshots link to an immutable engineering
commit; the documentation PR does not need local untracked files.

Neither PR was automatically merged. `stable` remained
`c69b2d2d39e3ffe785160f7ffad677f4c1b9a8fe`.
User untracked materials in the original checkout were preserved.
No production pointer was switched. No new release package was advertised.

## Necessary next decisions and checks

1. Supply a host and DNS domain, or select a hosting service and budget.
   Do not expose the current unbound database as a validated public candidate.
2. Provide a dedicated Neo4j candidate database and verified complete import.
   Global graph queries make same-database label-only separation insufficient.
   The preflight performs identity/provenance checks; it is not independent
   PDF-based validation of every financial observation.
3. On that host, test Linux dependencies and the default ONNX model's
   acquisition/hash/cache, container startup, protected HTTPS, numerical
   acceptance, restart and real rollback. The current Docker recipe is
   unexecuted and not a fully pinned/offline deployment.

The [deployment recipe](../deployment/README.md) fails closed on missing
configuration or mismatched stores. Public authentication and proxy route
restrictions are prepared, not verified live. Generation remains optional
and may incur provider costs; no public paid-generation budget is authorized.
