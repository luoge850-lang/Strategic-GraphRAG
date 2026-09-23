# Visual verification record — 2026-09-23

This record documents a live local browser verification of the Demo. It is
not a generated product mockup and is not a substitute for a release build.
The browser was opened only after the local launcher reported readiness.

## Runtime identity

| Field | Observed value |
|---|---|
| URL | `http://127.0.0.1:8000/` |
| Runtime | Local FastAPI + React/Vite Demo |
| Graph view | 134 nodes, 381 edges, 10 entity types |
| State | `ready` |
| Data scope | NVIDIA 2023/2024/2025 10-K development snapshot |
| Local source baseline | `50f696a` before the publication commit |
| Runtime build identity | `build_28697a5a0ce7bb70` |
| Capture label | `demo-live-2026-09-23` |

The live browser capture showed the real graph view and its `ready` state. The
old `docs/demo-dashboard-v3.png` image is not used as a public screenshot
because it represents a stale `0 NODES` state.

## Evidence-trace query

Query: `Compare revenue in 2023, 2024, and 2025`

Observed result:

- Execution completed, but the answer was withheld with `GROUNDING_FAILURE`.
- The UI exposed a verified evidence trace instead of returning an unsupported
  synthesis.
- The trace contained three claims: FY2023 page 85,
  FY2024 page 79, and FY2025 page 38.
- The interface reported that the retrieved evidence did not support the
  asserted relation. This is a visible fail-closed behavior, not a successful
  answer-quality score.

## Reasonable abstention query

Query: `How does NVIDIA mitigate supply chain risks?`

Observed result:

- Execution completed with `ANSWER: ABSTAINED` and `GROUNDING: NOT_APPLICABLE`.
- No causal path was returned.
- Related corpus evidence was reported as potentially relevant, but the UI
  explicitly prohibited treating it as proof of the requested relationship.

This is the intended distinction between “no direct verified answer” and an
unsupported absence claim.

## Screenshot and release policy

The live captures were displayed from the actual local runtime during this
audit and marked as user-facing deliverables. The repository intentionally
does not check in a machine-specific browser bitmap: the graph and external
service state are not shipped with the source tree, and a stale image could
be mistaken for a reproducible result. Before a public release, a maintainer
may export these same three states from a clean, replayable build and add
them under `docs/screenshots/` with the commit, build ID, data scope, and
`live`/`replay` label in the filename.
