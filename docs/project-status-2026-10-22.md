# Project overview and research status — 2026-10-22

Strategic-GraphRAG is a bounded financial-report research prototype. It combines PDF evidence extraction, Neo4j company/metric/dependency relations, Chroma text retrieval, typed financial observations, deterministic arithmetic and source-page review. Its central question is whether explicit identity, period, scope and evidence constraints can reduce answers to the wrong financial fact.

## Repository boundary

This is a documentation and derived-evidence update to the GitHub `stable` branch. **The latest local implementation is not included in this update.** The default-branch runtime, the older candidate branch and the current local research candidate are different versions. Cloning `stable` does not reproduce the latest five-report candidate, stores or scores. No live databases, private configuration, API credentials, original PDFs or local environments are published here.

Latest locally recorded candidate: `build_8a5ce6d74be82a7c`, frozen on 2026-10-18. Runtime source fingerprint: `5ff28e24e8b7dfc91a3af2ab4c51bac2f22abc17e600f62fac45d9e57d3ee942`. The 2026-10-22 audit matched the local source fingerprint and recomputed the saved score summary exactly. It did not rerun the API, extraction, browser evaluation or model generation. No matching project service listeners were observed in the inspected local port range during this audit; an old localhost browser tab is not evidence of availability.

## Corpus and implementation scope

The local candidate uses five primary reports, totaling 815 physical PDF pages:

| Issuer | Report | Physical pages | Principal financial table units |
|---|---|---:|---|
| NVIDIA | FY2023 10-K | 169 | USD millions |
| NVIDIA | FY2024 10-K | 96 | USD millions |
| NVIDIA | FY2025 10-K | 130 | USD millions |
| TSMC | 2023 Taiwan annual report | 326 | TWD thousands |
| TSMC | 2024 Taiwan annual report | 94 | TWD thousands |

The TSMC reports are not SEC 20-F filings. Fiscal calendars differ; no automatic currency conversion is implied. The recorded local candidate contains 695 financial observations, 1861 vector chunks and three narrow dependency/risk disclosure records. These are different inventories, not counts of independently audited correct facts. Bookkeeping edges must not be described as business knowledge volume.

Implemented local paths include company-scoped financial facts, supported arithmetic/unit conversion, independent request slots, disclosed dependency evidence and bounded conditional-risk explanations. Numeric answers are deterministic. Conditional explanations do not establish realized causality or a revenue-loss amount. Actual model-generation calls in the latest experiment: **0**; token and cost measurements are unavailable. Older model smoke tests belong to their historical builds.

## Recorded evaluation, recomputed on 2026-10-22

All questions in these groups were sent as natural language with generation and answer cache disabled in the October 18 service experiment. The October 22 action was offline score recomputation only.

| Group | Source-answerable core correct | False refusals | Other unfulfilled | Correct among successful numeric answers | Complete request/source/mechanical metadata |
|---|---:|---:|---:|---:|---:|
| Exposed development28 | 22/22 | 0/22 | 0 | 20/20 | 17/22 |
| Exposed development37 | 27/27 | 0/27 | 0 | 25/25 | 25/27 |
| Previous failed holdout, promoted to development | 10/10 | 0/10 | 0 | 10/10 | 8/10 |
| Source-first frozen sample25 | 23/25 | 1/25 | 1 NOT_REQUESTED | 23/23 | 9/25 |

The 25-question core score is 92.0%, with a descriptive Wilson95% interval of approximately 75.0%–97.8%. This is a small, selected sample with known metric families and correlated disclosure versions, **not universal accuracy or a fully unseen-family benchmark**. Human reviews: **0**. Mechanical scoring and separately retained AI/PDF adjudication must not be called human Gold.

The two retained failures are an unrecognized `Income from operations` synonym and an `Intangible assets, net` qualifier parsed as an extra requested item. A NOT_REQUESTED result is still an unfulfilled source-answerable request. The joint score of 9/25 shows why correct numbers alone do not establish complete fact identity. Exact frozen reference-page matching was 22/23 successful numeric answers; separate AI/PDF review supported an alternate source page for the remaining answer. The reference was not silently rewritten.

Zero severe wrong successful answers were recorded in these final groups, but earlier candidates had serious substitutions: accumulated OCI balance versus total comprehensive income, and annual revenue versus event-attributed loss. Their failure records remain local and must be retained. They are not erased by later development recovery.

[Machine-readable derived scores](research-status-2026-10-22/results.json) and [25-question diagnostics](research-status-2026-10-22/question-diagnostics.json) support inspection of these reported results. The latter is a derived diagnostic export, not the original raw HTTP response set. Original summary and ledger SHA256 values are included in the score export. Complete local raw records, frozen source packets, scorer and stores are not published by this documentation update; it does not establish clean-checkout reproduction of the latest candidate.

## Research and engineering gaps

- More representative, source-driven extraction coverage and necessary-field audits are needed. The number of extracted candidates is not extraction recall.
- Basic/diluted EPS, financial basis, complete dates and restrictive qualifiers require explicit identities and conservative handling.
- Earlier retrieval experiments did not establish a general graph advantage over fusion plus time constraints. Those historical results are not rerun scores for this candidate.
- The latest experiment did not establish a real LLM generation/verification loop, clean-machine installation or full engineering acceptance. Earlier lifecycle and concurrency checks must retain their own build identity.
- Graph paths support disclosed relationships and conditional explanations, not counterfactual identification, event occurrence or quantified investment outcomes.

The project is suitable as a bounded undergraduate research prototype with these limitations made explicit. It should not be presented as a reliable investment tool, a large-scale production platform or a generally validated causal-reasoning system.

## Local version and artifact management

Four registered Git working trees were observed: the original desktop checkout, active delivery checkout, detached reproduction checkout and README-only checkout. An additional clean reproduction clone is nested under a delivery report directory; it is not a registered worktree. The active delivery HEAD is `80dc9cc958711e6e7ed5da8c67a96a1f3ba1de59`, plus substantial uncommitted research code and evidence. A commit SHA alone therefore does not identify the current candidate implementation; the runtime source fingerprint is also required.

The read-only audit measured file metadata without traversing 413 reparse points. These logical sizes exclude skipped targets and are not allocated disk usage. It proposed no deletion based solely on age, ignore status or absence from search results. No local files were deleted or moved. See the [organization policy](project-organization.md) for the proposed separation of source, immutable experiments, large artifacts and disposable runtime state.
