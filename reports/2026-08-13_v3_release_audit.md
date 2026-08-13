# Strategic-GraphRAG v3 Release Audit

Date: 2026-08-13

## Verdict

Status: engineering candidate pass with research and deployment limitations.

The release is suitable as a technically substantial graduation/portfolio
prototype. It is not yet sufficient evidence for a paper claim of superior
retrieval, complete extraction, true temporal causal reasoning, or production
readiness.

## Verified release state

- Corpus: NVIDIA 2023, 2024, and 2025 10-K only; 395 PDF pages.
- Strict graph: 383 EvidenceClaims and 383 evidence-backed business edges.
- By filing: 126 / 129 / 128 claims.
- Evidence pages: 44 / 41 / 43.
- Vector collection: 1,825 active chunks.
- Stable IDs: 383/383 `claim_v2_*`; zero collisions.
- Strict two-hop chains: 49; invalid chains: 0.
- Derived temporal index: 99 `NEXT_DISCLOSURE` links with non-causal semantics.
- Legacy post-check: zero old edges, evidence nodes, or Chroma collections.
- Incremental plan: all three PDFs unchanged; no rebuild required.
- Tests: 10/10 focused contracts passed; Python compilation passed.
- Frontend: TypeScript/Vite build passed; largest JS chunk about 422 kB.
- API: Neo4j connected, DeepSeek v4 Flash configured, browser UI loaded 130
  visible entity nodes and 383 strict edges in all-filings scope.

## Functional correction found during browser audit

Before the final fix, a three-year revenue comparison anchored on `REVENUE` but
returned risk-to-revenue edges and incorrectly abstained from reporting actual
revenue. Exact metric queries now constrain retrieval to `REPORTS_METRIC`.

Post-fix result:

- FY2023 revenue: $26,974 million, 2023 10-K p.86.
- FY2024 revenue: $60,922 million, 2024 10-K p.79.
- FY2025 revenue: $130,497 million, 2025 10-K p.80.
- Returned years: 2023, 2024, 2025.
- Returned relations: three `REPORTS_METRIC` edges.
- Grounding status: `VERIFIED`.
- Engine latency in this check: 11,197 ms.

## Performance audit

A cold request after process restart took 16,551 ms wall time and 13,646.62 ms
inside the engine. Recorded stages included 1,617.38 ms graph path search and
5,137.86 ms LLM synthesis. The remainder includes connection, anchor resolution,
validation, and serialization that require finer instrumentation.

An identical second request returned in about 9 ms with `cache.hit=true`.
Caching is exposed in metadata and is not reported as uncached model latency.

The initial slow request also revealed that synchronous inference blocked the
FastAPI event loop. Query execution now runs in FastAPI's thread pool, so health
and lightweight endpoints remain responsive. Cross-Encoder loading is disabled
by default and can be explicitly enabled; deterministic causal scoring remains
the default demo path.

## Evidence and data integrity work

- Evidence IDs are content-derived from PDF identity, page, normalized quote,
  source, relation, and target. Runtime chunk offsets no longer control IDs.
- A corpus manifest records PDF SHA-256 values, ontology hash, provider/models,
  embedding model, vector collection, and Git state.
- A complete local archive was written before physical legacy cleanup. It
  includes graph properties and all 100 old 384-dimensional embeddings.
- After cleanup, active counts and strict paths remained unchanged.

## Deliberately not completed

- No new manual Golden QA set was created.
- No additional PDFs were ingested.
- The stale 38-item generated QA candidate was not presented as valid quality
  evidence.
- The final five-question external-LLM audit was not run because that action
  would transmit 2023/2024 filing evidence without explicit audit authorization.
- No arbitrary LLM-generated Cypher or autonomous agent layer was added.

## Priority assessment

P0 before paper claims:

1. Human-labeled extraction set and entity/relation precision, recall, F1.
2. Human Golden QA with retrieval and answer metrics plus unanswerable cases.
3. Error analysis by section, table/text source, relation, and year.

P1 before public deployment:

1. API authentication enabled, restrictive CORS, secrets manager, TLS gateway.
2. Request timeout/cancellation, concurrency limits, persistent cache strategy,
   structured logs, tracing, cost and provider-error monitoring.
3. Python 3.11/3.12 pinned environment, container build, CI, reproducible lock.

P2 research novelty:

1. Evidence-grounded temporal change classification beyond disclosure order.
2. Confidence calibration and contradiction detection across filings.
3. Graph/vector/reranker/evidence-guard ablations and statistical testing.
4. Multi-company edges only after the single-company extraction benchmark is
   credible; otherwise cross-company questions overclaim available evidence.

## Objective level

Current level: above a typical undergraduate RAG demo and credible as a strong
engineering portfolio project. For a graduation thesis, it needs a formal
research question, labeled data, baselines, metrics, ablations, and error
analysis. For a publishable paper, it additionally needs novelty demonstrated
against current GraphRAG/financial QA baselines with statistically defensible
results. For commercial readiness, governance, reliability, observability,
security, load testing, and deployment evidence remain mandatory.
