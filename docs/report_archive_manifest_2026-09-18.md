# Report archive manifest — 2026-09-18

The following 19 superseded, duplicate, smoke-only, or console artifacts were
moved from `reports/` to the recoverable local archive
`archive/cleanup-2026-09-18/historical-reports/`. No report was deleted.

- `_semantic_rerun_console_2026-09-15.txt`
- `2026-08-14_corpus_manifest.json`
- `2026-08-14_extraction_quality_baseline.json`
- `2026-08-14_p0_p1_acceptance.md`
- `coverage_2025_post_rebuild_2026-09-03.json`
- `extraction_annotation_audit_2025_post_repair_v2.json`
- `extraction_quality_2025_post_rebuild_2026-09-03.json`
- `golden_qa_human_v2_answer_level_smoke_2026-09-16.json`
- `golden_qa_v2_results.json`
- `neo4j_snapshot_post_rebuild_2025_diagnostics_2026-09-03.json`
- `post_optimization_pipeline_stats.json`
- `research_readiness_2026-09-15.json`
- `retrieval_benchmark_hybrid_no_fusion_2026-09-14.json`
- `retrieval_benchmark_silver_2026-09-09.json`
- `retrieval_benchmark_silver_2026-09-14.json`
- `retrieval_benchmark_silver_2026-09-14_rerun.json`
- `retrieval_benchmark_silver_2026-09-15.json`
- `standard_query_audit.json`
- `strict_chain_audit_2025_post_rebuild_2026-09-03.json`

Follow-up cleanup after the 2026-09-18 ranking and answer-level reruns moved
three superseded Golden QA outputs to the same recoverable archive:

- `golden_qa_human_v2_answer_level_2026-09-16.json`
- `golden_qa_human_v2_four_modes_2026-09-16.json`
- `golden_qa_human_v2_structural_2026-09-16.json`

The same optimization pass also moved these superseded readiness and semantic
audit duplicates:

- `research_readiness_2026-09-16.json`
- `research_readiness_2026-09-18.json` (dated duplicate; the refreshed
  `research_readiness_current.json` remains active)
- `graph_semantic_consistency_2026-09-15.json` (superseded by the fresh
  `graph_semantic_consistency_2026-09-18.json`)
- `corpus_manifest_2026-09-15.json` (superseded by the 2026-09-19 canonical
  manifest)
- `research_readiness_2026-09-18_final.json` (dated duplicate of
  `research_readiness_current.json`)
- `retrieval_benchmark_silver_2026-09-15_latency_optimized.json` (superseded
  by the 2026-09-18 semantic-gate baseline)
- `runtime_performance_2026-09-19_n2.json` (duplicate two-question smoke
  report; the primary report remains active)
- `graph_semantic_consistency_2026-09-18.json` (superseded by the fresh
  read-only audit from 2026-09-19)

The active report directory retains the current corpus manifest, current graph
and temporal audits, the pre/post Neo4j snapshots, the 2025 cache record/replay
pair, the current Silver baseline, the four-mode Golden QA report, the current
readiness reports, and the files required by existing provenance scripts.
The 126-versus-131 repeatability report was deliberately retained at the active
level because it is still the blocking audit evidence. Three reports initially
selected for archiving were restored because the readiness audit still consumes
their exact filenames: `kg_audit_2025_post_rebuild_2026-09-03.json`,
`retrieval_baselines_smoke.json`, and
`strict_chain_audit_2025_post_rebuild_2026-09-05.json`.

Three historical report directories were also moved intact because they were
superseded evidence bundles and were only inflating the active report count:

- `handoff_2026-09-05/`
- `system_audit_2026-09-07/`
- `system_audit_2026-09-11/`

They remain recoverable under the same archive root. The active `reports/`
directory now contains 40 entries and passes the readiness hygiene threshold.
