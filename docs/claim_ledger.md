# Public claim ledger

This ledger is the publication boundary for the development branch. A claim
is public only in the narrow form recorded here. `NOT_BOUND` is intentional:
the local Neo4j/Chroma stores are external assets and are not currently bound
to an immutable repository build ID. It must not be silently replaced by a
marketing version number.

The rows use `50f696a` when that is the source/evaluation baseline for a
recorded result. The public documentation and current source are published at
`8106c76` (with content first published at `02ce1bd`); the version strategy
records the successful CI runs. This distinction
prevents a publication commit from being mistaken for a fresh rerun of every
historical benchmark.

| ID | Allowed public statement | Commit/tag | Build ID | Data scope | Source report / artifact | Test or audit command | This run? | Limit / prohibited wording |
|---|---|---|---|---|---|---|---|---|
| C01 | The development checkout covers NVIDIA fiscal 2023/2024/2025 10-K filings. | `50f696a` dev baseline | `NOT_BOUND` | Three filings | `docs/canonical_project_state_2026-09-19.md`; corpus-manifest workflow | `scripts/create_corpus_manifest.py` | No; recorded inventory | Raw PDFs are local/ignored. Do not imply the public repository ships the filings. |
| C02 | The development snapshot records 395 physical pages, 381 active strict EvidenceClaims, and 1,686 vector chunks. | `50f696a` dev baseline | `NOT_BOUND` | 2023/2024/2025 snapshot | `docs/canonical_project_state_2026-09-19.md` | `/health/ready`; local manifest workflow | No; recorded inventory | Do not mix with stable’s historical 383-claim snapshot or older 362/843 inventories. |
| C03 | Active claims carry quote and provenance fields within the current audit boundary. | `50f696a` dev baseline | `claim-v2-schema` | Active EvidenceClaims | `docs/research_evaluation_protocol.md`; semantic/provenance audit | `scripts/audit_research_readiness.py` | No; recorded audit | A `VERBATIM` match is not semantic correctness, precision, recall, or F1. |
| C04 | Vector, Graph, Hybrid, and Hybrid+Temporal retrieval modes are implemented. | `50f696a` dev baseline | `retrieval-ranking-v2` | 37-question Silver set | `strategic_graphrag/engine/retrieval.py`; benchmark script | `scripts/evaluate_retrieval_benchmark.py` | No; recorded implementation | Implementation is not evidence that any mode is universally best. |
| C05 | Graph scored higher than Vector on Recall@5 and MRR in the recorded graph-derived Silver regression. | `50f696a` dev baseline | `silver-ranking-v2-2026-09-18` | 37 questions; 32 answerable | `docs/public_results_2026-09-22.md`; local ranking-v2 report | `scripts/evaluate_retrieval_benchmark.py` | No; recorded result | This is a graph-derived regression with self-test bias, not QA accuracy or independent gold. |
| C06 | Hybrid+Temporal did not exceed Graph on the recorded Silver snapshot. | `50f696a` dev baseline | `silver-ranking-v2-2026-09-18` | Same 37-question set | `docs/public_results_2026-09-22.md` | Same benchmark command | No; recorded result | It does not prove temporal retrieval is generally ineffective. |
| C07 | Direct structural evidence ranking rejects membership/runtime/composition wording as sufficient proof of `PRODUCES`. | `50f696a` dev baseline | `build_28697a5a0ce7bb70` | Deterministic relation gate | `tests/test_pipeline_contracts.py` | `.venv\\Scripts\\python.exe -m pytest -q` | Yes | This is a rule contract, not a human semantic accuracy estimate. |
| C08 | The local launcher is click-to-run and does not configure Windows boot-time auto-start. | `50f696a` dev baseline | `build_28697a5a0ce7bb70` | Local Windows launcher | `open_demo.cmd`; `scripts/open_demo.ps1` | `tests/test_demo_launcher_contract.py` | Yes | Neo4j Aura and `.env` still must be available and correct. |
| C09 | Readiness checks cover the API, Neo4j, Chroma, and configured LLM before the browser is opened. | `50f696a` dev baseline | `build_28697a5a0ce7bb70` | Local live Demo | `scripts/open_demo.ps1`; `/health/ready` | `scripts/open_demo.ps1` plus live browser check | Yes | This is a local smoke check, not production deployment acceptance. |
| C10 | The 30-row answer-level Golden QA is a single-reviewer engineering checkpoint. | `50f696a` dev baseline | `golden-qa-v2` | 30 rows; reviewer `louis` | `evaluation/golden_qa_human_v2.jsonl` | `scripts/run_golden_evaluation.py --help` | No; recorded annotation | It is not independent two-reviewer gold or paper-grade agreement evidence. |
| C11 | The 60-row table-quality queue is a candidate annotation set separate from answer-level QA. | `50f696a` dev baseline | `table-queue-2026-09-19` | 60 candidate rows | `evaluation/annotation/table_quality_candidate_2026-09-19.jsonl` | `scripts/export_table_annotation_queue.py --help` | No; P2 in progress | Do not publish cell Precision/Recall/F1 before independent annotation and adjudication. |
| C12 | Recorded response record/replay is deterministic for the frozen cache. | `50f696a` dev baseline | `llm-extraction-cache-v1` | 170 keys; 126/126 replay | `docs/reproducibility_freeze_2026-09-18.md` | `scripts/audit_research_readiness.py` | No; recorded freeze | Cache replay does not prove fresh external-model repeatability; readiness remains `NOT_READY`. |
| C13 | The current local Python regression and frontend production build execute in `.venv`. | `50f696a` dev baseline | `build_28697a5a0ce7bb70` | Current checkout | `requirements-lock-2026-09-19.txt`; this audit | `.venv\\Scripts\\python.exe -m pytest -q`; `npm run build`; `git diff --check` | Yes | CI status belongs only to the exact pushed commit; this working tree was dirty during the run. |
| C14 | The project is a research/engineering prototype, not a production-ready service or completed paper result. | `50f696a` dev baseline | `acceptance-ledger-2026-09-21` | Current development boundary | `docs/acceptance_ledger_2026-09-21.md` | Readiness and acceptance ledgers | No; recorded judgment | Clean install, isolated-store cutover, live recovery, load, independent gold, and fresh-model repeatability remain incomplete. |

## Required wording discipline

- Use “verbatim provenance match” for quote/page linkage; use “semantic
  correctness” only when an independent label supports it.
- Use retrieval `Precision@K`, `Recall@K`, `nDCG@K`, and `MRR` only with the
  evidence unit, question set, denominator, and report version. Never call
  them answer accuracy.
- Use “engineering checkpoint” for the current single-reviewer Golden QA.
- Use “read-only staging adapter” for the JSON adapter. It is not a Neo4j
  production integration.
- Use “parser pilot not run” for Docling; do not claim a parser accuracy gain.
- Use “recorded rollback/recovery plan” until a live recovery drill passes.
- A local-only report path is provenance, not public availability. Public
  claims must remain reproducible from the checked-in protocol, command, and
  disclosed data boundary.
