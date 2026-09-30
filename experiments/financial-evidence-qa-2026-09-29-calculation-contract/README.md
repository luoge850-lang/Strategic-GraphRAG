# 2026-09-29 calculation-contract repair and development replay

This bundle preserves three source-matched local candidate runs. The latest is
`build_7feb21b48e594a7a`; `build_1aa1530fdd382d17` and
`build_38321fdf7366bf38` remain separate historical runs. All are
**development experiments**, not a held-out test, not human Gold, and not
proof of a stable live service. The latest run binds the retrieval matrix to a
newline-stable source fingerprint after a clean Windows checkout exposed
line-ending conversion in both raw-file hashes and source identity. No
retrieval method, protocol, label or threshold changed and no tuning was
performed. Reusing the development labels for this third run exceeds the
frozen v4 protocol's primary-run budget; it is disclosed as a protocol
deviation, and scores/timings are not pooled.

## Artifacts

- [`financial_retrieval_raw_20260929_build_38321fdf7366bf38.jsonl`](financial_retrieval_raw_20260929_build_38321fdf7366bf38.jsonl): 156 raw retrieval records across six methods and 26 question forms.
- [`financial_retrieval_raw_20260929_build_38321fdf7366bf38.manifest.json`](financial_retrieval_raw_20260929_build_38321fdf7366bf38.manifest.json): run conditions, build/source identity, protocol/data/lock hashes, request timing, peak RSS, and source-PDF hashes.
- [`financial_retrieval_summary_20260929_build_38321fdf7366bf38.json`](financial_retrieval_summary_20260929_build_38321fdf7366bf38.json): metrics regenerated from the raw records and linked label files, with unrun answer metrics explicit.
- [`financial_retrieval_summary_20260929_build_38321fdf7366bf38.png`](financial_retrieval_summary_20260929_build_38321fdf7366bf38.png): chart generated from the same summary data.
- [`financial_retrieval_raw_20260929_build_1aa1530fdd382d17.jsonl`](financial_retrieval_raw_20260929_build_1aa1530fdd382d17.jsonl): second source-matched 156-record replay, retained separately.
- [`financial_retrieval_raw_20260929_build_1aa1530fdd382d17.manifest.json`](financial_retrieval_raw_20260929_build_1aa1530fdd382d17.manifest.json): that build's identity, hashes, runtime, resources and per-method timing.
- [`financial_retrieval_summary_20260929_build_1aa1530fdd382d17.json`](financial_retrieval_summary_20260929_build_1aa1530fdd382d17.json) and [chart](financial_retrieval_summary_20260929_build_1aa1530fdd382d17.png): summary and chart for the second replay, not the latest run.
- [`financial_retrieval_raw_20260930_build_7feb21b48e594a7a.jsonl`](financial_retrieval_raw_20260930_build_7feb21b48e594a7a.jsonl): 156-record replay bound to newline-canonical source fingerprint `ab2693cf127abd39a28c6d70f576ae996778302ee225e11dec8e574b389606d2`.
- [`financial_retrieval_raw_20260930_build_7feb21b48e594a7a.manifest.json`](financial_retrieval_raw_20260930_build_7feb21b48e594a7a.manifest.json): latest exact build identity, input hashes, runtime conditions, and measured timing/resource record.
- [`financial_retrieval_summary_20260930_build_7feb21b48e594a7a.json`](financial_retrieval_summary_20260930_build_7feb21b48e594a7a.json) and [chart](financial_retrieval_summary_20260930_build_7feb21b48e594a7a.png): latest metrics recomputed from that raw run.
- [`calculation_contract_after_20260930.json`](calculation_contract_after_20260930.json): calculation regression outcome on the newline-stable candidate; independent test accuracy remains unmeasured.
- [`baseline_calculation_contract_20260929.json`](baseline_calculation_contract_20260929.json) and [`calculation_contract_after_20260929.json`](calculation_contract_after_20260929.json): concrete before/after calculation failures and regression evidence.
- [`scoring_sensitivity_20260929.json`](scoring_sensitivity_20260929.json): strict grade-3 sensitivity for the historical v4 run; it does not edit or replace the frozen primary scores.
- [`multi_model_dispatch_record_20260929.json`](multi_model_dispatch_record_20260929.json): only public dispatch requests, outputs, adopted decisions, and verifiable limits; actual backend model/effort remain `NOT_VERIFIED`.
- [`delivery_hash_manifest_20260929.json`](delivery_hash_manifest_20260929.json): hashes for source PDFs, protocol/labels, raw run, summary, chart, calculation records, and environment lock.

The runner used these unchanged experiment inputs:

- Dataset: [`../financial-evidence-qa-2026-09-28/financial_qa_dev_source_review_20260924_v3.jsonl`](../financial-evidence-qa-2026-09-28/financial_qa_dev_source_review_20260924_v3.jsonl), SHA-256 `086083c4e9ab4d53fbffa869740276d83c50979b1a501d26fb9f887f74c1a3af`.
- Frozen protocol: [`../financial-evidence-qa-2026-09-28/protocol_v4.md`](../financial-evidence-qa-2026-09-28/protocol_v4.md), SHA-256 `d2a5baca4e49dcdf95444ea63e84050cbc44e2d14f0283183402e6080d28a4ee`.
- Table review input: [`../financial-evidence-qa-2026-09-28/table_quality_ai_visual_diagnostic_2026-09-24_v2.jsonl`](../financial-evidence-qa-2026-09-28/table_quality_ai_visual_diagnostic_2026-09-24_v2.jsonl), SHA-256 `9a9f9d709b10729066a2285ebad64f0c5c5dfc008ca5b1187ddf657ce3855536`.

The latest six-method matrix ran with answer generation disabled: 156/156
retrieval calls succeeded. Only the explicitly judged direct-support pages were scored; the
23 supported question forms belong to 17 semantic families and their 30
judged page instances are non-exhaustive. The headline request hits are not
full-corpus Recall@k. nDCG, numeric answer accuracy, complete-fact accuracy,
citation support/locator correctness, and answer abstention quality were not
measured by this retrieval-only matrix. The separate isolated candidate's
8/8 engineering acceptance checks are contract checks, not accuracy scores.

## Recompute

Follow the build and replay commands in the
[2026-09-29 repair report](../../docs/financial_qa_candidate_repair_delivery_2026-09-29.md).
Obtain local PDFs as described in [`data/README.md`](../../data/README.md) and
verify their hashes first. Do not upload PDFs, live database files, Chroma
runtime copies, or the ignored immutable candidate package in this bundle.

The source fingerprint normalizes CRLF/CR to LF for text files. The dated
artifact directory is marked `-text` in `.gitattributes` because its sidecars
bind byte-exact SHA-256 values to the JSONL/JSON records. On Windows, keep that
attribute enabled; do not run checkout with filters that rewrite the experiment
bytes. A clean Windows checkout at delivery commit `12dbe760456b9e6af1bfc12cfe2739f134c59495`
confirmed the source fingerprint and raw-record hash, passed all 227 tests, and
recomputed a byte-identical summary JSON from the committed raw records.

Historical v4 results belong to `build_f74bb1dfbf96b8a2` and remain in the
[2026-09-28 bundle](../financial-evidence-qa-2026-09-28/README.md); neither
their scores nor their timing are silently combined with this replay.
