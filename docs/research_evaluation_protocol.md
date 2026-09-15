# Strategic-GraphRAG research evaluation protocol

This document is the evaluation contract for the three-filing NVIDIA 10-K
experiment. It separates engineering integrity checks from semantic quality
claims. A report must not call a result a paper result unless the required gold
data and review metadata are present.

## Frozen experiment unit

- Corpus: the allowlisted NVIDIA 2023, 2024, and 2025 10-K PDFs.
- Retrieval units: `EvidenceClaim` IDs for graph retrieval and canonical
  `doc_id#page` keys when comparing graph and vector retrieval at a common page
  level.
- Every run records corpus hashes, source-tree hash, prompt version, model,
  embedding model, retrieval mode, top-k, cache state, timestamp, and git SHA.
- Every extraction run must fix and record `LLM_EXTRACTION_TEMPERATURE`.
  A value of 0.0 reduces sampling randomness for comparisons, but does not
  guarantee identical responses from an external model service.
- `uncertain` is counted as incorrect for the pilot extraction protocol, but is
  retained as a separate label count.

## Extraction evaluation

The existing 30-row post-repair sample is a predicted-claim sample. It supports
precision-like estimates only:

```text
precision-like relation score = accepted extracted relations / extracted relations
```

For Recall and F1, each sampled evidence unit must have a complete gold set,
including the empty set when no relation is present:

```text
TP = predicted relation in gold relation set
FP = predicted relation not in gold relation set
FN = gold relation not predicted
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

The stable key is:

```text
(doc_id, page, evidence_unit_id, source_id, relation_type, target_id)
```

Do not collapse two claims merely because their normalized triples match. If
the evidence text or page differs, retain both evidence units and record their
agreement separately.

### Annotation protocol

For each extracted claim, the annotator sees the exact evidence text and marks:

1. source entity correct;
2. target entity correct;
3. relation direction/type correct;
4. evidence explicitly supports the relation.

For a publication-quality label set, use two independent annotators or one
annotator plus an independent second pass on a random sample and every
disagreement. Report agreement per field and adjudicated final labels. A
duplicate-triple agreement score is not inter-annotator agreement.

### Current annotation status

`extraction_sample_2025_post_repair_v2.jsonl` is a historical prefilled
candidate audit. `extraction_sample_2025_post_repair_human_v1.jsonl` is a
30-row AI-assisted working set labeled with GPT-5.6/Sol and an explicit AI
annotator identity. It is not an independent human gold set and must not be
reported as human Golden QA. A publication-quality human Golden QA result
requires a separate blind human review or two independent human annotators,
followed by adjudication and reported inter-annotator agreement.

## Human Golden QA

The final QA file should contain 30--50 hand-verified questions across:

- single-hop relation lookup;
- multi-hop risk/strategy chains;
- financial table and year comparison questions;
- cross-filing temporal questions;
- unanswerable or insufficient-evidence questions.

Each row must include:

```json
{
  "id": "GQ-001",
  "question": "...",
  "reference_answer": "...",
  "gold_evidence_ids": ["claim_v2_..."],
  "relevant_evidence_grades": {"claim_v2_...": 2},
  "answerable": true,
  "requires_abstention": false,
  "review_status": "HUMAN_REVIEWED",
  "reviewer": "reviewer_a"
}
```

Candidate questions may be generated from the graph and evidence, but a human
must verify the question, reference answer, evidence IDs, and answerability.
`AUTO_GENERATED_REGRESSION_CANDIDATE` is a draft status and must not be used as
the paper Golden QA set.

## Retrieval metrics

Run vector, graph, hybrid, and hybrid-temporal on exactly the same reviewed
questions and frozen corpus. For each ranked result list:

```text
Precision@K = relevant retrieved units in top K / K
Recall@K = relevant retrieved units in top K / number of gold units
RR = 1 / rank of the first relevant unit
MRR = mean(RR over questions)
nDCG@K = DCG@K / ideal DCG@K
```

Report per-question scores, macro averages, 95% bootstrap intervals, and paired
comparisons between modes. Do not mix cached and uncached latency in one mean.
If page-level matching is used for fairness, label it as page-level retrieval;
do not present it as exact EvidenceClaim retrieval.

For this project, macro retrieval metrics and bootstrap intervals use one
question as the statistical unit. The percentile bootstrap must resample the
per-question scores with replacement, use a fixed recorded random seed, and
record the resampling count. A bootstrap interval is a descriptive uncertainty
summary; it is not a human-gold confidence interval and must not be used alone
to claim statistical superiority. Paired win/tie/loss comparisons use the same
`question_id` across modes and exclude errored, unanswerable, or missing-score
rows from that metric's eligible pairs.

### Automatic Silver benchmark

When human review is not yet available, `scripts/build_silver_benchmark.py`
creates `evaluation/silver_retrieval_v1.jsonl` from strict `VERBATIM`
EvidenceClaims, same-filing chains, and deterministic unsupported questions.
`scripts/evaluate_retrieval_benchmark.py` evaluates all four modes using the
common `doc_id#page` unit and records Recall@K, Precision@K, MRR, nDCG,
abstention accuracy, latency, per-question traces, question-type strata, and
question-level bootstrap/paired diagnostics. The report's primary retrieval
unit is `canonical_filing_page_key`; its paired and bootstrap unit is
`question_id`. This is an engineering regression and ablation instrument only:
its expected evidence is derived from the graph under test, so it has
self-test bias and cannot establish independent extraction recall, human answer
quality, or statistical superiority. The report must keep the status
`AUTO_GENERATED_SILVER_NOT_HUMAN_GOLD`, explicitly identify the data as
auto-generated Silver, and state that independent human Golden QA is missing.

The 2026-09-15 read-only semantic audit found 381 claims with complete linkage,
zero normalized quote mismatches, and zero obvious ontology direction/type
conflicts. It classified 69 repeated normalized-triple groups as different-
evidence duplicates and 0 as same-evidence duplicate groups. These counts are
machine classifications for review; different evidence is not proof of
semantic consistency and this audit is not human Golden QA.

The 2026-09-15 rerun in
`reports/retrieval_benchmark_silver_2026-09-15.json` observed the following
page-level macro results on 32 answerable questions (plus 5 deterministic
unsupported questions):

| Mode | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|
| Vector | 0.2188 | 0.0828 | 0.1103 |
| Graph | **0.7009** | **0.7083** | 0.6986 |
| Hybrid | 0.5134 | 0.4740 | 0.4764 |
| Hybrid Temporal | 0.5446 | 0.5000 | 0.5036 |

These are observations for this auto-generated Silver self-test, not a claim
that GraphRAG is generally superior to Vector RAG. Hybrid Temporal did not
exceed Graph on this snapshot, but that result does not establish that the
temporal module is ineffective: the Silver set contains only two
`temporal_metric` questions, its expected pages are graph-derived, and no
independently reviewed temporal gold set is available.

The targeted follow-up is a matched, human-reviewed temporal ablation: add
cross-filing questions with independently verified valid-time/recorded-time
evidence and narrative states, hold the query set and candidate budget fixed,
compare Hybrid versus Hybrid Temporal on temporal and non-temporal strata, and
report temporal-field/path-hit diagnostics separately. Regenerate the derived
temporal artifacts from the current graph before that experiment, and use a
versioned extraction cache or frozen artifact so external-LLM variation is not
confounded with the retrieval comparison.

### Extraction repeatability gate

`python -m strategic_graphrag.pipeline.pipeline --pdf data/pdfs/2025-10-K.pdf
--require_llm --dry_run --output_stats reports/rebuild_2025_repeatability.json`
performs an extraction-only replay: it parses the same PDF, uses the configured
provider and prompt, and deliberately skips Neo4j writes and post-processing.
The current corpus manifest is
`reports/corpus_manifest_2026-09-15.json`; the older
`reports/2026-08-14_corpus_manifest.json` is historical and is not the current
inventory. The readiness audit compares the document hash, model, prompt version,
temperature, and extracted-claim count. In the 2026-09-09 replay, the frozen
run produced 126 accepted claims and the second external-LLM call produced 131
at temperature 0.0. This 126-versus-131 observation remains historical context
about fresh external calls; it is not superseded by replay.

The versioned `llm_response_cache_v1` cache is now implemented. The 2025
2026-09-14 record/replay gate passed: both runs produced 126 accepted claims
with 170 unique keys; record made 170 LLM network calls and replay made zero.
This supports deterministic replay of the frozen response artifact only. It
does not establish fresh external-model repeatability, semantic correctness, or
human Golden QA.

### Versioned LLM response freeze cache

The extraction response cache fixes the input-to-LLM-output boundary for an
extraction run. It is an engineering reproducibility aid, not a new independent
model experiment: replaying a cached response does not measure a fresh model
call, provider variability, latency, or cost. The cache is also not a human
Golden QA set and does not establish semantic correctness.

The cache is controlled only through environment variables, so `.env` is not
modified:

```powershell
$env:LLM_RESPONSE_CACHE_PATH = "evaluation/cache/llm_extraction_v1.jsonl"
$env:LLM_RESPONSE_CACHE_MODE = "record"
python -m strategic_graphrag.pipeline.pipeline `
  --pdf data/pdfs/2025-10-K.pdf --require_llm --dry_run `
  --output_stats reports/rebuild_2025_cache_record_2026-09-14.json

$env:LLM_RESPONSE_CACHE_MODE = "replay"
python -m strategic_graphrag.pipeline.pipeline `
  --pdf data/pdfs/2025-10-K.pdf --require_llm --dry_run `
  --output_stats reports/rebuild_2025_cache_replay_2026-09-14.json
```

`off` is the default and does not inspect or write the cache. `record` calls
the configured provider and its existing explicit fallback sequence on a miss,
then appends successful structured JSON responses without overwriting an
existing key. `replay` is read-only and fails closed on a miss or malformed
record; it never calls an external provider. Each key binds the operation,
request provider and model, extraction temperature, max tokens, the SHA-256 of
the complete role-separated prompt, and the cache schema version. The JSONL
record stores key metadata, actual successful route metadata, structured JSON
response, response digest, and creation time, but never a separate raw
prompt/PDF payload or API keys. Structured response fields such as the model's
evidence quote remain part of the frozen response because they are required for
the same downstream provenance filtering during replay.

Every extraction report must record the cache path, schema, mode, and unique key
count, together with cache hits, misses, writes, and LLM network calls. A
record/replay comparison should also verify identical `document_sha256`,
`triples_extracted`, and cache key count; replay network calls must be zero.

## Answer-level scoring

Use a fixed 1--5 rubric with examples for:

- faithfulness: every factual claim is supported by returned evidence;
- relevance: the answer directly addresses the question;
- completeness: all required parts and requested years are covered;
- citation correctness: cited claim ID/page actually supports the sentence;
- abstention: the system refuses or states insufficiency when gold says the
  question is unanswerable.

Citation correctness and abstention should have deterministic checks in
addition to human scores. An LLM judge is a supplementary rater, not the gold
label; record its model, prompt version, temperature, and raw justifications.

## Agent extension

Add an agent only after the fixed four-mode baseline is complete. The agent may
decompose a question and call read-only retrieval tools, but it must not write
the graph or labels. Compare agent and non-agent systems on answer quality,
tool-call accuracy, unnecessary calls, latency, cost, and abstention accuracy.
