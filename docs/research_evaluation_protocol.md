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
