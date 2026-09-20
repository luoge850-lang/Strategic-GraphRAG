# External finance benchmark protocol — 2026-09-19

The project registers three public benchmark families without mixing their
tasks or scores:

- FinanceBench: open-book financial QA with human answers and evidence fields.
- FinQA: numerical reasoning over financial tables and text, including gold
  supporting facts and executable programs.
- TAT-QA: hybrid tabular/textual finance QA with numerical reasoning.

The files and SHA-256 hashes are recorded by
`scripts/prepare_external_benchmarks.py` in
`reports/external_benchmark_inventory_2026-09-19.json`.

The current NVIDIA index must not be evaluated on these questions as if it had
retrieved their source documents. The current implementation therefore reports
the external datasets as `REGISTERED_NOT_RUN_AGAINST_ACTIVE_NVIDIA_CORPUS`.
The next external-evaluation stage must create a separate document/index
version, run all four retrieval modes under the same protocol, and report each
dataset separately.

Required external metrics:

- FinanceBench: answer exact match/normalized answer match, citation precision,
  citation recall, citation completeness, faithfulness, abstention precision
  and abstention recall.
- FinQA: program/execution accuracy, numerical exact match, table-cell
  retrieval recall, and evidence alignment.
- TAT-QA: answer F1/EM, scale-aware numeric match, table-cell retrieval,
  operation/execution accuracy, and evidence alignment.

No single aggregate “financial QA accuracy” should combine these datasets.
