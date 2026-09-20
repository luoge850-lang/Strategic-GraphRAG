# External benchmark cache

`benchmarks/` is a local, ignored cache populated by
`scripts/prepare_external_benchmarks.py`. The machine-readable inventory and
SHA-256 hashes belong in `reports/`; the raw public datasets are not committed
to the project repository.

The current cache contains the open FinanceBench sample, FinQA public test
data, and TAT-QA development data. They are registered for separate external
evaluation only; they are not part of the active NVIDIA corpus and their scores
must not be mixed with the NVIDIA Silver or Human Gold reports.
