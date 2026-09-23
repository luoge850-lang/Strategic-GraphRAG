# External model and API decision record

This document is a configuration record, not a vendor comparison or a pricing
claim. Provider pricing, quotas, context limits, retention policies, and model
capabilities change over time and are not independently benchmarked by this
repository.

## Current development configuration

The development snapshot recorded in the corpus manifest uses DeepSeek as the
extraction, query, and report provider, with the model name and temperatures
captured in the run configuration. The exact installed dependency versions are
in `requirements-lock-2026-09-19.txt`; secrets and endpoint values stay in the
local `.env` file.

The project treats the following as separate configuration choices:

| Function | What must be frozen for a comparable run |
|---|---|
| Extraction | Provider, model identifier/snapshot, prompt version, temperature, structured-output schema, retry policy, and response-cache mode |
| Query planning | Provider/model, prompt version, temperature, query-plan schema, and whether remote anchor expansion is enabled |
| Answer synthesis | Provider/model, prompt version, temperature, evidence context, citation contract, and abstention policy |
| Embedding | Backend, model, chunking configuration, collection name, and index build identity |

## Reproducibility boundary

Temperature `0.0` reduces sampling randomness but does not prove that a hosted
model service will return byte-identical fresh responses. The project therefore
reports fresh external calls separately from versioned cache record/replay. A
cache replay pass demonstrates deterministic replay of the recorded artifact;
it does not demonstrate fresh-model repeatability or semantic correctness.

## Evaluation policy

All four retrieval modes must use the same question set, corpus snapshot,
top-k, prompt/model settings, and evaluation script. Synthesis runs are
reported separately from retrieval-only runs. A vendor benchmark, a claimed
free tier, or a model marketing description is not a result for this project.

For external benchmarks or a provider change, record the new model and
configuration as a new build identity and rerun the relevant quality and
latency checks. Do not overwrite historical reports or retroactively change
the claim ledger.

## Data handling

The active corpus consists of public SEC filings, but public availability does
not remove the need to document provider terms, retention, credentials, and
failure behavior. The application must not send secrets or local environment
files to a model provider. Production deployment would require a separate
security and data-governance review; this local Demo does not constitute that
review.

## Historical notes

Earlier versions of this file contained unverified comparisons of prices,
quotas, throughput, context windows, privacy policies, and extraction quality.
Those statements are intentionally removed from the current public decision
record. If a future experiment needs such a comparison, it must cite a dated
primary source and record the exact experiment rather than copy a stale table.
