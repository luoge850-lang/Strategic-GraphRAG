# Current evidence

`baseline.jsonl` contains all 25 original labels, raw responses and stored scores. R03 and R12 are retained failures. `score.py` is the unchanged layered scorer; `slot_score.py` preserves the mutation regressions. `sources.json` and `source-adjudication.json` retain the original PDF-based reference packet and AI review.

This is an exposed, known-family regression set, not a new independent test or human Gold. Human reviews: 0. Do not interpret the 23/25 core score as 23/25 complete facts: the strict metadata/source-location joint score is 9/25. Exact labeled citation location matches are 22/25; missing metadata is not silently promoted to verified source support.

Recompute original results: `python scripts/evaluate.py`.
Replay against the current local service: `python scripts/evaluate.py --url http://127.0.0.1:8000`.
Outputs go to ignored `.local/results/`. `multi-labels.json` and `boundary-labels.json` are exposed auxiliary regression references, not additional independent validation samples.
