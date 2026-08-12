# Strategic-GraphRAG v2

Evidence-grounded temporal causal GraphRAG for NVIDIA SEC filings.

This repository is the second implementation phase of the original
[Strategic-GraphRAG](https://github.com/luoge850-lang/Strategic-GraphRAG)
project. The current stabilization scope deliberately uses one NVIDIA SEC PDF
until ingestion, ontology, temporal filtering, evidence provenance, and the
interactive query path are reliable. Multi-document ingestion, evaluation, and
agentic planning are intentionally postponed.

## Current scope

```text
PDF → SEC section detection → hybrid triple extraction → Neo4j
    → temporal causal path retrieval → EvidenceClaim provenance
    → FastAPI → React/Vite dashboard
```

The main implementation is the `strategic_graphrag` package. The historical
`src/step1~7` scripts are retained for reference and are not the v2 execution
entrypoint.

## Stabilized single-PDF contract

The current active corpus is `2025-10-K.pdf`. `/query` defaults to the
`hybrid` mode: filing-scoped Chroma retrieval is fused with verified Neo4j
causal paths at the filing/page level. Use `retrieval_mode=graph` for the
graph-only ablation, or `/query/vector` for the independent vector baseline.

The runtime refuses unsupported temporal comparisons when the retrieved
evidence does not cover the requested fiscal years. Generated reports are
also checked against the exact EvidenceClaim IDs, pages, and years before they
are returned. A failed check is reported as `GROUNDING FAILURE`.

The current ontology is an evidence-grounded single-filing baseline. Direct
`RiskFactor -> FinancialMetric` impacts are labeled
`DIRECT_DISCLOSED_IMPACT`; mechanism-mediated reasoning is an extension target,
not a completed claim of this frozen version.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then initialize the Neo4j schema:

```powershell
python -m strategic_graphrag.schema.manager --init
python -m strategic_graphrag.pipeline.pipeline --pdf_dir data/pdfs
uvicorn strategic_graphrag.api.server:app --reload
```

The pipeline refuses to process more than one PDF by default. After the
single-filing graph and evidence contract has been reviewed, opt in explicitly
with `--allow_multiple_pdfs`.

To rebuild the same filing after changing extraction rules, add
`--replace_existing_filing`. It is disabled by default because it deletes only
that filing's evidence and relationships before ingestion; shared entity and
Year nodes are preserved.

For the frontend:

```powershell
cd frontend
npm install
npm run build
```

The API serves `frontend/dist/index.html` when the Vite build exists and falls
back to `frontend/public/index.html` during development.

Build the filing-scoped vector index before using the default hybrid mode:

```powershell
python scripts/build_single_pdf_vector_index.py
```

For a containerized run, configure `.env` and use:

```powershell
docker build -t strategic-graphrag .
docker run --env-file .env -p 8000:8000 strategic-graphrag
```

Set `API_AUTH_ENABLED=true` and a real `API_KEY` outside local development.
`CORS_ORIGINS`, `RATE_LIMIT_PER_MINUTE`, `GRAPH_ACTIVE_FILING`, and
`LLM_FALLBACK_PROVIDERS` are also deployment controls.

## Temporal and evidence contract

Every extracted graph edge should carry:

- `year`
- `source_filing`
- `source_page`
- `evidence_id`

Every evidence item is represented by an `EvidenceClaim` node connected to the
exact source sentence and both endpoint entities. LLM evidence is accepted
only when it is a verbatim normalized span of the input text.

The API exposes `/query`, `/evidence/{entity_id}`, and
`/graph/temporal/{risk_id}` for manual end-to-end checks against the active
Neo4j instance.

Run the read-only single-filing contract check with:

```powershell
python scripts/validate_single_pdf_kg.py --doc_id 2025-10-K --filename 2025-10-K.pdf
```

The check verifies EvidenceClaim links, source/target entities, exact text and
page alignment, relation IDs, years, verbatim verification status, and ontology
validation. It also reports the source/target category matrix for each
relation, which is useful for reviewing extraction errors before adding more
filings.

Path results expose a stable `fingerprint` derived from the ordered nodes,
relations, years, pages, and EvidenceClaim IDs. This makes later ranking and
ablation experiments reproducible even when two paths have the same score.

## Limitations of this freeze

- The current benchmark and legacy experiment artifacts are not paper results.
- `data/evaluation/golden_qa_v2.jsonl` is an automatically generated regression
  candidate set, not a human-confirmed Golden Dataset. It must be reviewed,
  deduplicated, and expanded with realistic negative questions before academic
  claims are made.
- `reports/golden_qa_v2_results.json` reports structural retrieval metrics only;
  LLM-judge Faithfulness and Answer Relevance were not enabled in this run.
- The current single filing cannot establish cross-year temporal trends.
- Neo4j integration must be verified against the active database instance.
- The one-PDF stabilization phase precedes the planned 12-document ingestion.
- Agent planning and reflection are future work, not part of this version.
