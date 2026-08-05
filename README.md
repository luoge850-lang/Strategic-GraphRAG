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

For the frontend:

```powershell
cd frontend
npm install
npm run build
```

The API serves `frontend/dist/index.html` when the Vite build exists and falls
back to `frontend/public/index.html` during development.

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
page alignment, relation IDs, years, and ontology validation.

## Limitations of this freeze

- The current benchmark and legacy experiment artifacts are not paper results.
- Neo4j integration must be verified against the active database instance.
- The one-PDF stabilization phase precedes the planned 12-document ingestion.
- Agent planning and reflection are future work, not part of this version.
