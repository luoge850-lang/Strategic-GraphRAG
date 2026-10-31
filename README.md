# Strategic-GraphRAG

Maintenance branch `codex/quality-closure`: response validation and accounting aliases are under verification. Local services are stopped. The build below is the last accepted snapshot, not a build of the modified working sources. Rebuild/live acceptance is blocked by automatic rejection of removing the superseded `.local/build` and `.local/neo4j` directories.

Last accepted frozen candidate (not compatible with modified working sources): **`build_3b918db3d1355d09`**. NVIDIA FY2023–2025 10-Ks and TSMC 2023–2024 Taiwan annual reports; five source PDFs, two companies. This is a financial-report research prototype.

The API supports company/period-specific financial facts, arithmetic and unit conversion, independently bound multi-item requests, evidence-backed disclosed dependencies and refusal of unsupported event-attributed losses. Financial answers use deterministic rules and calculation over typed observations. Optional model synthesis is a separate mechanism; it is disabled in the current local configuration and was not evaluated in this consolidation.

The desktop repository is the active development location. The canonical flow uses `scripts/manage.py`, `scripts/evaluate.py`, `data/pdfs/`, `data/corpus.json` and ignored `.local/` configuration/runtime. Two worktrees remain after Git removed the clean README and reproduction checkouts. The dirty delivery tree and dated entry points remain pending deletion because automatic execution review blocked deletion; they are not inputs to this flow. Consolidation and disk cleanup are therefore not complete.

## Install and run (Windows)

Requires Python 3.12, Node 22+, JDK 21 and Neo4j Community 5.26.0. Install these outside this repository. Keep the five original PDFs locally in `data/pdfs/`, with the filenames and SHA-256 values in [the corpus manifest](data/corpus.json). [Source provenance](data/sources.json) distinguishes official disclosure references from original download URLs. The three retained NVIDIA PDF download URLs were not recorded; a newly downloaded filing is not automatically byte-identical.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -c requirements-lock.txt
npm --prefix frontend ci
npm --prefix frontend run build
.\.venv\Scripts\python.exe scripts/manage.py init --neo4j-home "<Neo4j distribution>" --java-home "<JDK 21>"
.\.venv\Scripts\python.exe scripts/manage.py build
.\.venv\Scripts\python.exe scripts/manage.py db-start
.\.venv\Scripts\python.exe scripts/manage.py import
.\.venv\Scripts\python.exe scripts/manage.py serve
```

Open http://127.0.0.1:8000/. This binds to loopback. `init` refuses to overwrite configuration; `build` refuses to overwrite a current build; import requires an empty database. Fresh extraction reads the PDFs directly and does not consume an old candidate. ONNX MiniLM resources must be available in the normal model cache or downloadable on the first vector build. No LLM key is required.

Stop the API with Ctrl+C. `python scripts/manage.py stop` stops the recorded project-owned Neo4j console process tree after verifying launcher identity. Windows console-process termination is not a graceful database shutdown; restart recovery is checked locally.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/evaluate.py
.\.venv\Scripts\python.exe scripts/evaluate.py --url http://127.0.0.1:8000
```

Local full suite: **420 passed**, plus 7 subtests. Frontend installation/build and dependency restoration/check succeeded. The exposed 25-question regression recomputes **23/25 core correctness**, **9/25 complete fact/source/metadata joint correctness**, **1/25 false refusal**, **0/25 severe incorrect successful answers**; exact labeled citation matches **22/25**. These are AI/PDF diagnostic labels, with zero human reviews. R03 returns no requested numeric answer; R12 remains an incorrect refusal. This is not independent unseen validation.

Current build contents: **695 financial observations**, **3 directly disclosed dependency/risk records**, **1,914 vector chunks**; Neo4j **1,398 nodes / 5,389 relationships**, including provenance helpers. Auxiliary edges are not additional business knowledge.

[Current evaluation inputs](evaluation/current/README.md) · [File inventory](docs/FILE_INVENTORY.csv) · [Consolidation receipt](docs/CONSOLIDATION.md)
