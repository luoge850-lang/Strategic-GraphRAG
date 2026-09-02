# Data policy

The current experiment uses three allowlisted NVIDIA SEC 10-K PDFs under
`data/pdfs/`, covering fiscal years 2023, 2024, and 2025. The graph and vector
index are evaluated against this frozen three-filing corpus, not a single-PDF
stabilization stage.

Raw PDFs, ChromaDB files, and historical evaluation outputs are intentionally
ignored by Git because they are large/generated artifacts. The versioned
corpus manifest at
`reports/2026-08-14_corpus_manifest.json` records each source URL,
accession/document identifier, fiscal year, SHA256 hash, document type, and
extraction scope. Update the manifest before changing the corpus.
