# Data policy

The development experiment uses three allowlisted NVIDIA SEC 10-K PDFs for
fiscal 2023, 2024, and 2025. The active graph and vector index are evaluated
against this frozen three-filing corpus. Other files under `data/pdfs_other/`
are historical or exploratory material and are not part of the active corpus.

Raw PDFs, ChromaDB files, Neo4j data, extracted text, and model response caches
are intentionally ignored by Git because they are large or environment-bound.
The local corpus manifest records filename, byte size, SHA-256, page scope,
claim inventory, embedding configuration, and build metadata. The public
summary and its limits are in [`docs/claim_ledger.md`](../docs/claim_ledger.md)
and [`docs/public_results_2026-09-22.md`](../docs/public_results_2026-09-22.md).

To reproduce the corpus boundary, obtain the filings from SEC EDGAR or the
official company filing pages, rename them to the allowlisted filenames, and
run the manifest/audit scripts before indexing. Do not treat a filename or a
Neo4j database instance ID as proof that the PDF is the same; compare hashes.

The repository does not redistribute the PDFs or claim ownership of SEC and
third-party data. Check the source terms before redistribution.
