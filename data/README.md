# Data policy

The stabilization phase uses one local NVIDIA SEC PDF under `data/pdfs/`.
Raw PDFs, ChromaDB files, and historical evaluation outputs are intentionally
ignored by Git because they are large/generated artifacts and are not part of
the v2 reproducibility contract yet.

Before the 12-document phase, add a versioned manifest containing each source
URL, accession/document identifier, fiscal year, SHA256 hash, document type,
and extraction scope.
