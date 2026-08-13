# -*- coding: utf-8 -*-
"""Build a filing-scoped Chroma index for one or more selected filings.

This script accepts one or more explicitly selected PDFs.  The metadata
contract (``source_filing`` + ``page``) is required by Hybrid Retrieval so
semantic chunks can be joined back to graph evidence without cross-filing
leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import chromadb
import fitz
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategic_graphrag.pipeline.text_splitter import RecursiveTextSplitter


def clean_text(text: str) -> str:
    text = re.sub(r"-\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_page(text: str, chunk_size: int = 1200, overlap: int = 250) -> List[str]:
    if not text:
        return []
    splitter = RecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def build_index(pdf_path: Path, db_path: Path, collection_name: str) -> Dict[str, int | str]:
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Single PDF not found: {pdf_path}")

    source_filing = pdf_path.name
    documents: List[str] = []
    metadatas: List[Dict[str, int | str]] = []
    ids: List[str] = []

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            page_text = clean_text(page.get_text("text"))
            for chunk_index, chunk in enumerate(chunk_page(page_text)):
                chunk_id = f"{source_filing}:{page_index}:{chunk_index}"
                documents.append(chunk)
                metadatas.append({
                    "source_filing": source_filing,
                    "doc_id": source_filing,
                    "page": page_index,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                })
                ids.append(chunk_id)

    if not documents:
        raise ValueError(f"No text chunks extracted from {pdf_path}")

    embedding_model = os.getenv("GRAPH_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(db_path))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {
        "source_filing": source_filing,
        "pages": len(set(item["page"] for item in metadatas)),
        "chunks": len(documents),
        "collection": collection_name,
        "collection_count": collection.count(),
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        dest="pdfs",
        type=Path,
        action="append",
        default=None,
        help="PDF to index; repeat for selected filings only",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/chroma_db"),
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("GRAPH_VECTOR_COLLECTION", "nvidia_sec_filings_active"),
    )
    parser.add_argument(
        "--replace-collection",
        action="store_true",
        help="Delete and recreate the named collection before indexing the selected PDFs",
    )
    args = parser.parse_args()
    pdfs = args.pdfs or [Path("data/pdfs/2025-10-K.pdf")]
    if args.replace_collection:
        client = chromadb.PersistentClient(path=str(args.db_path))
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    results = [
        build_index(pdf, args.db_path, args.collection)
        for pdf in pdfs
    ]
    print(json.dumps({"files": results, "collection": args.collection}, indent=2))


if __name__ == "__main__":
    main()
