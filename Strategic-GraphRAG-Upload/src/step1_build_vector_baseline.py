"""
Module: step1_build_vector_baseline.py
Project: Strategic GraphRAG Analysis for SEC Filings (NUS Research)
-----------------------------------------------------------------------
Description:
    This module orchestrates the construction of a high-fidelity Vector 
    Database (ChromaDB) to serve as the baseline for ablation studies.
    
    It implements a symmetric data pipeline ensuring that text extraction, 
    heuristic cleaning, and recursive chunking parameters remain identical 
    to the GraphRAG ingestion phase to maintain academic rigor.

Technical Stack:
    - Text Extraction: PyMuPDF (fitz)
    - Vector Store: ChromaDB (Persistent)
    - Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)

Author: Louis Harrington
Date: 2026-03-24 (Optimized Version)
"""

import os
import re
import logging
import argparse
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. System Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("VectorBaselineBuilder")


# ==========================================
# 2. Core Logic: VectorDBBuilder
# ==========================================
class VectorDBBuilder:
    """
    Handles the end-to-end pipeline from raw PDF ingestion to 
    embedded vector storage in ChromaDB.
    """

    def __init__(
        self, 
        raw_data_dir: Path, 
        db_dir: Path, 
        collection_name: str = "nvidia_sec_filings",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1200, 
        chunk_overlap: int = 250
    ) -> None:
        """
        Initializes the builder with directory paths and chunking hyperparameters.

        Args:
            raw_data_dir (Path): Source directory for PDF artifacts.
            db_dir (Path): Destination directory for the persistent database.
            collection_name (str): Unique identifier for the ChromaDB collection.
            embedding_model (str): Transformer model for semantic encoding.
            chunk_size (int): Maximum character length per chunk.
            chunk_overlap (int): Overlapping window between adjacent chunks.
        """
        self.raw_data_dir = raw_data_dir
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Academic standard text splitting heuristics
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _clean_corpus(self, text: str) -> str:
        """
        Normalizes text by removing administrative noise and redundant whitespace.
        
        Args:
            text (str): Raw character stream from PDF.
            
        Returns:
            str: Normalized strategic text.
        """
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'-\n\s*', '', text)  # Handle hyphenated line breaks
        return text.strip()

    def process_pdfs_to_chunks(self) -> List[str]:
        """
        Performs batch processing of PDF files into normalized semantic chunks.
        
        Returns:
            List[str]: Aggregated list of text segments ready for embedding.
            
        Raises:
            FileNotFoundError: If the source directory is missing.
        """
        if not self.raw_data_dir.exists():
            logger.error(f"IO Error: Raw data directory not found at {self.raw_data_dir}")
            raise FileNotFoundError(f"Missing directory: {self.raw_data_dir}")

        pdf_files = list(self.raw_data_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF artifacts detected in {self.raw_data_dir}.")
            return []

        logger.info(f"Discovered {len(pdf_files)} PDF(s). Initializing extraction pipeline...")
        all_chunks: List[str] = []
        
        for pdf_path in pdf_files:
            logger.info(f"Processing Semantic Layer: {pdf_path.name}")
            try:
                # Page-level extraction
                doc = fitz.open(str(pdf_path))
                content = "".join([page.get_text() for page in doc])
                
                # Heuristic normalization
                cleaned_text = self._clean_corpus(content)
                
                # Recursive Chunking
                docs = self.text_splitter.create_documents([cleaned_text])
                extracted_chunks = [chunk.page_content for chunk in docs]
                all_chunks.extend(extracted_chunks)
                
                logger.info(f" -> Successfully mapped {len(extracted_chunks)} chunks.")
                
            except Exception as e:
                logger.error(f"Extraction Failure for {pdf_path.name}: {e}", exc_info=False)

        logger.info(f"Total aggregated semantic units: {len(all_chunks)}")
        return all_chunks

    def populate_database(self, documents: List[str]) -> None:
        """
        Computes dense embeddings and batch-populates the ChromaDB vector store.
        
        Args:
            documents (List[str]): List of processed text segments.
        """
        if not documents:
            logger.warning("Empty document set provided. Ingestion aborted.")
            return

        logger.info("Initializing ChromaDB Persistent Client (Compute: CPU/Local)...")
        client = chromadb.PersistentClient(path=str(self.db_dir))
        
        logger.info(f"Loading Neural Encoder: {self.embedding_model}")
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        collection = client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=embedding_fn
        )
        
        # Generating deterministic IDs for baseline consistency
        chunk_ids = [f"baseline_chunk_{i:06d}" for i in range(len(documents))]
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        logger.info(f"Executing Batch Ingestion: {total_batches} steps total...")
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = chunk_ids[i : i + batch_size]
            
            collection.add(documents=batch_docs, ids=batch_ids)
            
            # Periodic logging for long-running ingestion
            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(documents):
                logger.info(f"Ingestion Progress: {((i // batch_size) + 1) / total_batches:.1%}")

        logger.info(f"✅ Baseline System Ready: Vector DB populated with {len(documents)} units.")


# ==========================================
# 3. Execution Logic
# ==========================================
def parse_arguments() -> argparse.Namespace:
    """Configures command-line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Strategic GraphRAG: Baseline Vector DB Construction."
    )
    
    # Path resolution relative to project root
    root_dir = Path(__file__).resolve().parent.parent
    default_raw_dir = root_dir / "data" / "raw"
    default_db_dir = root_dir / "data" / "chroma_db"
    
    parser.add_argument(
        "--raw_dir", 
        type=str, 
        default=str(default_raw_dir),
        help="Input directory for SEC 10-K PDFs."
    )
    parser.add_argument(
        "--db_dir", 
        type=str, 
        default=str(default_db_dir),
        help="Output directory for ChromaDB vector persistence."
    )
    
    return parser.parse_args()


def main() -> None:
    """Orchestration entry point for the Vector DB pipeline."""
    args = parse_arguments()
    logger.info("--- Starting Baseline Vector Construction ---")
    
    builder = VectorDBBuilder(
        raw_data_dir=Path(args.raw_dir),
        db_dir=Path(args.db_dir)
    )
    
    try:
        chunks = builder.process_pdfs_to_chunks()
        builder.populate_database(chunks)
        logger.info("--- Baseline Construction Terminated Successfully ---")
    except Exception as e:
        logger.critical(f"Pipeline Fatality: {e}", exc_info=True)


if __name__ == "__main__":
    main()