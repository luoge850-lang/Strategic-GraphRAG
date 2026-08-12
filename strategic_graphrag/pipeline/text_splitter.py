"""Small, dependency-tolerant text splitter used by the single-filing pipeline.

LangChain's splitter is preferred when installed.  The fallback keeps the
project runnable in lightweight audit environments and follows the same basic
contract: prefer paragraph/sentence boundaries, then apply overlap.
"""

from __future__ import annotations

import re
from typing import List, Sequence

try:  # pragma: no cover - exercised when the optional dependency is present
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _LangChainSplitter
except ImportError:  # pragma: no cover - fallback is covered by smoke tests
    _LangChainSplitter = None


class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int = 2400,
        chunk_overlap: int = 300,
        separators: Sequence[str] | None = None,
    ) -> None:
        self.chunk_size = max(int(chunk_size), 100)
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size - 1))
        self.separators = list(separators or ["\n\n", "\n", ". ", " ", ""])
        self._delegate = (
            _LangChainSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
            )
            if _LangChainSplitter
            else None
        )

    def split_text(self, text: str) -> List[str]:
        if self._delegate is not None:
            return self._delegate.split_text(text)
        text = str(text or "").strip()
        if not text:
            return []
        pieces = [piece.strip() for piece in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if piece.strip()]
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if len(piece) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                step = self.chunk_size - self.chunk_overlap
                for start in range(0, len(piece), max(step, 1)):
                    part = piece[start : start + self.chunk_size].strip()
                    if part:
                        chunks.append(part)
                continue
            candidate = f"{current} {piece}".strip() if current else piece
            if current and len(candidate) > self.chunk_size:
                chunks.append(current)
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = f"{overlap} {piece}".strip()
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks
