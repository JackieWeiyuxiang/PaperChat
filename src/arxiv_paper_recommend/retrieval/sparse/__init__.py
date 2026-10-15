"""Sparse keyword retrieval backed by SQLite FTS5."""

from .bm25_retriever import BM25Retriever

__all__ = ["BM25Retriever"]
