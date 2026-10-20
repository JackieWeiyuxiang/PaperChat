"""Dense text retrieval backed by FAISS."""

from .hnsw import HNSWRetriever
from .multi_query_hnsw import MultiQueryHNSWRetriever

__all__ = ["HNSWRetriever", "MultiQueryHNSWRetriever"]
