"""Dense text retrieval backed by FAISS."""

from .divisional_hnsw import DivisionalHNSWRetriever
from .hnsw import HNSWRetriever

__all__ = ["DivisionalHNSWRetriever", "HNSWRetriever"]
