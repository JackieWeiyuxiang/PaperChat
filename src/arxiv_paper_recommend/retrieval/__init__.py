"""Retrieval interface, results, and a lazily imported research-request engine."""

from typing import TYPE_CHECKING

from .base import Retriever
from .search_result import SearchResult

if TYPE_CHECKING:
	from .search_engine import SearchEngine

__all__ = ["Retriever", "SearchEngine", "SearchResult"]


def __getattr__(name: str) -> object:
	if name == "SearchEngine":
		from .search_engine import SearchEngine

		globals()[name] = SearchEngine
		return SearchEngine
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
