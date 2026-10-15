"""Retrieval results and a lazily imported research-request search engine."""

from typing import TYPE_CHECKING

from .search_result import SearchResult

if TYPE_CHECKING:
    from .search_engine import SearchEngine

__all__ = ["SearchEngine", "SearchResult"]


def __getattr__(name: str) -> object:
    if name == "SearchEngine":
        from .search_engine import SearchEngine

        globals()[name] = SearchEngine
        return SearchEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
