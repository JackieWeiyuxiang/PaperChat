"""Lazily exported retrieval innovations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .fc_hnsw import FCHNSWRetriever
	from .hprf import HPRFRetriever

__all__ = ["FCHNSWRetriever", "HPRFRetriever"]


def __getattr__(name: str) -> object:
	if name == "FCHNSWRetriever":
		from .fc_hnsw import FCHNSWRetriever

		globals()[name] = FCHNSWRetriever
		return FCHNSWRetriever
	if name == "HPRFRetriever":
		from .hprf import HPRFRetriever

		globals()[name] = HPRFRetriever
		return HPRFRetriever
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
