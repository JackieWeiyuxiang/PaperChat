"""Retrieval-augmented recommendations over arXiv paper titles and abstracts."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .pipeline import PaperRecommender

__all__ = ["PaperRecommender"]


def __getattr__(name: str) -> object:
	if name == "PaperRecommender":
		from .pipeline import PaperRecommender

		globals()[name] = PaperRecommender
		return PaperRecommender
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
