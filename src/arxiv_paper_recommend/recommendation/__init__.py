"""Recommendation models and a lazily imported generation API."""

from typing import TYPE_CHECKING

from .models import PaperRecommendation

if TYPE_CHECKING:
	from .generation import RecommendationGenerator

__all__ = ["PaperRecommendation", "RecommendationGenerator"]


def __getattr__(name: str) -> object:
	if name == "RecommendationGenerator":
		from .generation import RecommendationGenerator

		globals()[name] = RecommendationGenerator
		return RecommendationGenerator
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
