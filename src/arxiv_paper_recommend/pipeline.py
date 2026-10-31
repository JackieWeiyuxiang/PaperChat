"""Run SD Cascade retrieval and evidence-grounded recommendation generation."""

from dataclasses import asdict

from .common.lifecycle import ManagedResource
from .common.validation import validate_integer, validate_text
from .recommendation.base import MAX_GENERATION_CANDIDATES, MAX_RECOMMENDATIONS
from .recommendation.innovation import EvidenceGroundedRecommendationGenerator
from .retrieval import SearchEngine
from .retrieval.constants import (
	DEFAULT_MAX_QUERIES,
	DEFAULT_MAX_TERMS,
	MAX_TERM_COUNT,
	MIN_GENERATED_TERM_COUNT,
)


class PaperRecommender(ManagedResource):
	"""Reuse lazy components; injected engines and generators remain caller-owned."""

	def __init__(
		self,
		*,
		search_engine: SearchEngine | None = None,
		generator: EvidenceGroundedRecommendationGenerator | None = None,
	) -> None:
		super().__init__()
		self.search_engine = search_engine
		self.generator = generator

	def recommend(
		self,
		request: str,
		*,
		candidate_limit: int = MAX_GENERATION_CANDIDATES,
		limit: int = MAX_RECOMMENDATIONS,
		max_terms: int = DEFAULT_MAX_TERMS,
		max_queries: int = DEFAULT_MAX_QUERIES,
	) -> dict[str, object]:
		"""Return the normalized request and ranked recommendations as plain data.

		Validate all inputs before creating resources or calling either stage.
		An empty retrieval skips generator creation. Stage errors propagate without
		returning partial recommendations; citation markers remain unchanged.
		"""
		self._ensure_open()
		request = validate_text(request, "Research request", strip=True)
		validate_integer(
			candidate_limit, "Candidate limit", maximum=MAX_GENERATION_CANDIDATES
		)
		validate_integer(limit, "Recommendation limit", maximum=MAX_RECOMMENDATIONS)
		validate_integer(
			max_terms,
			"max_terms",
			minimum=MIN_GENERATED_TERM_COUNT,
			maximum=MAX_TERM_COUNT,
		)
		validate_integer(max_queries, "max_queries")

		papers = self._get_search_engine().search(
			request,
			method="SD_CASCADE",
			limit=candidate_limit,
			max_terms=max_terms,
			max_queries=max_queries,
		)
		if not papers:
			return {"request": request, "recommendations": []}

		recommendations = self._get_generator().generate(request, papers, limit=limit)
		return {
			"request": request,
			"recommendations": [asdict(item) for item in recommendations],
		}

	def _get_search_engine(self) -> SearchEngine:
		if self.search_engine is None:
			self.search_engine = SearchEngine()
			self._resources.callback(self.search_engine.close)
		return self.search_engine

	def _get_generator(self) -> EvidenceGroundedRecommendationGenerator:
		if self.generator is None:
			self.generator = EvidenceGroundedRecommendationGenerator()
			self._resources.callback(self.generator.close)
		return self.generator
