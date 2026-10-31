"""Rerank the complete multi-query HNSW pool by facet coverage."""

from collections.abc import Sequence
from dataclasses import replace

from arxiv_paper_recommend.common.validation import (
	normalize_queries,
	validate_integer,
	validate_text,
	validate_weight,
)

from .._ranking import rank_ids
from .._scoring import DenseScoreCache
from ..constants import DEFAULT_RESULT_LIMIT
from ..dense import MultiQueryHNSWRetriever
from ..search_result import SearchResult
from ._facet_coverage import (
	DEFAULT_CANDIDATE_LIMIT,
	DEFAULT_COVERAGE_WEIGHT,
	raw_coverage_scores,
)


class FCHNSWRetriever(MultiQueryHNSWRetriever):
	"""Use one raw-similarity FC formula for search, recall, and ID scoring."""

	def search(
		self,
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		recall_queries: Sequence[str] | None = None,
		candidate_limit: int | None = None,
		coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
	) -> list[SearchResult]:
		"""Recall per facet, score the full union, and return at most limit results.

		Candidate depth defaults to max(50, limit), without a hybrid output cap.
		None uses the main query as the only facet; an empty sequence returns [].
		FC blends raw query similarity with geometric mean nonnegative facets.
		Ties are resolved by ascending document ID; unavailable scores go last.
		"""
		self._ensure_open()
		validate_text(query, "Query")
		validate_integer(limit, "Result limit")
		if candidate_limit is None:
			candidate_limit = max(DEFAULT_CANDIDATE_LIMIT, limit)
		validate_integer(candidate_limit, "Candidate limit", minimum=limit)
		return self.recall(
			query,
			limit=candidate_limit,
			recall_queries=recall_queries,
			coverage_weight=coverage_weight,
		)[:limit]

	def score(
		self,
		query: str,
		document_ids: list[int],
		*,
		recall_queries: Sequence[str] | None = None,
		coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
	) -> dict[int, float | None]:
		"""Score IDs with the same pool-independent FC formula as search and recall.

		None uses query as the only facet; an empty sequence uses raw query
		scores only. Missing IDs return None, in first-occurrence ID order.
		"""
		self._ensure_open()
		validate_text(query, "Query")
		facets = (
			[query] if recall_queries is None else normalize_queries(recall_queries)
		)
		coverage_weight = validate_weight(coverage_weight, "Coverage weight")
		return raw_coverage_scores(
			document_ids, query, facets, coverage_weight, DenseScoreCache(self)
		)

	def recall(
		self,
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		recall_queries: Sequence[str] | None = None,
		coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
	) -> list[SearchResult]:
		"""Return the complete recall union sorted and scored by raw-similarity FC.

		Each facet contributes at most limit neighbors; the union is not truncated.
		None uses query as the only facet; an empty sequence returns no candidates.
		"""
		self._ensure_open()
		validate_text(query, "Query")
		validate_integer(limit, "Result limit")
		facets = (
			[query] if recall_queries is None else normalize_queries(recall_queries)
		)
		coverage_weight = validate_weight(coverage_weight, "Coverage weight")

		results = self.raw_recall(query, limit=limit, recall_queries=facets)
		if not results:
			return []
		candidates = {result.document_id: result for result in results}
		cache = DenseScoreCache(
			self,
			{query: {result.document_id: result.score for result in results}},
		)
		scores = raw_coverage_scores(
			list(candidates), query, facets, coverage_weight, cache
		)
		return [
			replace(candidates[document_id], score=scores[document_id])
			for document_id in rank_ids(candidates, scores, descending=True)
		]
