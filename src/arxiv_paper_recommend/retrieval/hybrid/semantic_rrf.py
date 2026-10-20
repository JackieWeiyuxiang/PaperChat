"""Weight two-list RRF by each candidate's semantic similarity."""

from collections.abc import Sequence
from dataclasses import replace

from .._ranking import rank_ids, rrf_scores
from ..constants import DEFAULT_RESULT_LIMIT
from ..search_result import SearchResult
from .base import HybridRetriever


class SemanticRRFRetriever(HybridRetriever):
	"""Recall half as many sparse candidates, then weight each paper's RRF."""

	def search(
		self,
		terms: list[str],
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		title_weight: float | None = None,
		recall_queries: Sequence[str] | None = None,
	) -> list[SearchResult]:
		"""Multiply each complete RRF score by cosine + 1; None gets weight zero."""
		self._validate_inputs(query, limit, recall_queries)
		sparse_limit = (limit + 1) // 2
		sparse_results = self.sparse_retriever.search(
			terms, limit=sparse_limit, title_weight=title_weight
		)
		sparse_candidates = {result.document_id: result for result in sparse_results}
		sparse_scores = {result.document_id: result.score for result in sparse_results}
		sparse_ids = rank_ids(sparse_candidates, sparse_scores)[:sparse_limit]
		dense_candidates, cache = self._dense_candidates(
			query, limit, recall_queries=recall_queries
		)
		if not sparse_candidates and not dense_candidates:
			return []

		cache.ensure({query: [*dense_candidates, *sparse_ids]})
		dense_scores = cache.scores[query]
		dense_ids = rank_ids(dense_candidates, dense_scores, descending=True)[:limit]
		scores = rrf_scores((sparse_ids, dense_ids))
		for document_id, score in scores.items():
			similarity = dense_scores[document_id]
			scores[document_id] = score * (
				0.0 if similarity is None else similarity + 1.0
			)

		candidates = {**dense_candidates, **sparse_candidates}
		return [
			replace(candidates[document_id], score=scores[document_id])
			for document_id in rank_ids(scores, scores, descending=True)[:limit]
		]
