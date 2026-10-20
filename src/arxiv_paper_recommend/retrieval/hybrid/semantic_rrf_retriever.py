"""Weight two-list RRF by each candidate's semantic similarity."""

from dataclasses import replace

from .._ranking import rank_ids, rrf_scores
from ..search_result import SearchResult
from .base import HybridRetriever
from .constants import DEFAULT_RESULT_LIMIT


class SemanticRRFRetriever(HybridRetriever):
	"""Recall half as many sparse candidates, then weight each paper's RRF."""

	def search(
		self,
		terms: list[str],
		queries: str | set[str],
		score_query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
	) -> list[SearchResult]:
		"""Multiply each complete RRF score by cosine + 1; None gets weight zero."""
		query_texts = self._validate_inputs(queries, score_query, limit)
		sparse_limit = (limit + 1) // 2
		sparse_results = self.sparse_retriever.search(terms, limit=sparse_limit)
		sparse_candidates = {result.id: result for result in sparse_results}
		sparse_scores = {result.id: result.score for result in sparse_results}
		sparse_ids = rank_ids(sparse_candidates, sparse_scores)[:sparse_limit]
		dense_candidates, cache = self._dense_candidates(query_texts, limit)
		if not sparse_candidates and not dense_candidates:
			return []

		cache.ensure({score_query: [*dense_candidates, *sparse_ids]})
		dense_scores = cache.scores[score_query]
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
