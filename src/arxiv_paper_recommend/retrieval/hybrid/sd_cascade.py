"""Use BM25 candidates for multi-query dense selection and reranking."""

from dataclasses import replace

from .._ranking import rank_ids
from ..search_result import SearchResult
from .base import DenseScoreCache, HybridRetriever
from .constants import DEFAULT_RESULT_LIMIT, SPARSE_CANDIDATE_MULTIPLIER


class SDCascadeRetriever(HybridRetriever):
	"""Search densely within BM25 candidates without a full-index dense search."""

	def search(
		self,
		terms: list[str],
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		title_weight: float | None = None,
		divisional_queries: str | set[str] | None = None,
	) -> list[SearchResult]:
		"""Recall limit * 60 sparse candidates, select per query, then rerank."""
		query_texts = self._validate_inputs(query, limit, divisional_queries)
		sparse_results = self.sparse_retriever.search(
			terms,
			limit=limit * SPARSE_CANDIDATE_MULTIPLIER,
			title_weight=title_weight,
		)
		candidates = {result.id: result for result in sparse_results}
		if not candidates:
			return []

		cache = DenseScoreCache(self.dense_retriever)
		cache.ensure({division: list(candidates) for division in query_texts})
		selected_ids: dict[int, None] = {}
		for division in query_texts:
			scores = cache.scores[division]
			scored_ids = [
				document_id
				for document_id in candidates
				if scores[document_id] is not None
			]
			for document_id in rank_ids(scored_ids, scores, descending=True)[:limit]:
				selected_ids.setdefault(document_id, None)
		if not selected_ids:
			return []

		cache.ensure({query: list(selected_ids)})
		dense_scores = cache.scores[query]
		return [
			replace(candidates[document_id], score=dense_scores[document_id])
			for document_id in rank_ids(selected_ids, dense_scores, descending=True)[
				:limit
			]
		]
