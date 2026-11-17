"""Rerank all BM25 candidates with the dense component's score."""

from collections.abc import Sequence
from dataclasses import replace
from math import ceil, sqrt

from arxiv_paper_recommend.common.cancellation import check_cancelled

from .._ranking import rank_ids
from ..constants import DEFAULT_RESULT_LIMIT
from ..innovation import FCHNSWRetriever
from ..search_result import SearchResult
from .base import HybridRetriever


class SDCascadeRetriever(HybridRetriever):
	"""Score BM25 candidates densely without a full-index dense search."""

	def search(
		self,
		terms: list[str],
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		title_weight: float | None = None,
		recall_queries: Sequence[str] | None = None,
	) -> list[SearchResult]:
		"""Recall ceil(sqrt(limit * corpus size)) candidates and rank dense scores.

		FC-HNSW uses the normalized facets; other dense scorers use query only.
		Unavailable scores are excluded; equal scores use ascending document ID.
		"""
		query_texts = self._validate_inputs(query, limit, recall_queries)
		check_cancelled()
		candidate_limit = max(
			1, ceil(sqrt(limit * self.sparse_retriever.count_documents()))
		)
		check_cancelled()
		sparse_results = self.sparse_retriever.search(
			terms,
			limit=candidate_limit,
			title_weight=title_weight,
		)
		check_cancelled()
		candidates = {result.document_id: result for result in sparse_results}
		if not candidates:
			return []

		document_ids = list(candidates)
		if isinstance(self.dense_retriever, FCHNSWRetriever):
			dense_scores = self.dense_retriever.score(
				query, document_ids, recall_queries=query_texts
			)
		else:
			dense_scores = self.dense_retriever.score(query, document_ids)
		check_cancelled()
		scored_ids = [
			document_id
			for document_id in document_ids
			if dense_scores[document_id] is not None
		]
		return [
			replace(candidates[document_id], score=dense_scores[document_id])
			for document_id in rank_ids(scored_ids, dense_scores, descending=True)[
				:limit
			]
		]
