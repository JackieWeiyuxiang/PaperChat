"""Rerank each candidate list with weighted RRF, then fuse their new rankings."""

from collections.abc import Collection, Sequence
from dataclasses import replace

from arxiv_paper_recommend.common.validation import validate_weight

from .._ranking import rank_ids, rrf_scores
from ..constants import DEFAULT_RESULT_LIMIT
from ..search_result import SearchResult
from .base import HybridRetriever


class HRRFRetriever(HybridRetriever):
	"""Keep each route's candidates, rerank each route, then fuse positions."""

	def search(
		self,
		terms: list[str],
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		title_weight: float | None = None,
		recall_queries: Sequence[str] | None = None,
		sparse_semantic_weight: float = 0.5,
		dense_semantic_weight: float = 0.5,
		fusion_weight: float = 0.5,
	) -> list[SearchResult]:
		"""Return weighted RRF scores, breaking ties by ascending document ID.

		The semantic weights control raw query similarity within each route;
		fusion_weight controls the dense route's share in the final fusion.
		Each route retains at most limit candidates, including at weight endpoints.
		"""
		self._validate_inputs(query, limit, recall_queries)
		sparse_semantic_weight = validate_weight(
			sparse_semantic_weight, "Sparse semantic weight"
		)
		dense_semantic_weight = validate_weight(
			dense_semantic_weight, "Dense semantic weight"
		)
		fusion_weight = validate_weight(fusion_weight, "Fusion weight")

		sparse_results = self.sparse_retriever.search(
			terms, limit=limit, title_weight=title_weight
		)
		sparse_candidates = {result.document_id: result for result in sparse_results}
		sparse_scores = {result.document_id: result.score for result in sparse_results}
		dense_candidates, cache = self._dense_candidates(
			query, limit, recall_queries=recall_queries
		)
		if not sparse_candidates and not dense_candidates:
			return []

		cache.ensure({query: [*dense_candidates, *sparse_candidates]})
		dense_scores = cache.scores[query]
		dense_ids = rank_ids(dense_candidates, dense_scores, descending=True)[:limit]
		missing_sparse_ids = [
			document_id for document_id in dense_ids if document_id not in sparse_scores
		]
		if missing_sparse_ids:
			sparse_scores.update(
				self.sparse_retriever.score(
					terms, missing_sparse_ids, title_weight=title_weight
				)
			)

		sparse_ids = _rerank(
			sparse_candidates, sparse_scores, dense_scores, sparse_semantic_weight
		)
		dense_ids = _rerank(
			dense_ids, sparse_scores, dense_scores, dense_semantic_weight
		)
		candidates = {
			document_id: dense_candidates[document_id] for document_id in dense_ids
		}
		candidates.update(sparse_candidates)
		scores = rrf_scores(
			(sparse_ids, dense_ids), weights=(1.0 - fusion_weight, fusion_weight)
		)
		return [
			replace(candidates[document_id], score=scores[document_id])
			for document_id in rank_ids(scores, scores, descending=True)[:limit]
		]


def _rerank(
	document_ids: Collection[int],
	sparse_scores: dict[int, float | None],
	dense_scores: dict[int, float | None],
	semantic_weight: float,
) -> list[int]:
	scores = rrf_scores(
		(
			rank_ids(document_ids, sparse_scores),
			rank_ids(document_ids, dense_scores, descending=True),
		),
		weights=(1.0 - semantic_weight, semantic_weight),
	)
	return rank_ids(scores, scores, descending=True)
