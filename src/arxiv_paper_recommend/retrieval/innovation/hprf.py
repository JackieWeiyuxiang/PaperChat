"""Rerank each candidate route, then fuse weighted rank percentiles."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace

from arxiv_paper_recommend.common.validation import validate_weight

from .._ranking import rank_ids, rank_percentiles
from ..constants import DEFAULT_RESULT_LIMIT
from ..hybrid.base import HybridRetriever
from ..search_result import SearchResult


class HPRFRetriever(HybridRetriever):
	"""Keep HRRF's candidate pools and replace its three fusions with percentiles."""

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
		"""Return hierarchical percentile scores, breaking ties by ascending ID.

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

		sparse_percentiles = _rerank_percentiles(
			sparse_candidates, sparse_scores, dense_scores, sparse_semantic_weight
		)
		dense_percentiles = _rerank_percentiles(
			dense_ids, sparse_scores, dense_scores, dense_semantic_weight
		)
		candidates = {
			document_id: dense_candidates[document_id] for document_id in dense_ids
		}
		candidates.update(sparse_candidates)
		scores = {
			document_id: (1.0 - fusion_weight)
			* sparse_percentiles.get(document_id, 0.0)
			+ fusion_weight * dense_percentiles.get(document_id, 0.0)
			for document_id in candidates
		}
		return [
			replace(candidates[document_id], score=scores[document_id])
			for document_id in rank_ids(candidates, scores, descending=True)[:limit]
		]


def _rerank_percentiles(
	document_ids: Collection[int],
	sparse_scores: Mapping[int, float | None],
	dense_scores: Mapping[int, float | None],
	semantic_weight: float,
) -> dict[int, float]:
	"""Blend raw-score percentiles within one route, then percentile-rank the blend."""
	sparse_percentiles = rank_percentiles(document_ids, sparse_scores)
	dense_percentiles = rank_percentiles(document_ids, dense_scores, descending=True)
	scores = {
		document_id: (1.0 - semantic_weight) * sparse_percentiles[document_id]
		+ semantic_weight * dense_percentiles[document_id]
		for document_id in document_ids
	}
	return rank_percentiles(document_ids, scores, descending=True)
