"""Rerank HRRF candidates by their weakest semantic query facet."""

from collections.abc import Sequence
from dataclasses import replace
from math import isfinite

from arxiv_paper_recommend.common.validation import validate_integer

from .._ranking import rank_ids
from ..constants import DEFAULT_RESULT_LIMIT
from ..hybrid import HRRFRetriever
from ..hybrid.base import DenseScoreCache, HybridRetriever
from ..hybrid.constants import MAX_RESULT_LIMIT
from ..search_result import SearchResult

DEFAULT_CANDIDATE_LIMIT = 50
DEFAULT_COVERAGE_WEIGHT = 0.5


class FacetCoverageRetriever(HybridRetriever):
	"""Balance HRRF relevance with every semantic facet's candidate rank.

	The retriever first obtains a fixed-depth HRRF candidate list. It then
	scores every candidate against each recall query and uses the weakest
	facet percentile as a coverage signal. The same recall queries therefore
	drive both HRRF candidate generation and coverage reranking.
	"""

	def search(
		self,
		terms: list[str],
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		title_weight: float | None = None,
		recall_queries: Sequence[str] | None = None,
		candidate_limit: int | None = None,
		coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
	) -> list[SearchResult]:
		"""Return coverage-aware scores, with ties resolved by document ID.

		``candidate_limit`` fixes the HRRF pool depth and must be at least the
		requested result limit. Its default is the greater of 50 and ``limit``.
		With an explicitly empty recall-query sequence, there is no coverage
		signal and the original HRRF order is preserved.
		"""
		facets = self._validate_inputs(query, limit, recall_queries)
		if candidate_limit is None:
			candidate_limit = max(DEFAULT_CANDIDATE_LIMIT, limit)
		validate_integer(
			candidate_limit,
			"Candidate limit",
			minimum=limit,
			maximum=MAX_RESULT_LIMIT,
		)
		coverage_weight = _validate_coverage_weight(coverage_weight)

		with HRRFRetriever(
			sparse_retriever=self.sparse_retriever,
			dense_retriever=self.dense_retriever,
		) as candidate_retriever:
			candidates = candidate_retriever.search(
				terms,
				query,
				limit=candidate_limit,
				title_weight=title_weight,
				recall_queries=recall_queries,
			)
		if not candidates:
			return []
		if not facets or coverage_weight == 0.0:
			return [replace(result) for result in candidates[:limit]]

		document_ids = [result.document_id for result in candidates]
		cache = DenseScoreCache(self.dense_retriever)
		cache.ensure({facet: document_ids for facet in facets})
		scores = _facet_coverage_scores(
			document_ids,
			{facet: cache.scores[facet] for facet in facets},
			coverage_weight,
		)
		by_id = {result.document_id: result for result in candidates}
		return [
			replace(by_id[document_id], score=scores[document_id])
			for document_id in rank_ids(document_ids, scores, descending=True)[:limit]
		]


def _facet_coverage_scores(
	document_ids: list[int],
	facet_scores: dict[str, dict[int, float | None]],
	coverage_weight: float,
) -> dict[int, float]:
	"""Combine HRRF rank percentile with the weakest facet percentile."""
	candidate_count = len(document_ids)
	base_percentiles = {
		document_id: (candidate_count - rank + 1) / candidate_count
		for rank, document_id in enumerate(document_ids, start=1)
	}
	coverage = dict.fromkeys(document_ids, 1.0)
	for scores in facet_scores.values():
		scored_ids = [
			document_id
			for document_id in document_ids
			if scores.get(document_id) is not None
		]
		facet_percentiles = {
			document_id: (candidate_count - rank + 1) / candidate_count
			for rank, document_id in enumerate(
				rank_ids(scored_ids, scores, descending=True), start=1
			)
		}
		for document_id in document_ids:
			coverage[document_id] = min(
				coverage[document_id], facet_percentiles.get(document_id, 0.0)
			)
	return {
		document_id: (1.0 - coverage_weight) * base_percentiles[document_id]
		+ coverage_weight * coverage[document_id]
		for document_id in document_ids
	}


def _validate_coverage_weight(coverage_weight: float) -> float:
	"""Accept finite numeric weights from zero through one, excluding booleans."""
	if isinstance(coverage_weight, bool) or not isinstance(
		coverage_weight, (int, float)
	):
		raise TypeError("Coverage weight must be an integer or a float.")
	try:
		coverage_weight = float(coverage_weight)
	except OverflowError as error:
		raise ValueError(
			"Coverage weight must be finite and between 0 and 1."
		) from error
	if not isfinite(coverage_weight) or not 0.0 <= coverage_weight <= 1.0:
		raise ValueError("Coverage weight must be finite and between 0 and 1.")
	return coverage_weight
