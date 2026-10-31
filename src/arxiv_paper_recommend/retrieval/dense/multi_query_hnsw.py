"""Multi-query HNSW recall and exact reranking with internal batching."""

from collections.abc import Sequence
from math import ceil, sqrt

from arxiv_paper_recommend.common.validation import (
	normalize_queries,
	validate_integer,
	validate_text,
)

from .._ranking import rank_ids
from ..constants import DEFAULT_RESULT_LIMIT
from ..search_result import SearchResult
from .hnsw import HNSWRetriever


class MultiQueryHNSWRetriever(HNSWRetriever):
	"""Reuse one index and vectorizer for multi-query recall and query scoring."""

	def search(
		self,
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		recall_queries: Sequence[str] | None = None,
	) -> list[SearchResult]:
		"""Rerank multi-query candidates by query, or search once when omitted.

		Each facet recalls at most ceil(limit * sqrt(facet count)) documents.
		Facet count is deduplicated. An empty sequence returns [];
		final scores are exact inner products, with ties resolved by document ID.
		Neighbor IDs are deduplicated before loading documents; result objects
		are created only for the final selection.
		"""
		if recall_queries is None:
			return super().search(query, limit=limit)
		return self._recall(
			query, limit, recall_queries=recall_queries, result_limit=limit
		)

	def recall(
		self,
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		recall_queries: Sequence[str] | None = None,
	) -> list[SearchResult]:
		"""Return the full recall pool with exact query scores via raw_recall()."""
		return self.raw_recall(query, limit=limit, recall_queries=recall_queries)

	def raw_recall(
		self,
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		recall_queries: Sequence[str] | None = None,
	) -> list[SearchResult]:
		"""Return the full deduplicated recall pool with exact query scores.

		Each recall query contributes at most limit neighbors. None uses query
		as the sole recall query; an empty sequence returns no candidates.
		The complete pool is ordered by exact inner product, then document ID,
		without truncating the merged pool to limit.
		"""
		return self._recall(
			query, limit, recall_queries=recall_queries, result_limit=None
		)

	def _recall(
		self,
		query: str,
		limit: int,
		*,
		recall_queries: Sequence[str] | None,
		result_limit: int | None,
	) -> list[SearchResult]:
		"""Scale search depth by facet count; raw recall keeps the supplied depth."""
		self._ensure_open()
		validate_text(query, "Query")
		validate_integer(limit, "Result limit")
		queries = (
			[query] if recall_queries is None else normalize_queries(recall_queries)
		)
		if not queries or self.index.ntotal == 0:
			return []

		recall_limit = (
			ceil(limit * sqrt(len(queries))) if result_limit is not None else limit
		)
		documents, raw_scores = self._recall_candidates(queries, [query], recall_limit)
		if not documents:
			return []
		scores = raw_scores[query]
		return [
			SearchResult(*documents[document_id], score=scores[document_id])
			for document_id in rank_ids(documents, scores, descending=True)[
				:result_limit
			]
		]

	def _recall_candidates(
		self,
		recall_queries: list[str],
		scoring_queries: list[str],
		limit: int,
	) -> tuple[
		dict[int, tuple[int, str, str, str | None]],
		dict[str, dict[int, float | None]],
	]:
		"""Encode recall/scoring queries once and score one complete candidate pool."""
		if not recall_queries or self.index.ntotal == 0:
			return {}, {}
		queries = list(dict.fromkeys([*scoring_queries, *recall_queries]))
		query_vectors = self._prepare_queries(queries)
		positions = {query: position for position, query in enumerate(queries)}
		_, ids = self.index.search(
			query_vectors[[positions[query] for query in recall_queries]],
			min(limit, self.index.ntotal),
		)
		candidate_ids = list(
			dict.fromkeys(
				document_id for document_id in ids.ravel().tolist() if document_id >= 0
			)
		)
		documents = {
			document[0]: document
			for document in self._retrieve_documents(candidate_ids)
		}
		candidate_ids = [
			document_id for document_id in candidate_ids if document_id in documents
		]
		if not candidate_ids:
			return {}, {}
		return documents, self._score_queries(
			scoring_queries,
			candidate_ids,
			query_vectors=query_vectors[
				[positions[query] for query in scoring_queries]
			],
		)
