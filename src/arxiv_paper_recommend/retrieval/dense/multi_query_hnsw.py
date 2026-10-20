"""Multi-query HNSW recall and exact reranking with internal batching."""

from collections.abc import Sequence

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

		Each division recalls at most limit documents. An empty sequence returns [];
		final scores are exact inner products, with ties resolved by document ID.
		Neighbor IDs are deduplicated before loading documents; result objects
		are created only for the final selection.
		"""
		if recall_queries is None:
			return super().search(query, limit=limit)
		self._ensure_open()
		validate_text(query, "Query")
		validate_integer(limit, "Result limit")
		queries = normalize_queries(recall_queries)
		if not queries or self.index.ntotal == 0:
			return []

		_, ids = self.index.search(
			self._prepare_queries(queries), min(limit, self.index.ntotal)
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
			return []

		scores = self.score(query, candidate_ids)
		return [
			SearchResult(*documents[document_id], score=scores[document_id])
			for document_id in rank_ids(candidate_ids, scores, descending=True)[:limit]
		]
