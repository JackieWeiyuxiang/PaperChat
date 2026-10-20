"""Divisional HNSW recall and exact reranking with internal batching."""

from .._ranking import rank_ids
from .._validation import normalize_queries, validate_integer, validate_text
from ..base import DEFAULT_RESULT_LIMIT
from ..search_result import SearchResult
from .hnsw import HNSWRetriever


class DivisionalHNSWRetriever(HNSWRetriever):
	"""Reuse one index and vectorizer for divisional recall and query scoring."""

	def search(
		self,
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		divisional_queries: str | set[str] | None = None,
	) -> list[SearchResult]:
		"""Rerank divisional candidates by query, or search once when omitted.

		Each division recalls at most limit documents. An empty set returns [];
		final scores are exact inner products, with ties resolved by document ID.
		Neighbor IDs are deduplicated before loading documents; result objects
		are created only for the final selection.
		"""
		if divisional_queries is None:
			return super().search(query, limit=limit)
		self._ensure_open()
		validate_text(query, "Query")
		validate_integer(limit, "Result limit")
		queries = normalize_queries(divisional_queries)
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

		scores = self.get_score(candidate_ids, query)
		return [
			SearchResult(*documents[document_id], score=scores[document_id])
			for document_id in rank_ids(candidate_ids, scores, descending=True)[:limit]
		]
