"""FAISS HNSW text retrieval and exact candidate scoring."""

from collections.abc import Iterator
from pathlib import Path
from sqlite3 import SQLITE_LIMIT_VARIABLE_NUMBER, connect

import faiss
import numpy as np
from numpy.typing import NDArray

from arxiv_paper_recommend.config import DATABASE_PATH, INDEX_PATH
from arxiv_paper_recommend.vectorization import Qwen3Vectorizer

from .._validation import normalize_queries, validate_integer, validate_text
from ..base import DEFAULT_RESULT_LIMIT, Retriever
from ..search_result import SearchResult

DEFAULT_BATCH_RESULT_LIMIT = 5


class HNSWRetriever(Retriever[str]):
	"""Reuse an index, a read-only connection and a lazily created vectorizer.

	The index must stay unchanged throughout this retriever's lifetime.
	An injected vectorizer remains owned by the caller.
	"""

	def __init__(
		self,
		index_path: str | Path = INDEX_PATH,
		database_path: str | Path = DATABASE_PATH,
		*,
		vectorizer: Qwen3Vectorizer | None = None,
	) -> None:
		super().__init__()
		self.index = faiss.read_index(str(index_path))
		database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"
		self.conn = connect(database_uri, uri=True)
		self._resources.callback(self.conn.close)
		self.vectorizer = vectorizer
		self._indexed_ids: set[int] | None = None

	def search(
		self,
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
	) -> list[SearchResult]:
		"""Return one query's nearest neighbors in FAISS order."""
		self._ensure_open()
		validate_text(query, "Query")
		return self.batch_search(query, limit=limit)[query]

	def get_score(self, ids: list[int], query: str) -> dict[int, float | None]:
		"""Return exact inner products for one query and the specified IDs."""
		self._ensure_open()
		validate_text(query, "Query")
		return self.batch_get_score({query: ids})[query]

	def batch_search(
		self,
		query: str | set[str],
		limit: int = DEFAULT_BATCH_RESULT_LIMIT,
	) -> dict[str, list[SearchResult]]:
		"""Return per-query results in FAISS order, with at most limit each.

		Sets are processed in string order. Empty sets return {}; empty indexes
		return an empty list per valid query without loading the embedding model.
		Invalid FAISS IDs and records missing from the database are omitted.
		"""
		self._ensure_open()
		validate_integer(limit, "Result limit")
		queries = normalize_queries(query)
		if not queries:
			return {}
		limit = min(limit, self.index.ntotal)
		if limit == 0:
			return {text: [] for text in queries}

		scores, ids = self.index.search(self._prepare_queries(queries), limit)
		documents = {
			document[0]: document
			for document in self._retrieve_documents(ids.ravel().tolist())
		}
		return {
			text: [
				SearchResult(*documents[document_id], score=score)
				for document_id, score in zip(query_ids.tolist(), query_scores.tolist())
				if document_id in documents
			]
			for text, query_ids, query_scores in zip(queries, ids, scores)
		}

	def batch_get_score(
		self,
		query_ids: dict[str, list[int]],
	) -> dict[str, dict[int, float | None]]:
		"""Return exact inner products for each query's requested internal IDs.

		Requires IndexIDMap2 with the inner-product metric. Missing database or
		index IDs score None. Query and first-occurrence ID order are preserved.
		Unique valid document vectors are reconstructed in one batch; each query
		scores only its own IDs. Empty or entirely missing IDs skip encoding.
		"""
		self._ensure_open()
		if not query_ids:
			return {}
		queries = list(query_ids)
		for text in queries:
			validate_text(text, "Query")
		if (
			not isinstance(self.index, faiss.IndexIDMap2)
			or self.index.metric_type != faiss.METRIC_INNER_PRODUCT
		):
			raise ValueError("Scoring requires an IndexIDMap2 inner-product index.")

		scores: dict[str, dict[int, float | None]] = {
			text: dict.fromkeys(query_ids[text]) for text in queries
		}
		document_ids = list(
			dict.fromkeys(
				document_id
				for query_scores in scores.values()
				for document_id in query_scores
			)
		)
		if not document_ids or self.index.ntotal == 0:
			return scores
		if self._indexed_ids is None:
			self._indexed_ids = set(faiss.vector_to_array(self.index.id_map).tolist())
		indexed_ids = [
			document_id
			for document_id in document_ids
			if document_id in self._indexed_ids
		]
		existing_ids = self._existing_ids(indexed_ids)
		valid_ids = [
			document_id for document_id in indexed_ids if document_id in existing_ids
		]
		if not valid_ids:
			return scores

		vectors = self.index.reconstruct_batch(np.asarray(valid_ids, dtype=np.int64))
		positions = {
			document_id: position for position, document_id in enumerate(valid_ids)
		}
		active_queries = [
			text
			for text in queries
			if any(document_id in positions for document_id in scores[text])
		]
		query_vectors = self._prepare_queries(active_queries)
		for text, query_vector in zip(active_queries, query_vectors):
			ids = [
				document_id for document_id in scores[text] if document_id in positions
			]
			rows = [positions[document_id] for document_id in ids]
			similarities = vectors[rows] @ query_vector
			scores[text].update(zip(ids, similarities.tolist()))
		return scores

	def _prepare_queries(self, queries: list[str]) -> NDArray[np.float32]:
		"""Copy before in-place normalization, protecting injected vectorizer output."""
		if self.vectorizer is None:
			self.vectorizer = Qwen3Vectorizer()
		vectors = np.array(
			self.vectorizer.encode_queries(queries),
			dtype=np.float32,
			order="C",
			copy=True,
		)
		if vectors.shape != (len(queries), self.index.d):
			raise ValueError(
				f"Query vectors must have shape ({len(queries)}, {self.index.d})."
			)
		faiss.normalize_L2(vectors)
		return vectors

	def _document_batches(self, ids: list[int]) -> Iterator[tuple[list[int], str]]:
		"""Deduplicate valid IDs and respect SQLite's parameter limit."""
		document_ids = list(
			dict.fromkeys(document_id for document_id in ids if document_id >= 0)
		)
		batch_size = self.conn.getlimit(SQLITE_LIMIT_VARIABLE_NUMBER)
		for start in range(0, len(document_ids), batch_size):
			batch = document_ids[start : start + batch_size]
			yield batch, ",".join("?" for _ in batch)

	def _existing_ids(self, ids: list[int]) -> set[int]:
		"""Check existence without reading titles or abstracts."""
		document_ids: set[int] = set()
		for batch, placeholders in self._document_batches(ids):
			rows = self.conn.execute(
				f"SELECT id FROM documents WHERE id IN ({placeholders})",
				batch,
			).fetchall()
			document_ids.update(row[0] for row in rows)
		return document_ids

	def _retrieve_documents(
		self, ids: list[int]
	) -> list[tuple[int, str, str, str | None]]:
		"""Fetch result payloads only for the nearest-neighbor results."""
		documents = []
		for batch, placeholders in self._document_batches(ids):
			documents.extend(
				self.conn.execute(
					f"""
                SELECT id, doc_id, title, abstract
                FROM documents
                WHERE id IN ({placeholders})
            """,
					batch,
				).fetchall()
			)
		return documents
