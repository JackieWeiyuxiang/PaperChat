"""FAISS HNSW text retrieval and exact candidate scoring."""

from collections.abc import Iterator
from pathlib import Path
from sqlite3 import SQLITE_LIMIT_VARIABLE_NUMBER, connect

import faiss
import numpy as np
from numpy.typing import NDArray

from arxiv_paper_recommend.common.validation import validate_integer, validate_text
from arxiv_paper_recommend.config import DATABASE_PATH, INDEX_PATH
from arxiv_paper_recommend.vectorization import Qwen3Vectorizer

from ..base import Retriever
from ..constants import DEFAULT_RESULT_LIMIT
from ..search_result import SearchResult


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
		validate_integer(limit, "Result limit")
		limit = min(limit, self.index.ntotal)
		if limit == 0:
			return []
		scores, ids = self.index.search(self._prepare_queries([query]), limit)
		return self._results_from_neighbors(ids, scores)[0]

	def score(self, query: str, document_ids: list[int]) -> dict[int, float | None]:
		"""Return exact inner products for one query and the specified IDs."""
		return self.raw_score(query, document_ids)

	def raw_score(self, query: str, document_ids: list[int]) -> dict[int, float | None]:
		"""Return exact inner products independently of subclass score formulas."""
		self._ensure_open()
		validate_text(query, "Query")
		scores: dict[int, float | None] = dict.fromkeys(document_ids)
		valid_ids, vectors = self._reconstruct_documents(list(scores))
		if not valid_ids:
			return scores
		query_vector = self._prepare_queries([query])[0]
		scores.update(zip(valid_ids, (vectors @ query_vector).tolist()))
		return scores

	def _results_from_neighbors(
		self,
		ids: NDArray[np.int64],
		scores: NDArray[np.float32],
	) -> list[list[SearchResult]]:
		"""Read result payloads once and preserve each FAISS neighbor row's order."""
		documents = {
			document[0]: document
			for document in self._retrieve_documents(ids.ravel().tolist())
		}
		return [
			[
				SearchResult(*documents[document_id], score=score)
				for document_id, score in zip(query_ids.tolist(), query_scores.tolist())
				if document_id in documents
			]
			for query_ids, query_scores in zip(ids, scores)
		]

	def _reconstruct_documents(
		self, ids: list[int]
	) -> tuple[list[int], NDArray[np.float32]]:
		"""Reconstruct only IDs present in both the inner-product index and database."""
		if (
			not isinstance(self.index, faiss.IndexIDMap2)
			or self.index.metric_type != faiss.METRIC_INNER_PRODUCT
		):
			raise ValueError("Scoring requires an IndexIDMap2 inner-product index.")

		document_ids = list(dict.fromkeys(ids))
		if not document_ids or self.index.ntotal == 0:
			return [], np.empty((0, self.index.d), dtype=np.float32)
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
			return [], np.empty((0, self.index.d), dtype=np.float32)

		vectors = self.index.reconstruct_batch(np.asarray(valid_ids, dtype=np.int64))
		return valid_ids, vectors

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
