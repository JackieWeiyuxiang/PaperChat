"""SQLite FTS5 retrieval with independently scored title and abstract fields."""

from pathlib import Path
from sqlite3 import SQLITE_LIMIT_VARIABLE_NUMBER, connect
from unicodedata import category

from arxiv_paper_recommend.common.validation import validate_integer
from arxiv_paper_recommend.config import DATABASE_PATH

from ..base import Retriever
from ..constants import (
	DEFAULT_RESULT_LIMIT,
	MAX_QUERY_LENGTH,
	MAX_TERM_COUNT,
	MIN_TERM_COUNT,
)
from ..search_result import SearchResult


class BM25Retriever(Retriever[list[str]]):
	"""Reuse a read-only connection; use with or close() when finished."""

	def __init__(self, database_path: str | Path = DATABASE_PATH) -> None:
		super().__init__()
		database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"
		self.conn = connect(database_uri, uri=True)
		self._resources.callback(self.conn.close)

	def count_documents(self) -> int:
		"""Return the total document count, independent of keyword matches."""
		self._ensure_open()
		return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

	def search(
		self, query: list[str], limit: int = DEFAULT_RESULT_LIMIT
	) -> list[SearchResult]:
		"""Match any keyword or phrase, ordered by summed field BM25 then ID.

		Accept 1–20 distinct terms after normalization and a positive integer
		limit. The escaped OR query must not exceed 1,347 characters.
		"""
		return self._search(query, limit=limit)

	def score(
		self, query: list[str], document_ids: list[int]
	) -> dict[int, float | None]:
		"""Score specified IDs with 1–20 terms, without search length or count limits.

		Preserve first-occurrence ID order. Existing nonmatches score 0.0;
		missing documents score None. Duplicate IDs are scored only once.
		"""
		return self._score(query, document_ids)

	def _search(
		self,
		query: list[str],
		limit: int,
		*,
		title_weight: float | None = None,
	) -> list[SearchResult]:
		"""Search with an optional title weight and unit abstract weight."""
		self._ensure_open()
		validate_integer(limit, "Result limit")
		terms = _normalize_terms(query)
		query_length = sum(len(term) + term.count('"') + 2 for term in terms) + len(
			" OR "
		) * (len(terms) - 1)
		if query_length > MAX_QUERY_LENGTH:
			raise ValueError(
				f"Search query length must be at most {MAX_QUERY_LENGTH} characters."
			)
		score_sql, score_parameters = self._score_sql(_build_query(terms), title_weight)
		rows = self.conn.execute(
			f"""
            {score_sql}
            SELECT documents.id, documents.doc_id, documents.title,
                   documents.abstract, document_scores.score
            FROM document_scores
            JOIN documents ON documents.id = document_scores.document_id
            ORDER BY document_scores.score, documents.id
            LIMIT :limit
        """,
			{**score_parameters, "limit": limit},
		).fetchall()
		return [SearchResult(*row) for row in rows]

	def _score(
		self,
		query: list[str],
		document_ids: list[int],
		*,
		title_weight: float | None = None,
	) -> dict[int, float | None]:
		"""Score IDs with an optional title weight and unit abstract weight."""
		self._ensure_open()
		terms = _normalize_terms(query)
		document_ids = list(dict.fromkeys(document_ids))
		scores: dict[int, float | None] = dict.fromkeys(document_ids)
		if not document_ids:
			return scores

		match_query = _build_query(terms)
		_, score_parameters = self._score_sql(match_query, title_weight)
		batch_size = self.conn.getlimit(SQLITE_LIMIT_VARIABLE_NUMBER) - len(
			score_parameters
		)
		for start in range(0, len(document_ids), batch_size):
			batch = document_ids[start : start + batch_size]
			placeholders = ",".join("?" for _ in batch)
			documents = self.conn.execute(
				f"SELECT id FROM documents WHERE id IN ({placeholders})",
				batch,
			).fetchall()
			for (document_id,) in documents:
				scores[document_id] = 0.0
			score_sql, parameters = self._score_sql(match_query, title_weight, batch)
			rows = self.conn.execute(
				f"""
                {score_sql}
                SELECT document_id, score FROM document_scores
            """,
				parameters,
			).fetchall()
			scores.update(rows)
		return scores

	@staticmethod
	def _score_sql(
		match_query: str,
		title_weight: float | None,
		document_ids: list[int] | None = None,
	) -> tuple[str, dict[str, str | float | int]]:
		"""Materialize field scores before summing, optionally filtering IDs.

		Named parameters are reused by both MATCH branches, so each query,
		weight and document ID consumes only one SQLite variable slot.
		"""
		parameters: dict[str, str | float | int] = {
			"match_query": match_query,
			"title_weight": 1.0 if title_weight is None else title_weight,
		}
		id_filter = ""
		if document_ids is not None:
			id_parameters = {
				f"document_id_{index}": document_id
				for index, document_id in enumerate(document_ids)
			}
			parameters.update(id_parameters)
			placeholders = ",".join(f":{name}" for name in id_parameters)
			id_filter = f"AND rowid IN ({placeholders})"
		return (
			f"""
            WITH field_scores AS MATERIALIZED (
                SELECT rowid AS document_id,
                       :title_weight * bm25(documents_title_fts) AS score
                FROM documents_title_fts
                WHERE documents_title_fts MATCH :match_query {id_filter}
                UNION ALL
                SELECT rowid AS document_id, bm25(documents_abstract_fts) AS score
                FROM documents_abstract_fts
                WHERE documents_abstract_fts MATCH :match_query {id_filter}
            ),
            document_scores AS (
                SELECT document_id, SUM(score) AS score
                FROM field_scores
                GROUP BY document_id
            )
            """,
			parameters,
		)


def _normalize_terms(terms: list[str]) -> list[str]:
	"""Normalize whitespace and case, retaining first-occurrence term order."""
	if not isinstance(terms, list):
		raise TypeError("Keywords must be a list of strings.")
	processed: dict[str, None] = {}
	for keyword in terms:
		if not isinstance(keyword, str):
			raise TypeError("Keywords must contain only strings.")
		keyword = " ".join(keyword.split()).lower()
		if not keyword:
			continue
		# unicode61 (used by porter) accepts letters, numbers and private-use characters.
		if not any(
			char_category[0] in ("L", "N") or char_category == "Co"
			for char_category in map(category, keyword)
		):
			raise ValueError(
				"Each keyword must contain at least one searchable token character."
			)
		processed.setdefault(keyword, None)
	if not MIN_TERM_COUNT <= len(processed) <= MAX_TERM_COUNT:
		raise ValueError(
			f"Keyword count must be between {MIN_TERM_COUNT} and {MAX_TERM_COUNT}."
		)
	return list(processed)


def _build_query(terms: list[str]) -> str:
	"""Quote phrases and escape embedded double quotes for FTS5."""
	return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
