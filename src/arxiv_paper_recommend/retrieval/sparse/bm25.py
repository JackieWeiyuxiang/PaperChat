"""SQLite FTS5 keyword retrieval and BM25 scoring."""

from pathlib import Path
from sqlite3 import SQLITE_LIMIT_VARIABLE_NUMBER, connect
from unicodedata import category

from arxiv_paper_recommend.config import DATABASE_PATH

from .._validation import validate_integer
from ..base import DEFAULT_RESULT_LIMIT, Retriever
from ..search_result import SearchResult

MIN_KEYWORD_COUNT = 1
MAX_KEYWORD_COUNT = 20
MAX_QUERY_LENGTH = 1347


class BM25Retriever(Retriever[list[str]]):
	"""Reuse a read-only connection; use with or close() when finished."""

	def __init__(self, database_path: str | Path = DATABASE_PATH) -> None:
		super().__init__()
		database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"
		self.conn = connect(database_uri, uri=True)
		self._resources.callback(self.conn.close)

	def search(
		self, query: list[str], limit: int = DEFAULT_RESULT_LIMIT
	) -> list[SearchResult]:
		"""Match any normalized keyword or phrase, ordered by ascending BM25.

		Accept 1–20 distinct terms after normalization and a positive integer
		limit. The escaped OR query must not exceed 1,347 characters.
		"""
		return self._search(query, limit=limit)

	def get_score(self, ids: list[int], query: list[str]) -> dict[int, float | None]:
		"""Score specified IDs with 1–20 terms, without search length or count limits.

		Preserve first-occurrence ID order. Existing nonmatches score 0.0;
		missing documents score None. Duplicate IDs are scored only once.
		"""
		return self._get_score(ids, query)

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
		terms = _process_keywords(query)
		query_length = sum(len(term) + term.count('"') + 2 for term in terms) + len(
			" OR "
		) * (len(terms) - 1)
		if query_length > MAX_QUERY_LENGTH:
			raise ValueError(
				f"Search query length must be at most {MAX_QUERY_LENGTH} characters."
			)
		score_expression, score_parameters = self._get_score_sql(title_weight)
		order_expression = "rank" if title_weight is None else "score"
		rows = self.conn.execute(
			f"""
            SELECT documents_fts.rowid, doc_id, title, abstract, {score_expression} AS score
            FROM documents_fts
            WHERE documents_fts MATCH ?
            ORDER BY {order_expression}
            LIMIT ?
        """,
			(*score_parameters, _build_query(terms), limit),
		).fetchall()
		return [SearchResult(*row) for row in rows]

	def _get_score(
		self,
		ids: list[int],
		query: list[str],
		*,
		title_weight: float | None = None,
	) -> dict[int, float | None]:
		"""Score IDs with an optional title weight and unit abstract weight."""
		self._ensure_open()
		terms = _process_keywords(query)
		document_ids = list(dict.fromkeys(ids))
		scores: dict[int, float | None] = dict.fromkeys(document_ids)
		if not document_ids:
			return scores

		match_query = _build_query(terms)
		score_expression, score_parameters = self._get_score_sql(title_weight)
		batch_size = (
			self.conn.getlimit(SQLITE_LIMIT_VARIABLE_NUMBER) - len(score_parameters) - 1
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
			rows = self.conn.execute(
				f"""
                SELECT rowid, {score_expression}
                FROM documents_fts
                WHERE documents_fts MATCH ? AND documents_fts.rowid IN ({placeholders})
            """,
				[*score_parameters, match_query, *batch],
			).fetchall()
			scores.update(rows)
		return scores

	@staticmethod
	def _get_score_sql(
		title_weight: float | None,
	) -> tuple[str, tuple[float, ...]]:
		"""Bind only the title weight; the abstract weight is always one."""
		if title_weight is None:
			return "rank", ()
		return "bm25(documents_fts, ?, 1.0)", (title_weight,)


def _process_keywords(keywords: list[str]) -> list[str]:
	"""Normalize whitespace and case, retaining first-occurrence term order."""
	if not isinstance(keywords, list):
		raise TypeError("Keywords must be a list of strings.")
	processed: dict[str, None] = {}
	for keyword in keywords:
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
	if not MIN_KEYWORD_COUNT <= len(processed) <= MAX_KEYWORD_COUNT:
		raise ValueError(
			f"Keyword count must be between {MIN_KEYWORD_COUNT} and {MAX_KEYWORD_COUNT}."
		)
	return list(processed)


def _build_query(terms: list[str]) -> str:
	"""Quote phrases and escape embedded double quotes for FTS5."""
	return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
