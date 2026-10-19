"""BM25 retrieval that gives paper titles more influence than abstracts."""

from pathlib import Path

from arxiv_paper_recommend.config import DATABASE_PATH
from arxiv_paper_recommend.retrieval.search_result import SearchResult
from arxiv_paper_recommend.retrieval.sparse.bm25_retriever import (
	DEFAULT_RESULT_LIMIT,
	BM25Retriever,
)

TITLE_WEIGHT = 3.0
ABSTRACT_WEIGHT = 1.0


class TitleAwareBM25Retriever(BM25Retriever):
	"""Use weighted FTS5 BM25, boosting title matches over abstract matches."""

	def __init__(self, database_path: str | Path = DATABASE_PATH) -> None:
		super().__init__(database_path)

	def term_search(
		self,
		terms: list[str],
		limit: int = DEFAULT_RESULT_LIMIT,
	) -> list[SearchResult]:
		"""Return title-aware lexical matches, using the configured 3:1 weights."""
		return self._term_search(
			terms,
			limit,
			column_weights=(TITLE_WEIGHT, ABSTRACT_WEIGHT, 0.0),
		)
