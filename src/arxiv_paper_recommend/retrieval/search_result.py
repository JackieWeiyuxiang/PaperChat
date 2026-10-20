"""Shared result records for all retrieval methods."""

from dataclasses import dataclass


@dataclass
class SearchResult:
	"""单条搜索结果，包含论文标识、标题和摘要。"""

	document_id: int
	arxiv_id: str
	title: str
	abstract: str | None
	score: float | None = None
