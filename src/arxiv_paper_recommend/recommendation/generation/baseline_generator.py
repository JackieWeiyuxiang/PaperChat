"""Generate short, evidence-grounded reasons for retrieved paper suggestions."""

import json
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from arxiv_paper_recommend.config import LLM_API_KEY, LLM_API_URL, LLM_MODEL
from arxiv_paper_recommend.retrieval._lifecycle import ManagedResource
from arxiv_paper_recommend.retrieval._validation import validate_integer, validate_text
from arxiv_paper_recommend.retrieval.search_result import SearchResult

MAX_RECOMMENDATIONS = 10
MAX_GENERATION_CANDIDATES = 30


class _GeneratedRecommendation(BaseModel):
	"""Structured model output for one retrieved paper."""

	model_config = ConfigDict(extra="forbid")
	doc_id: str = Field(min_length=1)
	reason: str = Field(min_length=1)


class _GenerationResponse(BaseModel):
	"""Structured list of recommended retrieved paper identifiers and reasons."""

	model_config = ConfigDict(extra="forbid")
	recommendations: list[_GeneratedRecommendation] = Field(
		max_length=MAX_RECOMMENDATIONS,
	)


@dataclass(frozen=True)
class PaperRecommendation:
	"""A retrieved paper paired with its generated recommendation reason."""

	rank: int
	result: SearchResult
	reason: str


class RecommendationGenerator(ManagedResource):
	"""Create a ranked paper shortlist with concise reasons grounded in abstracts."""

	instructions = """
You recommend academic papers for a user's research request using only the
retrieved papers supplied in the input. Return up to the requested number of
papers, ordered by usefulness to the request. Select only supplied doc_id values
and never invent papers, findings, methods, or claims. Each reason must be one
or two concise sentences supported by that paper's title or abstract. Explain
the connection to the request, and omit a paper when the supplied evidence is
too weak to justify a recommendation. Do not use outside knowledge.
"""

	def __init__(
		self, *, model: str | None = None, client: OpenAI | None = None
	) -> None:
		super().__init__()
		selected_model = LLM_MODEL if model is None else model
		if not isinstance(selected_model, str) or not selected_model.strip():
			raise ValueError("A non-empty generation model must be configured.")
		self.model = selected_model.strip()
		self.client = (
			client
			if client is not None
			else OpenAI(
				api_key=LLM_API_KEY,
				base_url=LLM_API_URL,
			)
		)
		if client is None:
			self._resources.callback(self.client.close)

	def generate(
		self,
		request: str,
		papers: list[SearchResult],
		limit: int = MAX_RECOMMENDATIONS,
	) -> list[PaperRecommendation]:
		"""Generate up to limit recommendations from the retrieved candidate list."""
		self._ensure_open()
		request = validate_text(request, "Research request", strip=True)
		validate_integer(
			limit, "Recommendation limit", minimum=1, maximum=MAX_RECOMMENDATIONS
		)
		if not isinstance(papers, list):
			raise TypeError("Papers must be a list of SearchResult objects.")
		for index, paper in enumerate(papers):
			if not isinstance(paper, SearchResult):
				raise TypeError(f"Paper at index {index} must be a SearchResult.")
		if not papers:
			return []

		candidates = papers[:MAX_GENERATION_CANDIDATES]
		papers_by_doc_id: dict[str, SearchResult] = {}
		for paper in candidates:
			papers_by_doc_id.setdefault(paper.doc_id, paper)
		payload = {
			"request": request,
			"max_recommendations": min(limit, len(papers_by_doc_id)),
			"retrieved_papers": [
				{
					"doc_id": paper.doc_id,
					"title": paper.title,
					"abstract": paper.abstract or "",
				}
				for paper in papers_by_doc_id.values()
			],
		}
		response = self.client.responses.parse(
			model=self.model,
			reasoning={"effort": "none"},
			temperature=0,
			instructions=self.instructions,
			input=json.dumps(payload, ensure_ascii=False),
			text_format=_GenerationResponse,
		)
		parsed = response.output_parsed
		if parsed is None:
			raise RuntimeError("Generation did not return a structured result.")

		recommendations: list[PaperRecommendation] = []
		seen_doc_ids: set[str] = set()
		for item in parsed.recommendations:
			if item.doc_id not in papers_by_doc_id:
				raise RuntimeError(
					f"Generation returned a paper outside the retrieved candidates: {item.doc_id!r}."
				)
			if item.doc_id in seen_doc_ids:
				raise RuntimeError(
					f"Generation returned duplicate paper {item.doc_id!r}."
				)
			reason = item.reason.strip()
			if not reason:
				raise RuntimeError(
					f"Generation returned an empty reason for paper {item.doc_id!r}."
				)
			seen_doc_ids.add(item.doc_id)
			recommendations.append(
				PaperRecommendation(
					rank=len(recommendations) + 1,
					result=papers_by_doc_id[item.doc_id],
					reason=reason,
				)
			)
			if len(recommendations) == limit:
				break
		return recommendations
