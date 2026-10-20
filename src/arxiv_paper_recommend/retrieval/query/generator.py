"""Generate structured keywords and semantic queries with the configured LLM."""

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, create_model

from arxiv_paper_recommend.common.validation import validate_integer, validate_text
from arxiv_paper_recommend.llm import StructuredGenerator

from ..constants import DEFAULT_MAX_QUERIES, DEFAULT_MAX_TERMS, MIN_GENERATED_TERM_COUNT


class TermsGenerationResult(BaseModel):
	"""Validated keywords for sparse paper retrieval."""

	model_config = ConfigDict(extra="forbid")
	terms: list[str] = Field(
		min_length=1,
		description="English technical terms, synonyms and acronyms for BM25 retrieval.",
	)


class QueriesGenerationResult(BaseModel):
	"""Validated semantic queries for dense paper retrieval."""

	model_config = ConfigDict(extra="forbid")
	queries: list[str] = Field(
		min_length=1,
		description="Concise English semantic queries for dense vector retrieval.",
	)


@lru_cache(maxsize=32)
def _result_model[Result: BaseModel](
	base: type[Result],
	field_name: str,
	limit: int,
) -> type[Result]:
	description = (
		f"1 to {limit} English technical terms, synonyms and acronyms for BM25 retrieval."
		if field_name == "terms"
		else f"1 to {limit} concise English semantic queries for dense vector retrieval."
	)
	return create_model(
		base.__name__,
		__base__=base,
		**{
			field_name: (
				list[str],
				Field(min_length=1, max_length=limit, description=description),
			)
		},
	)


class QueryGenerator(StructuredGenerator):
	"""Reuse one client; each generation method makes one structured LLM call."""

	terms_prompt_template: str = """
        Given a user's research request, generate 1 to {max_terms} English match terms for retrieving academic papers by title and abstract.

        - **terms**: 1 to {max_terms} relevant, specialized keywords and phrases for sparse retrieval (BM25). Include relevant synonyms and acronyms while avoiding redundant terms.
        - Use fewer items when sufficient; do not add items just to reach the limits.
        - Preserve the user's focuses and constraints. Prioritize retrieval relevance and coverage.

    """
	queries_prompt_template: str = """
        Given a user's research request, generate 1 to {max_queries} English semantic queries for retrieving academic papers by title and abstract.

        - **queries**: 1 to {max_queries} concise, relevant queries for dense vector retrieval. Generate multiple only when they cover distinct relevant aspects.
        - Use fewer items when sufficient; do not add items just to reach the limits.
        - Preserve the user's focuses and constraints. Prioritize retrieval relevance and coverage.

    """
	single_query_prompt_template: str = """
        Given a user's research request, generate exactly one English semantic query for retrieving academic papers by title and abstract.

        - **queries**: a list containing exactly one concise, self-contained query for dense vector retrieval.
        - Integrate the core research topic, relevant relationships, and essential constraints into one coherent natural-language query.
        - Preserve the user's intent and use precise academic terminology. Do not split the request into separate queries or produce a keyword list.
        - Prioritize retrieval relevance; omit incidental details and do not invent requirements.

    """

	def generate_terms(
		self, request: str, max_terms: int = DEFAULT_MAX_TERMS
	) -> list[str]:
		"""Generate up to max_terms keywords; the limit must be at least 5."""
		self._ensure_open()
		validate_integer(max_terms, "max_terms", minimum=MIN_GENERATED_TERM_COUNT)
		request = validate_text(request, "Research request", strip=True)
		result = self._generate(
			request,
			self.terms_prompt_template.format(max_terms=max_terms),
			_result_model(TermsGenerationResult, "terms", max_terms),
		)
		for term in result.terms:
			if not term.strip():
				raise RuntimeError("Generation returned an empty term.")
		return result.terms

	def generate_queries(
		self, request: str, max_queries: int = DEFAULT_MAX_QUERIES
	) -> list[str]:
		"""Generate queries; max_queries=1 uses the integrated-query prompt."""
		self._ensure_open()
		validate_integer(max_queries, "max_queries")
		request = validate_text(request, "Research request", strip=True)
		instructions = (
			self.single_query_prompt_template
			if max_queries == 1
			else self.queries_prompt_template.format(max_queries=max_queries)
		)
		result = self._generate(
			request,
			instructions,
			_result_model(QueriesGenerationResult, "queries", max_queries),
		)
		for query in result.queries:
			if not query.strip():
				raise RuntimeError("Generation returned an empty query.")
		return result.queries
