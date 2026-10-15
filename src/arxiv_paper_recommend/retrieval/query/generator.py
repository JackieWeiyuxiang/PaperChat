"""Generate structured keywords and semantic queries with the configured LLM."""

from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, create_model

from arxiv_paper_recommend.config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

from .._lifecycle import ManagedResource
from .._validation import validate_integer, validate_text


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
    base: type[Result], field_name: str, limit: int,
) -> type[Result]:
    description = (
        f"1 to {limit} English technical terms, synonyms and acronyms for BM25 retrieval."
        if field_name == "terms"
        else f"1 to {limit} concise English semantic queries for dense vector retrieval."
    )
    return create_model(
        base.__name__,
        __base__=base,
        **{field_name: (list[str], Field(min_length=1, max_length=limit, description=description))},
    )


class QueryGenerator(ManagedResource):
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

    def __init__(self) -> None:
        super().__init__()
        if not LLM_MODEL or not LLM_MODEL.strip():
            raise ValueError("LLM_MODEL must be configured with a non-empty model name.")
        self.model = LLM_MODEL.strip()
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_URL)
        self._resources.callback(self.client.close)

    def generate_terms(self, request: str, max_terms: int = 12) -> list[str]:
        """Generate up to max_terms keywords; the limit must be at least 5."""
        self._ensure_open()
        validate_integer(max_terms, "max_terms", minimum=5)
        request = validate_text(request, "Research request", strip=True)
        result = self._generate(
            request,
            self.terms_prompt_template.format(max_terms=max_terms),
            _result_model(TermsGenerationResult, "terms", max_terms),
        )
        return result.terms

    def generate_queries(self, request: str, max_queries: int = 4) -> list[str]:
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
            request, instructions, _result_model(QueriesGenerationResult, "queries", max_queries),
        )
        return result.queries

    def _generate[Result: BaseModel](
        self, request: str, instructions: str, result_model: type[Result],
    ) -> Result:
        response = self.client.responses.parse(
            model=self.model,
            reasoning={"effort": "none"},
            temperature=0,
            instructions=instructions,
            input=request,
            text_format=result_model,
        )
        if response.output_parsed is None:
            raise RuntimeError("Generation did not return a structured result.")
        return response.output_parsed
