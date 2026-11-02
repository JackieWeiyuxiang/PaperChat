# arxiv-paper-recommend

[中文](README.md) | English

A retrieval-augmented paper recommendation project for AI6127 (2026), based on arXiv titles and abstracts. Search a corpus of 10,000 papers and receive up to 10 recommendations with short reasons and quoted evidence.

## Run the local UI

Requirements: Python **3.14 or later**, [uv](https://docs.astral.sh/uv/), and SQLite **3.35 or later** with FTS5 support.

From the project root, install the backend and embedding dependencies:

```powershell
uv sync --extra vectorization
```

For first-time setup, copy `.env.example` to `.env`. If `.env` already exists, edit it instead of overwriting it:

```powershell
Copy-Item .env.example .env
```

Set the API credentials and model in `.env`:

```ini
LLM_API_KEY=your_api_key
LLM_API_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
```

The configured service and model must support the structured Responses API and the reasoning parameters used by this project. Credentials are read only by the server; `.env` is excluded from Git. Process environment variables override `.env` settings.

Start the UI:

```powershell
uv run python -m arxiv_paper_recommend.web
```

Open **http://127.0.0.1:8000**. To use another port:

```powershell
uv run python -m arxiv_paper_recommend.web --port 8001
```

Describe a research topic or problem, then press **Enter** to search. Use **Shift+Enter** for a new line. Results show paper links, recommendation reasons, numbered quotations, and expandable abstracts. The input clears when submitted and is restored if the search fails.

The UI keeps only the current search results in memory. Refreshing clears them; there is no history panel. Each query is independent and does not include earlier requests. Example suggestions fill the input for editing before submission.

This is a local demonstration server bound to `127.0.0.1`. It runs one search at a time and reuses the pipeline on a dedicated worker thread so SQLite connections remain on their owning thread. The page handles loading, empty results, busy responses, and errors. It can load without API configuration, but real searches require dependencies and credentials.

### Frontend and API

| File | Responsibility |
| --- | --- |
| `src/arxiv_paper_recommend/web/index.html` | Browser layout, CSS, input handling, and result rendering |
| `src/arxiv_paper_recommend/web/__main__.py` | Local HTTP server and adapter to `PaperRecommender` |

The frontend submits:

```http
POST /api/recommend
Content-Type: application/json

{"request": "My RAG system retrieves relevant documents but still gives inaccurate answers. Find related papers."}
```

The endpoint returns the pipeline result described below. It accepts a nonempty request of up to 4,000 characters. Responses include HTTP 400 for invalid input, 409 when another search is running, 503 for missing dependencies or configuration, and 502 for other backend failures.

The current API is synchronous: results arrive when retrieval and generation finish. There is no streamed output or overall synthesis field. Reasons retain their full wording, including quoted text, with citation markers hidden; matching quoted spans are highlighted and numbered inside the expandable abstract; they are **not automatically verified** against the source.

## Capabilities

The repository includes a 10,000-paper corpus snapshot, a SQLite FTS5 database, and a FAISS vector index. It supports keyword retrieval, local Qwen3 embeddings, LLM query preparation, and two-stage recommendation generation.

| Retrieval method | Behavior | Score direction |
| --- | --- | --- |
| `BM25` | Independently scores titles and abstracts, then adds scores with equal weights | Lower is better |
| `TITLE_WEIGHTED_BM25` | Weights the title score before adding the abstract score; engine default is 1:1 | Lower is better |
| `HNSW` | Approximate dense retrieval for a single query | Higher is better |
| `MULTI_QUERY_HNSW` | Merges multiple query results and reranks by the main query | Higher is better |
| `FC_HNSW` | Blends main-query similarity with coverage across query facets | Higher is better |
| `HRRF` | Reranks sparse and dense candidates, then fuses their rankings | Higher is better |
| `SEMANTIC_RRF` | Weights reciprocal-rank fusion by semantic similarity | Higher is better |
| `SD_CASCADE` | Recalls with BM25, densely scores the candidate pool, and returns the top results | Higher is better |
| `HPRF` | Uses percentile-based reranking within each retrieval branch and fuses their rank percentiles | Higher is better |

Scores from different methods are not directly comparable and are not confidence percentages. `SearchEngine.search()` defaults to `FC_HNSW`; the full `PaperRecommender` pipeline explicitly uses `SD_CASCADE`.

Sparse retrieval uses separate title and abstract FTS5 indexes, each with its own length statistics and IDF. The total score is `title_weight * title BM25 + abstract BM25`.

Formal relevance and generation-quality evaluation remains separate from functional checks. See the linked project and evaluation documents for outstanding coursework deliverables.

## Data and model setup

The repository includes matching `data/documents.db` and `data/documents.faiss` files. Rebuild them from the archived snapshot when needed:

```powershell
uv run python -m arxiv_paper_recommend.corpus.build_database
uv run python -m arxiv_paper_recommend.corpus.build_index
```

To upgrade an older combined-field FTS5 database, run `build_database`. It preserves existing integer document IDs while rebuilding separate title and abstract indexes. If document content and IDs are unchanged, this FTS-only upgrade does not require rebuilding FAISS. Online retrievers open the database read-only and do not perform migrations automatically.

The embedding model is a pinned revision of `Qwen/Qwen3-Embedding-0.6B`. The default loader uses cached Hugging Face files first and anonymously downloads missing files without using the machine's Hugging Face login. The first download is approximately 1.2 GB. Once cached, embedding can run offline; LLM query preparation and recommendation generation still require the configured API service. Custom model sources retain their own loading and authentication behavior.

## Full recommendation pipeline

```python
import json
from arxiv_paper_recommend import PaperRecommender

with PaperRecommender() as recommender:
    result = recommender.recommend(
        "How can retrieval improve language model reasoning?",
        candidate_limit=30,
        limit=10,
    )

print(json.dumps(result, ensure_ascii=False, indent=2))
```

The pipeline prepares queries, retrieves candidates with SD Cascade, assesses each paper independently, and ranks the approved papers using `EvidenceGroundedRecommendationGenerator`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `request` | Required | Research request; must remain nonempty after trimming |
| `candidate_limit` | `30` | Integer from 1 to 30; maximum retrieved papers passed to generation |
| `limit` | `10` | Integer from 1 to 10; maximum final recommendations |
| `max_terms` | `12` | Integer from 5 to 20; maximum generated keywords |
| `max_queries` | `4` | Positive integer; maximum generated query facets |

Boolean values are rejected for integer parameters. All parameters are validated before creating resources or searching.

The returned dictionary has this structure; nullable Python values become `null` in JSON:

```text
{
    "request": str,
    "recommendations": [
        {
            "rank": int,
            "paper": {
                "document_id": int,
                "arxiv_id": str,
                "title": str,
                "abstract": str | None,
                "score": float | None
            },
            "reason": str
        }
    ]
}
```

Recommendations are ordered by final rank, starting at 1. The paper score retains its retrieval value. The reason retains generated `[[cite]]...[[/cite]]` markers. Only final recommendations are returned, not all candidates or rejected assessments.

If retrieval or screening yields no approved papers, the list is empty. Individual assessments that exhaust their retries are logged and skipped. A single approved paper is returned directly; otherwise a ranking call selects `min(limit, approved_count)` papers. Input, initialization, retrieval, query-generation, and final-ranking errors propagate to the caller.

Reuse a `PaperRecommender` across queries to reuse resources. A context manager or `close()` closes resources it created. Injected engines and generators remain caller-owned. Closing is idempotent; a closed recommender cannot be reused.

## Direct retrieval

Direct BM25 requires neither an LLM configuration nor embedding dependencies; `uv sync` is sufficient:

```python
from arxiv_paper_recommend.retrieval.sparse import BM25Retriever

with BM25Retriever() as retriever:
    papers = retriever.search(
        ["retrieval augmented generation", "reasoning"], limit=10,
    )
    for paper in papers:
        print(paper.arxiv_id, paper.title, paper.score)
```

Dense retrieval requires embedding dependencies but can also be called without an LLM:

```python
from arxiv_paper_recommend.retrieval.innovation import FCHNSWRetriever

with FCHNSWRetriever() as retriever:
    papers = retriever.search(
        "retrieval for language model reasoning",
        limit=10,
        recall_queries=["evidence retrieval", "multi-step reasoning"],
    )
```

FC scoring combines main-query inner product and the geometric mean of nonnegative facet inner products, with `coverage_weight=0.5` by default. `search()`, `score()`, and `recall()` use this score. Use `raw_score()` for main-query inner products and `raw_recall()` for the original multi-query candidate union. See [Retrieval API](docs/retrieval.md) for details.

## Separate retrieval and generation

```python
from arxiv_paper_recommend.recommendation import RecommendationGenerator
from arxiv_paper_recommend.retrieval import SearchEngine

request = "How can retrieval improve language model reasoning?"
with SearchEngine() as engine, RecommendationGenerator() as generator:
    papers = engine.search(request, limit=30)  # Defaults to FC_HNSW
    recommendations = generator.generate(request, papers, limit=10)
    for item in recommendations:
        print(item.rank, item.paper.arxiv_id, item.reason)
```

Method names are case-insensitive. Explicit `HNSW` and `MULTI_QUERY_HNSW` requests retain their baseline algorithms. Multi-query dense and hybrid methods prepare keywords, query facets, and a main query in one structured LLM call with high reasoning effort. BM25 and single-query HNSW use their dedicated preparation methods.

HRRF and HPRF accept three weights between 0 and 1, each defaulting to 0.5:

```python
with SearchEngine() as engine:
    papers = engine.search(
        request,
        method="HRRF",  # HPRF accepts the same weights
        limit=30,
        sparse_semantic_weight=0.5,
        dense_semantic_weight=0.5,
        fusion_weight=0.5,
    )
```

These control semantic weighting within sparse candidates, semantic weighting within dense candidates, and the dense branch's final fusion weight. The low-level retrievers expose the same weights. The former `FACET_COVERAGE` entry point has been removed; see [Migration](docs/migration.md).

Both recommendation generators deduplicate the first 30 candidates and assess papers with at most five concurrent calls. Assessments use no reasoning effort and temperature 0. If at least two papers are approved, a separate ranking call uses high reasoning effort and omits temperature. Reasons are retained from screening.

The evidence-grounded variant asks the model to support paper-specific claims with quoted source spans. Both variants return `PaperRecommendation`; there is no separate `evidence` field. The code checks structure, nonempty reasons, and ranking validity, but does not validate citation markers, quotation matches, or semantic support. These require separate evaluation.

Unified query preparation allows five retries (six attempts total). Dedicated keyword/query generation, individual assessments, and ranking allow three retries (four attempts total). API and validation failures share each call's retry budget; SDK retries are disabled. Validation errors trigger regeneration with feedback; API errors use exponential backoff. Individual assessment failures are skipped, while other exhausted failures propagate.

## Documentation

The detailed documents below are currently primarily in Chinese.

| Document | Contents |
| --- | --- |
| [Project structure](structure.md) | Module responsibilities, dependencies, and data flow |
| [Retrieval API](docs/retrieval.md) | Retrieval methods, scores, and resource management |
| [Recommendation generation](docs/recommendation.md) | Generation interfaces and validation |
| [Evidence-grounded generation](docs/evidence_grounded_generation.md) | Evidence prompts, citation format, and validation boundaries |
| [Vectorization](docs/vectorization.md) | Model, device, encoding, and query instructions |
| [Data and index construction](docs/data.md) | Sources, statistics, storage, and offline construction |
| [Migration](docs/migration.md) | Breaking interface changes |
| [Project goals and progress](docs/project.md) | Coursework requirements and remaining deliverables |
| [Evaluation](docs/evaluation.md) | Annotation, metrics, and experiment plans |

[Example queries](data/example_queries.json) contains 40 bilingual research requests across ten topics, without human relevance labels.
