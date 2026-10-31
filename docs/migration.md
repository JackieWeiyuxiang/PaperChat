# 接口迁移

本文记录不兼容变更。当前用法见[检索 API](retrieval.md)、[推荐生成](recommendation.md)和[数据与构建](data.md)。除明确列出的兼容名称外，旧名称不提供别名或转发模块。

## 两阶段推荐与统一结果（2026-10-31）

两种推荐生成器共同实现 `BaseRecommendationGenerator`，公开 `generate(request, papers, limit=10)` 调用方式不变，统一返回 `list[PaperRecommendation]`：

- 第一阶段逐篇返回严格的 `recommended: bool` 与 `reason: str`，最多 5 个请求并发。
- `reason` 必须包含 1～2 处 `[[cite]]原文片段[[/cite]]`，两种实现都校验精确原文，包括不推荐项。
- 第二阶段只筛选排序初筛推荐项，最终数量为 `min(limit, 初筛推荐数)`，初筛全部不推荐时返回空列表。
- 所有推荐调用开启 `reasoning.effort="high"` 并省略 `temperature`；查询生成设置不变。

`EvidenceGroundedRecommendation` 保留原有导入路径，现为 `PaperRecommendation` 的别名。独立 `evidence` 字段已移除；调用方应使用包含引用标记的 `reason`，不再访问 `item.evidence` 或以旧 `evidence` 参数构造结果。生成对比脚本也统一保存带标记的 `reason`，不再写入单独的 `evidence`。

以下结构调整表保留更早的迁移记录；推荐响应校验与数量规则以本节及[推荐生成](recommendation.md)为准。

## 本轮结构与命名调整

| 重构前 | 当前接口 |
| --- | --- |
| `get_score(ids, query)` | `score(query, document_ids)`；查询参数在前 |
| `TitleAwareBM25Retriever` | `TitleWeightedBM25Retriever` |
| `DivisionalHNSWRetriever` | `MultiQueryHNSWRetriever` |
| `divisional_queries` | `recall_queries`，接受字符串序列或 `None` |
| `TITLE_BM25` | `TITLE_WEIGHTED_BM25` |
| `DIVISIONAL_HNSW` | `MULTI_QUERY_HNSW` |
| `SearchResult.id` / `.doc_id` | `.document_id` / `.arxiv_id` |
| `PaperRecommendation.result` | `.paper` |
| `recommendation.generation.PaperRecommendation` | `recommendation.PaperRecommendation` |
| `recommendation.generation.baseline_generator` | `recommendation.generation.generator` |
| `retrieval.sparse.title_aware_bm25` | `retrieval.sparse.title_weighted_bm25` |
| `retrieval.dense.divisional_hnsw` | `retrieval.dense.multi_query_hnsw` |
| `retrieval.sparse.build_database` | `corpus.build_database` |
| `retrieval.dense.build_index` | `corpus.build_index` |
| `retrieval._lifecycle` / `retrieval._validation` | `common.lifecycle` / `common.validation`；均为实现工具 |
| 配置模块的 `LLM_API_KEY`、`LLM_API_URL`、`LLM_MODEL` 常量 | 构造生成器时调用 `LLMSettings.from_env()`；同名环境变量不变 |
| 问候命令 `arxiv-paper-recommend` | 已移除；使用 Python API 或构建模块命令 |

`recall_queries` 使用列表或元组，按首次出现顺序去重。原先的单字符串应改为 `[query]`；集合应改为有确定顺序的列表。`None` 继续使用主查询，空序列表示没有召回分路。

```python
from arxiv_paper_recommend.retrieval.dense import MultiQueryHNSWRetriever

with MultiQueryHNSWRetriever() as retriever:
    papers = retriever.search(
        "retrieval for language model reasoning",
        limit=10,
        recall_queries=["evidence retrieval", "multi-step reasoning"],
    )
    scores = retriever.score(
        "retrieval for language model reasoning",
        [paper.document_id for paper in papers],
    )
```

推荐入口统一为 `from arxiv_paper_recommend.recommendation import PaperRecommendation, RecommendationGenerator`。只导入推荐模型不会加载 LLM 客户端模块。`QueryGenerator` 现在与推荐生成器一样支持 `model` 和 `client` 注入。

数据库、JSONL、NPZ 的字段不变：存储的 `id`、`doc_id` 映射到 Python 结果的 `document_id`、`arxiv_id`。现有数据库和向量索引无需重建。检索算法、默认参数、分数尺度及查询生成调用次数保持不变。

查询生成拒绝空白项。当前推荐流程检查全部初筛理由及引用，再校验集中排序结果的候选归属、去重与准确数量；不会通过截断响应跳过错误项。

## 历史迁移记录（本轮重构前）

以下保留此前版本的迁移说明，其中“当前调用”指当时版本，不是现行接口。升级到现行接口还需应用上表。


本次底层 API 直接迁移，不提供旧调用方式或参数别名：

| 原调用 | 当前调用 |
| --- | --- |
| BM25 旧关键词搜索 `term_search(terms, limit)` | `search(query=terms, limit=limit)` |
| BM25 `get_score(ids, terms=terms)` | `get_score(ids, query=terms)`；位置参数调用不变 |
| `TitleAwareBM25Retriever(title_weight=weight)` | 构造时不传权重，改为在 `search` 和 `get_score` 中分别传入 `title_weight=weight`；省略时回退普通 BM25 |
| `SearchEngine(title_aware_retriever=retriever)` | 改为 `SearchEngine(sparse_retriever=retriever)`，统一供普通 BM25、标题加权 BM25 和混合检索复用 |
| 引擎或混合检索器注入普通 `BM25Retriever` | 改为注入 `TitleAwareBM25Retriever`，支持直接透传 `title_weight=None` |
| HNSW 单查询搜索后读取 `results[query]` | `search(query, limit)` 直接返回列表，默认 limit 从 5 改为 20 |
| HNSW 分组搜索 `search(query_or_queries, limit)` | 分路召回与重排使用 `DivisionalHNSWRetriever.search(query, limit, divisional_queries=queries)` |
| HNSW `get_score(query_ids)` | 按查询调用 `get_score(ids, query)` |
| 旧公开方法 `batch_search` / `batch_get_score` | 已移除；使用 `search` 和 `get_score`，分路召回的批量计算在类内执行 |
| 引擎或混合检索器注入普通 `HNSWRetriever` | 改为注入 `DivisionalHNSWRetriever`，支持直接透传 `divisional_queries=None` |
| 引擎 `method="HNSW"` 的多查询召回与重排 | 改用 `method="DIVISIONAL_HNSW"`；`HNSW` 现为单查询召回 |

`SearchEngine.search` 的参数及返回结构保持不变，引擎将生成的分路查询通过 `divisional_queries` 传给混合检索器。三种混合检索器移除 `queries` 参数，并将 `score_query` 改名为 `query`：原调用 `search(terms, queries, score_query, limit)` 改为 `search(terms, query, limit, divisional_queries=queries)`；省略分路查询时使用 `query` 召回或筛选。标题权重通过可选且仅限关键字传入的 `title_weight` 设置。

模块文件已统一为 snake_case，旧模块路径不提供转发文件。推荐改用本页的包级导入。

| 原模块名 | 当前模块名 |
| --- | --- |
| `SearchEngine` / `SearchResult` | `search_engine` / `search_result` |
| `sparse.BM25Retriever` | `sparse.bm25` |
| `dense.HNSWRetriever` | `dense.hnsw` |
| `query.Generator` | `query.generator` |
| `vectorization.Qwen3Vectorizer` | `vectorization.qwen3_vectorizer` |

混合检索器类名统一使用 `HRRFRetriever`、`SemanticRRFRetriever` 和 `SDCascadeRetriever`，保留 `Retriever` 后缀；引擎方法标识分别为 `HRRF`、`SEMANTIC_RRF` 和 `SD_CASCADE`，不添加 `HYBRID_` 前缀。调用方需同步更新类名、模块路径和方法标识，不提供旧名称别名；推荐使用本文开头的包级导入，模块路径见[项目结构](../structure.md)。

旧有调用方无需再访问 `retriever.conn.close()`、`generator.client.close()` 或逐个关闭混合检索器的内部组件，统一使用对象的 `close()` 或上下文管理。
