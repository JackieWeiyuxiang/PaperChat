# 接口迁移

本文记录不兼容变更。当前用法见[检索 API](retrieval.md)、[推荐生成](recommendation.md)和[数据与构建](data.md)。除明确列出的兼容名称外，旧名称不提供别名或转发模块。

## 分面搜索深度按分面数调整（2026-10-31）

`MultiQueryHNSWRetriever.search()` 与 `FCHNSWRetriever.search()` 的默认每分面召回深度
统一为 `ceil(limit * sqrt(m))`，其中 `m` 是去重后的分面数，最终仍最多返回 `limit` 篇。
单查询 HNSW 对应 `m=1`，空分面序列仍返回空列表，实际深度不超过索引文档数。
FC-HNSW 显式指定的 `candidate_limit` 仍直接作为每分面深度；`recall()`、`raw_recall()`
继续使用显式传入的每分面 `limit`，不再次缩放。混合检索默认复用新的搜索预算。
无需重建数据库或向量索引，本次未执行测试或实验。

## HRRF 支持三层权重（2026-10-31）

`HRRFRetriever.search()` 和 `SearchEngine.search(..., method="HRRF")` 现在接受与 HPRF
相同的三个仅限关键字参数：`sparse_semantic_weight=0.5`、`dense_semantic_weight=0.5`、
`fusion_weight=0.5`，范围及校验规则均相同。两路内部及最终融合分别使用
`(1 - w) / (60 + rank_A) + w / (60 + rank_B)`，候选预算与补评分流程保持不变。

默认排序与此前等权相加版本一致，最终分数为原来的一半；依赖绝对分数阈值的调用方需调整。
无需重建数据库或向量索引。本次未执行测试或实验。

## HPRF 替换旧 Facet Coverage 混合检索（2026-10-31）

删除 `retrieval.innovation.FacetCoverageRetriever`、其 `facet_coverage` 模块和
`FACET_COVERAGE` 引擎方法，不提供兼容别名。底层调用改用
`from arxiv_paper_recommend.retrieval.innovation import HPRFRetriever`，引擎调用改为
`engine.search(request, method="HPRF", ...)`。

新方法保留 HRRF 的两路候选预算，每路最多 `limit` 篇，不再接受旧混合检索器的
`candidate_limit` 或 `coverage_weight`。三处融合均使用排名百分位加权，三个仅限
关键字的参数为 `sparse_semantic_weight=0.5`、`dense_semantic_weight=0.5` 和
`fusion_weight=0.5`，分别控制稀疏候选内语义占比、稠密候选内语义占比和最终稠密路占比。
它们独立接受 `[0, 1]` 内有限整数或浮点数，拒绝布尔值；旧混合算法的参数不可直接映射。

HPRF 实现位于 `retrieval.innovation.hprf`，由 `retrieval.innovation` 延迟导出。
此前使用 `retrieval.hybrid.HPRFRetriever` 或 `retrieval.hybrid.hprf` 的调用方需更新导入路径，
旧路径不保留别名；算法、搜索参数及 `HPRF` 引擎方法名保持不变。

`SearchEngine.search()` 同样接受上述三个权重，用于 `HRRF` 和 `HPRF`，在调用 LLM 前校验；
其他方法忽略这些参数，自定义 `SearchHandler` 签名保持不变。独立 FC-HNSW 的
`coverage_weight`、`candidate_limit`、默认 dense 地位以及引擎默认 `FC_HNSW` 均保留，
HRRF 的加权接口及分数尺度见上节，无需重建数据库或向量索引。

旧实验脚本及专用测试已删除，历史结果和报告仍保存在 `experiments/results/`，
仅代表旧版 FC-HRRF 算法，不能作为 HPRF 的评估结论；本次未执行测试或实验。

## 默认 dense 切换为 FC-HNSW（2026-10-31）

新增独立检索器 `retrieval.innovation.FCHNSWRetriever`，继承原多查询 HNSW 的构造器、
`raw_score()` 和 `raw_recall()`。`search()`、`score()`、`recall()` 统一使用原始分数公式：
`coverage_weight × GM(max(0, 分面内积)) + (1 − coverage_weight) × 主查询内积`。
分面等权，任一非正或缺失分面使覆盖度为零，主查询保留正负值。默认覆盖权重为 `0.5`，
搜索每分面候选深度默认 `ceil(limit * sqrt(m))`（`m` 为去重后的分面数）；独立 FC-HNSW 不受混合检索的 100 条上限限制。

`score()` 新增仅限关键字参数 `recall_queries=None`、`coverage_weight=0.5`；
`recall()` 新增 `coverage_weight=0.5`，返回按 FC 排序的完整召回并集。
需要原始内积的调用方应从 `score()` 迁移至 `raw_score()`，需要完整原始召回并集的调用方
应从 `recall()` 迁移至 `raw_recall()`。原 HNSW、多查询 HNSW 的评分及召回行为保持不变。

`SearchEngine.search(request, method="FC_HNSW", ...)` 的 `method` 现在可以省略；
省略时生成分面查询和主查询，执行 FC-HNSW。显式 `HNSW`、`MULTI_QUERY_HNSW` 的
评分仍使用原始内积，多查询搜索的分面预算见上节。

引擎和混合检索默认创建 `FCHNSWRetriever`。HRRF、HPRF、SemanticRRF 的默认 dense 候选
因此可能变化，后续重排及融合规则不变。其原始内积通过 `raw_score()` 获取，不能复用 FC
分数。SD Cascade 仍在稀疏候选内评分。旧 FC-HRRF 混合实现已移除，替代接口见上节。

原类和导入路径保留，显式注入组件继续覆盖默认实现。需要按主查询内积选择 dense 候选时，
显式注入 `MultiQueryHNSWRetriever`；需要默认 FC 搜索时，使用新类或省略注入。
历史检索实验显式注入原多查询检索器，保存的结果仍属于旧基线。本次不重跑实验、不更新历史报告，
不需要重建数据库或向量索引。

## 标题与摘要独立 BM25（2026-10-31）

稀疏检索从双字段 `documents_fts` 改为单字段 `documents_title_fts` 与 `documents_abstract_fts`。两张表分别索引标题、摘要，沿用 `porter` 分词器和 `documents.id` 作为 rowid；字段长度统计与 IDF 分别计算。

Python 构造器、`search()`、`score()` 和 `SearchResult` 结构不变，但分数及排名会变化。总分为 `title_weight × 标题 BM25 + 摘要 BM25`；标题权重乘在字段最终分数上，取代旧的 `bm25(documents_fts, title_weight, 1.0)` 列权重。普通 BM25 和 `title_weight=None` 均为 `1:1`，引擎的 `TITLE_WEIGHTED_BM25` 默认也为 `1:1`。候选仍为任一字段匹配的并集，结果按总分升序、整数 ID 升序排列。

旧数据库必须先运行现有构建命令升级，检索器不会自动迁移：

```powershell
uv run python -m arxiv_paper_recommend.corpus.build_database
```

导入、两张索引的重建及旧索引删除在同一事务中完成。已有 `doc_id` 的整数 ID 保留，不删除源文件之外的论文。仓库内 `data/documents.db` 已完成本次迁移，论文内容及全部 ID 映射保持不变，因此无需重建 FAISS；论文表和向量归档格式未变。评分使用物化 CTE，运行环境要求支持 FTS5 的 SQLite >= 3.35。

已保存的检索代理实验结果对应旧 BM25 实现，不能作为新公式的效果或延迟结论。本次保留历史结果，不重新执行实验。

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

本节结构与命名调整未改变数据库论文表、JSONL、NPZ 的字段：存储的 `id`、`doc_id` 映射到 Python 结果的 `document_id`、`arxiv_id`；该调整本身未要求重建数据库或向量索引，也未改变检索算法、默认参数、分数尺度或查询生成调用次数。后续的全文索引结构与评分变更需另按本文开头的“标题与摘要独立 BM25”升级说明处理。

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
