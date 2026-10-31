# 检索 API

安装和入门见 [README](../README.md)，离线准备见[数据与索引构建](data.md)，模型配置见[向量化](vectorization.md)。检索结果可交给[推荐生成器](recommendation.md)；升级已有调用方见[接口迁移](migration.md)。

## 公开入口与共同接口

```python
from arxiv_paper_recommend.retrieval import Retriever, SearchEngine, SearchResult
from arxiv_paper_recommend.retrieval.sparse import BM25Retriever, TitleWeightedBM25Retriever
from arxiv_paper_recommend.retrieval.dense import HNSWRetriever, MultiQueryHNSWRetriever
from arxiv_paper_recommend.retrieval.hybrid import (
    HybridRetriever,
    HRRFRetriever,
    SemanticRRFRetriever,
    SDCascadeRetriever,
)
from arxiv_paper_recommend.retrieval.innovation import FCHNSWRetriever, HPRFRetriever
from arxiv_paper_recommend.retrieval.query import QueryGenerator
```

底层检索器不调用 LLM。稀疏检索需要数据库及支持 FTS5 的 SQLite >= 3.35；稠密和混合检索还需要向量索引及 `vectorization` 可选依赖。`SearchEngine` 从自然语言生成检索输入，需要 LLM 配置。

`Retriever[QueryT]` 定义：

```text
search(query: QueryT, limit: int = 20) -> list[SearchResult]
score(query: QueryT, document_ids: list[int]) -> dict[int, float | None]
```

BM25 使用关键词列表，HNSW 使用单个字符串。评分按 ID 首次出现顺序返回，每个 ID 只评分一次；无法评分时为 `None`。混合检索器使用独立的 `HybridRetriever`，只提供搜索接口。

仅从检索根包导入 `Retriever` 或 `SearchResult` 不会加载 FAISS、向量化器或 LLM 客户端模块；引擎延迟导出。

## 结果与分数

`SearchResult` 是数据类：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `document_id` | `int` | 数据库整数主键，也是 FAISS 向量 ID，用于评分 |
| `arxiv_id` | `str` | 去掉 `arxiv:` 前缀、保留版本号的论文标识 |
| `title` | `str` | 标题 |
| `abstract` | `str \| None` | 摘要 |
| `score` | `float \| None` | 当前方法的分数，默认 `None` |

BM25 越小越相关；内积、FC、RRF 和语义加权 RRF 越大越相关，分数不能跨方法直接比较。归一化向量的内积对应余弦相似度。FC 分数不依赖候选集合，不能当作主查询内积使用。

共享重排将不可评分结果放在末尾，同分按整数 ID 升序；BM25 召回同样以整数 ID 升序处理同分，基础 HNSW 召回保留 FAISS 的返回顺序。多查询及混合检索创建新的结果对象，不改写底层返回对象。

结果不含作者、日期、来源 URL、推荐理由或独立引用证据。存储字段及映射见[论文与数据库格式](data.md#论文与数据库格式)。

## 稀疏检索

### BM25Retriever

```text
BM25Retriever(database_path=DATABASE_PATH)
count_documents() -> int
search(query: list[str], limit=20)
score(query: list[str], document_ids: list[int])
```

连接以只读模式打开，要求已有 `documents_title_fts` 和 `documents_abstract_fts` 单字段索引；旧数据库先按[数据构建说明](data.md#构建数据库)升级，检索时不修改索引。关键词规范化空白并转为小写，按首次出现顺序去重，忽略空白项；处理后需有 1～20 项。每项必须含 FTS5 可索引字符，短语与双引号会正确转义，使用 OR 匹配。

`count_documents()` 返回当前数据库 `documents` 表的文档总数，与关键词匹配数量无关；`TitleWeightedBM25Retriever` 继承此方法。

标题与摘要分别使用 `porter` 分词器和各自索引的字段长度、平均长度及 IDF 计算 BM25，再以 `1:1` 相加。保留 SQLite BM25 的负分方向，分数越小越相关；未匹配字段贡献 `0.0`。

`search` 要求正整数 `limit`，转义后的查询最多 1,347 字符。候选为标题或摘要匹配的并集，通过物化 CTE 计算两侧分数，合并加总后按总分升序、整数 ID 升序排列，最后应用 `limit`；不会先对各字段单独截取候选。

`score` 使用相同加总公式，不受搜索查询长度及结果数量限制，仍要求 1～20 个有效关键词；现存但不匹配的论文为 `0.0`，不存在的 ID 为 `None`。评分查询按 SQLite 参数上限分批执行，两路查询复用命名绑定参数。

### TitleWeightedBM25Retriever

继承 BM25，沿用构造器，两个方法均增加仅限关键字参数 `title_weight: float | None = None`。总分为 `title_weight × 标题 BM25 + 摘要 BM25`：权重乘在标题侧完成 BM25 计算后的最终分数上，不作为 `bm25()` 的列权重传入。省略或传入 `None` 等价于 `1.0`，与普通 BM25 完全相同；显式权重仅影响本次调用，摘要权重固定为 `1.0`，论文 ID 不参与全文评分。

```python
from arxiv_paper_recommend.retrieval.sparse import TitleWeightedBM25Retriever

terms = ["retrieval augmented generation", "reasoning"]
with TitleWeightedBM25Retriever() as retriever:
    papers = retriever.search(terms, limit=10, title_weight=3.0)
    scores = retriever.score(
        terms, [paper.document_id for paper in papers], title_weight=3.0,
    )
```

权重接受有限非负整数或浮点数，包括零；布尔值和其他类型报 `TypeError`，负数、非有限值或超出浮点范围的数值报 `ValueError`。权重为零时仍保留标题匹配候选，其标题贡献为零。搜索与补评分应传入同一权重。

引擎的标题加权方法默认使用 `1:1`，与普通 BM25 一致；上例中的 `3.0` 仅演示显式加权，不代表经过效果评估。混合检索的稀疏召回和补评分也使用同一字段加总公式。

## 稠密检索

### HNSWRetriever

```text
HNSWRetriever(index_path=INDEX_PATH, database_path=DATABASE_PATH, *, vectorizer=None)
search(query: str, limit=20)
score(query: str, document_ids: list[int])
raw_score(query: str, document_ids: list[int])
```

加载 FAISS 索引并打开只读数据库连接，向量化器在首次需要编码时创建；可注入复用实例。查询必须为非空白字符串，`limit` 为正整数，最多召回索引中的向量数；空索引返回空列表。

搜索执行近似召回，`score()` 委托 `raw_score()` 计算指定 ID 的精确内积；子类可覆盖
`score()` 实现自身公式，`raw_score()` 仍提供原始内积。评分要求内积度量的 `IndexIDMap2`，
只重建同时存在于索引和数据库的向量，其余 ID 为 `None`；无可用 ID 时不编码查询。
查询向量需与索引维度一致，归一化前复制，避免改写注入向量化器的输出。

索引在实例生命周期内应保持不变；更新文件后重新创建检索器。

### MultiQueryHNSWRetriever

继承 HNSW，保留构造器和评分接口：

```text
search(query: str, limit=20, *, recall_queries: Sequence[str] | None = None)
recall(query: str, limit=20, *, recall_queries: Sequence[str] | None = None)
raw_recall(query: str, limit=20, *, recall_queries: Sequence[str] | None = None)
```

`recall_queries` 使用列表、元组等字符串序列，拒绝裸字符串和集合；元素必须为非空白字符串，按首次出现顺序去重。

- `None`：沿用父类单查询召回。
- 非空序列：令 `m` 为去重后的分面数，每条查询最多召回 `ceil(limit * sqrt(m))` 篇，合并并去重后按主查询 `query` 精确评分，返回前 `limit` 篇。实际召回深度不超过索引文档数。
- 空序列：返回空列表，不回退主查询。

多查询在类内批量编码和召回；无有效候选时直接返回空列表。最终分数为主查询精确内积。

`recall()` 使用相同的输入校验、批量召回和去重逻辑，但返回完整召回并集，不将合并结果
截取为 `limit` 篇。这里的 `limit` 仅限制每条召回查询的邻居数量；所有候选按主查询的
精确内积降序、文档 ID 升序返回，分数为主查询精确内积。`recall_queries=None` 使用
主查询作为唯一召回查询，显式空序列仍返回空列表。`recall()` 与 `raw_recall()` 的
`limit` 是显式的每分面深度，不额外乘以分面数的平方根。
`recall()` 委托 `raw_recall()`，底层多查询召回使用 `raw_score()`，使显式调用原类搜索
实现时仍获得原始内积。`raw_recall()` 也供 FC 子类及混合检索获取完整原始召回并集。

### FCHNSWRetriever（默认 dense）

位于 `retrieval.innovation`，继承 `MultiQueryHNSWRetriever`，保留构造器，
`search()`、`score()` 和 `recall()` 统一使用原始相似度 FC 公式：

```text
FCHNSWRetriever(index_path=INDEX_PATH, database_path=DATABASE_PATH, *, vectorizer=None)
search(
    query: str, limit=20, *,
    recall_queries: Sequence[str] | None = None,
    candidate_limit: int | None = None,
    coverage_weight: float = 0.5,
)
score(
    query: str, document_ids: list[int], *,
    recall_queries: Sequence[str] | None = None,
    coverage_weight: float = 0.5,
)
recall(
    query: str, limit=20, *,
    recall_queries: Sequence[str] | None = None,
    coverage_weight: float = 0.5,
)
```

`search()` 每条召回查询取最多 `candidate_limit` 篇，对完整去重并集计算 FC，
按 FC 降序排列后截取 `limit` 篇。令 `m` 为去重后的分面数，`candidate_limit` 默认
`ceil(limit * sqrt(m))`，向上取整；未提供分面时 `m=1`，空分面序列返回空列表。
显式传入的 `candidate_limit` 直接覆盖每分面深度，不额外乘以平方根，
要求为不小于 `limit` 的正整数；独立 dense 的 `limit` 和 `candidate_limit` 均无 100 条上限。
实际召回深度不超过索引文档数。`recall()` 使用同一评分流程，但其 `limit` 直接限制
每条分面的召回数量，不额外乘以平方根，返回完整去重并集。
`score()` 为指定 ID 返回相同公式的分数，键按 ID 首次出现顺序保留，不执行召回。

`query` 是 score query，`recall_queries` 是 fc queries。令 `q(d)` 为主查询精确内积，
`s_j(d)` 为第 `j` 个分面的精确内积：

```text
coverage(d) = GM(max(0, s_1(d)), ..., max(0, s_m(d)))
FC(d) = (1 - coverage_weight) * q(d) + coverage_weight * coverage(d)
```

各分面等权；任一分面非正或缺失时覆盖度为零，均为正时使用
`exp(mean(log(s_j(d))))` 计算几何平均，主查询内积保留正负值。缺失主查询分数时，
主查询项贡献为零；全部参与加权的评分均不可用时返回 `None`。不存在于索引或数据库的
ID 返回 `None`。分数与候选集合大小无关；三个接口使用相同参数时，同一文档分数一致。
搜索和召回同分按整数 ID 升序，`None` 排末尾。
`coverage_weight` 接受 `[0, 1]` 内有限整数或浮点数，拒绝布尔值和其他类型；默认 `0.5`
尚未通过新实验调参。权重为零时仅使用主查询原始内积，跳过额外分面评分；权重为一时
仅使用分面几何平均，主查询不作为独立加权项。

`recall_queries` 沿用多查询检索的序列校验与去重规则。`None` 使用主查询作为唯一分面，
此时主查询内积非负时 FC 等于该内积，为负时分面贡献为零，FC 为
`(1 - coverage_weight) × q(d)`。显式空序列时，`search()`、`recall()` 返回空列表，
`score()` 仅返回主查询原始内积；无候选同样返回空列表。该检索器不调用 LLM。

继承的 `raw_score(query, document_ids)` 返回指定 ID 的主查询精确内积，
`raw_recall(query, limit=20, *, recall_queries=None)` 返回完整原始召回并集及主查询
精确内积，按主查询内积降序、文档 ID 升序排列。这两个入口供混合检索补评分和重新计算
候选集合内的百分位使用，均不计算 FC。

## 混合检索

三种基础混合实现及 `retrieval.innovation` 中的 HPRF 共享构造器和基本搜索签名；HRRF、HPRF 另有三个权重参数：

```text
HybridRetriever(sparse_retriever=None, dense_retriever=None)
search(
    terms: list[str], query: str, limit=20, *,
    title_weight: float | None = None,
    recall_queries: Sequence[str] | None = None,
) -> list[SearchResult]
```

默认创建 `TitleWeightedBM25Retriever` 和 `FCHNSWRetriever`，也可注入复用实例，包括原 `MultiQueryHNSWRetriever`。`limit` 为 1～100。`title_weight` 同时用于稀疏召回和 HRRF、HPRF 补评分，省略时为普通 BM25。

HRRF、HPRF、SemanticRRF 默认用 FC-HNSW 的 `search()` 选择 dense 候选，每分面深度为
`ceil(limit * sqrt(m))`（`m` 为去重后的分面数），保留 `limit` 篇；后续按各自算法重排及融合。原始内积统一通过
`raw_score()` 获取并在单次搜索内缓存，不把 FC 分数用作内积。显式注入组件覆盖默认实现，
因此默认组件和分面预算调整均可能改变混合检索候选；旧实验采用当时的多查询实现与预算。

`query` 用于语义评分；`recall_queries` 指定稠密召回查询或 SD Cascade 的 FC 评分分面，校验与多查询 HNSW 一致。`None` 使用主查询；空序列下 HRRF、HPRF、SemanticRRF 仅保留稀疏候选，SD Cascade 使用主查询原始内积重排稀疏候选。所有方法候选不足时返回较少结果。

### HRRFRetriever

Hierarchical Reciprocal Rank Fusion，分层倒数排名融合：

1. 稀疏侧召回最多 `limit` 篇；稠密侧调用所选 dense 组件保留前 `limit` 篇，默认使用 FC-HNSW 选择。
2. 在每路候选内分别按 BM25 升序和主查询内积降序排名，通过加权 RRF 得到该路新排序；缺少的稀疏分数通过公开评分接口补齐。
3. 对两份新排序再执行一次加权 RRF，取前 `limit` 篇。

搜索接口在基本参数后增加三个仅限关键字的权重，与 HPRF 一致：

```text
sparse_semantic_weight: float = 0.5  # 稀疏候选内部的语义占比
dense_semantic_weight: float = 0.5   # 稠密候选内部的语义占比
fusion_weight: float = 0.5           # 最终稠密路的占比
```

两阶段均使用一基名次及常量 60，对两份排名 A、B 加权：

```text
RRF_w(d) = (1 - w) / (60 + rank_A(d)) + w / (60 + rank_B(d))
```

路内重排中 A 为 BM25 排名，B 为主查询内积排名，`w` 分别取
`sparse_semantic_weight`、`dense_semantic_weight`；最终融合中 A 为稀疏路新排序，
B 为稠密路新排序，`w` 取 `fusion_weight`。缺席某路贡献为零，交集论文获得两路贡献；
不为交集预留位置，各阶段同分按文档 ID 升序。最终分数为第二阶段加权 RRF。

权重独立接受 `[0, 1]` 内有限整数或浮点数，拒绝布尔值、其他类型及非有限值。
权重为 `0` 或 `1` 时仍保留两路召回及补评分；最终贡献为零的路所独有的候选仍在并集中，
以零分参与截取。默认三个权重均为 `0.5`，排序与此前等权相加版本一致，最终分数为原来的
一半；依赖绝对分数阈值的调用方需相应调整。

### SemanticRRFRetriever

Semantic-Weighted Reciprocal Rank Fusion，语义加权倒数排名融合。稀疏侧最多召回 `ceil(limit / 2)` 篇；稠密侧默认使用 FC-HNSW 选择前 `limit` 篇，再按主查询内积排序。对两路列表计算一基名次的等权 RRF，再按逐篇主查询相似度加权：

```text
score(d) = RRF(d) × (cos(query, d) + 1)
```

相似度为 `None` 时权重为零。直接按加权分数从候选并集选取前 `limit` 篇，零分候选仍参与截取，不预留交集位置。

### SDCascadeRetriever

Sparse-to-Dense Cascade，稀疏到稠密级联检索：

1. BM25 按 `max(1, ceil(sqrt(limit × N)))` 的预算召回候选，并按文档 ID 去重。`N` 通过稀疏检索器的 `count_documents()` 获取，为整个 `documents` 表的文档数，与关键词匹配数量及向量索引大小无关。
2. 对全部稀疏候选调用稠密组件的 `score()`。默认 FC-HNSW 接收校验、去重后的 `recall_queries`，使用主查询内积与分面覆盖度的加权分数；注入原 HNSW 或多查询 HNSW 时使用其 `score(query, document_ids)`，得到主查询原始内积。
3. 排除评分为 `None` 的候选，按稠密分数降序、文档 ID 升序排列，最后返回前 `limit` 篇。`SearchResult.score` 为稠密评分，论文信息沿用稀疏召回结果。

`recall_queries=None` 使用主查询作为唯一 FC 分面；显式空序列沿用 FC-HNSW 的 `score()` 行为，仅计算主查询原始内积。缺少向量的候选不参与排序；无稀疏候选或没有可评分候选时返回空列表。本方法不执行全库 HNSW 搜索，也不计算 RRF。

## HPRFRetriever

位于 `retrieval.innovation` 的混合检索创新，继承 `HybridRetriever`。
Hierarchical Percentile Rank Fusion，分层排名百分位融合。候选选择与 HRRF 一致，
将其三处 RRF 替换为排名百分位加权：

1. 稀疏侧召回最多 `limit` 篇，稠密侧通过所选 dense 组件保留最多 `limit` 篇。
2. 为两路候选补齐 BM25 和主查询原始内积评分，在各路候选内分别计算 BM25 升序、
   主查询内积降序的百分位，再按该路的语义权重加权重排。
3. 将两路新排序分别转换为百分位，加权融合，在候选并集中返回前 `limit` 篇。

搜索接口在基本参数后增加三个仅限关键字的权重：

```text
sparse_semantic_weight: float = 0.5  # 稀疏候选内部的语义占比
dense_semantic_weight: float = 0.5   # 稠密候选内部的语义占比
fusion_weight: float = 0.5           # 最终稠密路的占比
```

对当前候选集合，`N` 为实际候选数，`rank` 为一基名次；`P_BM25`、`P_query` 分别为
该集合内的 BM25、主查询原始内积百分位：

```text
P(d) = (N - rank(d) + 1) / N
sparse_score(d) = (1 - sparse_semantic_weight) * P_BM25_in_sparse(d)
                  + sparse_semantic_weight * P_query_in_sparse(d)
dense_score(d) = (1 - dense_semantic_weight) * P_BM25_in_dense(d)
                 + dense_semantic_weight * P_query_in_dense(d)
score(d) = (1 - fusion_weight) * P_sparse_reranked(d)
           + fusion_weight * P_dense_reranked(d)
```

最终使用两路重排后的排名百分位，不直接融合 `sparse_score` 和 `dense_score`。
每次百分位转换均使用所在候选路的实际大小，不用并集大小。缺失原始评分 `None` 的信号
百分位为零，BM25 的 `0.0` 是有效评分；路内融合分数即使为零，仍参与该路的新排名。
最终缺席某路的文档在该路贡献为零，交集文档获得两路贡献，各阶段同分按文档 ID 升序。

权重独立接受 `[0, 1]` 内有限整数或浮点数，拒绝布尔值、其他类型及非有限值。
权重为 `0` 或 `1` 时仍保留两路召回及补评分，只改变评分贡献；当某路贡献为零时，
其独有候选仍留在最终并集中，以零分参与截取。没有候选时返回空列表。
`recall_queries=None` 使用主查询召回；显式空序列关闭稠密召回，稀疏候选仍补齐主查询
评分并参与分层融合。候选预算沿用 HRRF，不接受 `candidate_limit`。

默认 dense 组件仍为 FC-HNSW，负责选择稠密候选；HPRF 内部的语义排名使用
`raw_score()` 的主查询原始内积，原始评分通过单次搜索内的缓存复用。
此方法替代旧 `FacetCoverageRetriever` 与 `FACET_COVERAGE` 引擎入口，不保留旧名称。
独立 FC-HNSW 和 HRRF 保留；HPRF 及 HRRF 的新权重配置尚未进行效果或性能评估。

## SearchEngine

```text
SearchEngine(*, query_generator=None, sparse_retriever=None, dense_retriever=None)
search(
    request: str, method: str = "FC_HNSW", limit=20, *,
    max_terms=12, max_queries=4,
    sparse_semantic_weight: float = 0.5,
    dense_semantic_weight: float = 0.5,
    fusion_weight: float = 0.5,
)
```

引擎默认共享标题加权稀疏组件和 FC-HNSW 稠密组件，按需创建并跨调用复用。省略 `method`
等价于指定 `FC_HNSW`。显式 `HNSW`、`MULTI_QUERY_HNSW` 分别调用原类的搜索实现，
在共享组件上保留原算法、候选预算和内积分数。显式注入的 dense 组件覆盖默认实现，
`FC_HNSW` 及混合路径使用注入组件的公开接口；注入原多查询类不会自动为其增加 FC 搜索。

| 方法 | 使用的数量参数 | limit | 每次搜索的 LLM 调用 |
| --- | --- | --- | --- |
| `BM25` | max_terms：5～20 | 正整数 | 1 次关键词生成 |
| `TITLE_WEIGHTED_BM25` | max_terms：5～20 | 正整数 | 1 次关键词生成 |
| `HNSW` | 无 | 正整数 | 1 次主查询生成 |
| `MULTI_QUERY_HNSW` | max_queries：正整数 | 正整数 | 召回查询、主查询各 1 次 |
| `FC_HNSW`（默认） | max_queries：正整数 | 正整数 | 分面查询、主查询各 1 次 |
| `HRRF` | max_terms：5～20；max_queries：正整数 | 1～100 | 关键词、召回查询、主查询各 1 次 |
| `SEMANTIC_RRF` | 同上 | 1～100 | 同上 |
| `SD_CASCADE` | 同上 | 1～100 | 同上 |
| `HPRF` | 同上 | 1～100 | 同上 |

上表为正常情况下的逻辑调用次数。关键词、召回／分面查询和主查询生成各自最多重试 3 次，含首次最多请求 4 次；网络/API 异常、结构解析、缺少结构化结果及空白项校验失败共用各自的预算，SDK 内置重试已关闭。因此计入重试后的 API 请求总数最多为表中逻辑调用次数的 4 倍。多查询和混合路径在检索前完成上述生成，空召回不减少调用次数。方法名去掉首尾空白后转为大写；未知方法、空白请求及当前方法使用的无效数量参数在调用 LLM 前报错。数量参数拒绝布尔值和非整数，未使用的参数不参与校验。数据库、模型和最终生成异常直接传播。

三个权重用于 `HRRF` 和 `HPRF`，在创建查询生成器及调用 LLM 前完成校验；其他方法忽略
这些参数，不对其校验或透传。权重含义与各自底层检索器一致，自定义处理器继续使用下述原有签名。

`available_methods` 返回注册顺序的名称元组。`register_method(name, handler)` 支持新增方法，拒绝空名称、重复名称和不可调用处理器。处理器签名保持：

```text
handler(request: str, limit: int, *, max_terms: int, max_queries: int) -> list[SearchResult]
```

自定义处理器负责专属参数校验和外部资源管理。

## QueryGenerator

```text
QueryGenerator(*, model=None, client=None)
generate_terms(request: str, max_terms=12) -> list[str]
generate_queries(request: str, max_queries=4) -> list[str]
```

两个方法各执行一次逻辑 LLM 调用，生成 1 至指定上限的英文检索输入。`max_terms` 至少为 5，独立生成器没有上界；接入引擎的稀疏路径限制为 20。`max_queries` 为正整数，设为 1 时使用整合主题与约束的专用提示词。结构化模型按类型和上限缓存，最多 32 项，不缓存 LLM 响应；返回内容中的空白项与结构校验失败一样，附带最近一次错误反馈重新生成，共用该逻辑调用最多 4 次请求的预算。网络/API 异常保留已有校验反馈，按共享尝试序号等待 1、2、4 秒后重试；解析和业务校验失败立即重试，预算耗尽后原样抛出最后一次异常。

生成器与推荐生成器共同使用 `StructuredGenerator`，配置在构造时读取：进程环境变量优先于根目录 `.env`，不修改进程环境。显式 `model` 覆盖配置中的模型。自建客户端显式设置 `max_retries=0`；每次逻辑调用通过 `client.with_options(max_retries=0)` 获取关闭 SDK 重试的客户端视图，覆盖注入客户端的重试配置，不修改原客户端配置。注入客户端不创建新的底层连接，也不接管其资源，仍由调用方关闭。

| 环境变量 | 用途 |
| --- | --- |
| `LLM_API_KEY` | 客户端 API 凭证 |
| `LLM_API_URL` | 客户端 `base_url` |
| `LLM_MODEL` | 非空模型名称，构造时校验 |

当前调用为 `OpenAI.responses.parse`，传入 Pydantic `text_format`。查询生成和[推荐生成](recommendation.md)的逐篇判断使用 `reasoning={"effort": "none"}` 和 `temperature=0`；集中排序使用 `reasoning={"effort": "high"}`，不发送 `temperature`。共享调用层支持按调用指定 reasoning，服务与模型必须支持相应组合；没有自动切换其他生成接口的逻辑。

查询生成不接收检索结果，也不生成推荐理由。自然语言日期、分类等约束只保留在提示词中，不构成元数据硬过滤。

## 资源生命周期

检索器、查询生成器和引擎支持 `with` 或手动 `close()`。关闭幂等，关闭后的实例不能继续操作或再次进入上下文。

自建连接与客户端由所属对象关闭，注入组件由调用方关闭。混合检索器初始化失败会释放已创建的资源；引擎的稀疏路径不会创建稠密组件或加载向量模型。模型和索引随引用生命周期释放，不强制清理全局 CUDA 缓存。

这些接口和算法尚未经过正式效果评估，评估方案见[实验与评估](evaluation.md)。
