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
from arxiv_paper_recommend.retrieval.innovation import FacetCoverageRetriever
from arxiv_paper_recommend.retrieval.query import QueryGenerator
```

底层检索器不调用 LLM。稀疏检索需要数据库；稠密和混合检索还需要向量索引及 `vectorization` 可选依赖。`SearchEngine` 从自然语言生成检索输入，需要 LLM 配置。

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

BM25 越小越相关；内积、RRF 和语义加权 RRF 越大越相关，分数不能跨方法直接比较。归一化向量的内积对应余弦相似度。

共享重排将不可评分结果放在末尾，同分按整数 ID 升序；基础 BM25 和 HNSW 召回保留 SQLite、FAISS 的返回顺序。多查询及混合检索创建新的结果对象，不改写底层返回对象。

结果不含作者、日期、来源 URL、推荐理由或独立引用证据。存储字段及映射见[论文与数据库格式](data.md#论文与数据库格式)。

## 稀疏检索

### BM25Retriever

```text
BM25Retriever(database_path=DATABASE_PATH)
search(query: list[str], limit=20)
score(query: list[str], document_ids: list[int])
```

连接以只读模式打开。关键词规范化空白并转为小写，按首次出现顺序去重，忽略空白项；处理后需有 1～20 项。每项必须含 FTS5 可索引字符，短语与双引号会正确转义，使用 OR 匹配。

`search` 要求正整数 `limit`，转义后的查询最多 1,347 字符，结果按原始 BM25 升序排列。`score` 不受搜索查询长度及结果数量限制，仍要求 1～20 个有效关键词；现存但不匹配的论文为 `0.0`，不存在的 ID 为 `None`。评分查询按 SQLite 参数上限分批执行。

### TitleWeightedBM25Retriever

继承 BM25，沿用构造器，两个方法均增加仅限关键字参数 `title_weight: float | None = None`。省略或传入 `None` 时使用普通 BM25；显式权重仅影响本次调用，摘要权重固定为 `1.0`，论文 ID 不参与全文评分。

```python
from arxiv_paper_recommend.retrieval.sparse import TitleWeightedBM25Retriever

terms = ["retrieval augmented generation", "reasoning"]
with TitleWeightedBM25Retriever() as retriever:
    papers = retriever.search(terms, limit=10, title_weight=3.0)
    scores = retriever.score(
        terms, [paper.document_id for paper in papers], title_weight=3.0,
    )
```

权重接受有限非负整数或浮点数，包括零；布尔值和其他类型报 `TypeError`，负数、非有限值或超出浮点范围的数值报 `ValueError`。搜索与补评分应传入同一权重。

引擎的标题加权方法使用 `3:1`；该设置仍待评估，不能据此认定优于普通 BM25。

## 稠密检索

### HNSWRetriever

```text
HNSWRetriever(index_path=INDEX_PATH, database_path=DATABASE_PATH, *, vectorizer=None)
search(query: str, limit=20)
score(query: str, document_ids: list[int])
```

加载 FAISS 索引并打开只读数据库连接，向量化器在首次需要编码时创建；可注入复用实例。查询必须为非空白字符串，`limit` 为正整数，最多召回索引中的向量数；空索引返回空列表。

搜索执行近似召回，评分计算指定 ID 的精确内积。评分要求内积度量的 `IndexIDMap2`，只重建同时存在于索引和数据库的向量，其余 ID 为 `None`；无可用 ID 时不编码查询。查询向量需与索引维度一致，归一化前复制，避免改写注入向量化器的输出。

索引在实例生命周期内应保持不变；更新文件后重新创建检索器。

### MultiQueryHNSWRetriever

继承 HNSW，保留构造器和评分接口：

```text
search(query: str, limit=20, *, recall_queries: Sequence[str] | None = None)
```

`recall_queries` 使用列表、元组等字符串序列，拒绝裸字符串和集合；元素必须为非空白字符串，按首次出现顺序去重。

- `None`：沿用父类单查询召回。
- 非空序列：每条查询最多召回 `limit` 篇，合并并去重后按主查询 `query` 精确评分，返回前 `limit` 篇。
- 空序列：返回空列表，不回退主查询。

多查询在类内批量编码和召回；无有效候选时直接返回空列表。最终分数为主查询精确内积。

## 混合检索

三种实现共享构造器和搜索签名：

```text
HybridRetriever(sparse_retriever=None, dense_retriever=None)
search(
    terms: list[str], query: str, limit=20, *,
    title_weight: float | None = None,
    recall_queries: Sequence[str] | None = None,
) -> list[SearchResult]
```

默认创建 `TitleWeightedBM25Retriever` 和 `MultiQueryHNSWRetriever`，也可注入复用实例。`limit` 为 1～100。`title_weight` 同时用于稀疏召回和 HRRF 补评分，省略时为普通 BM25。

`query` 用于最终语义重排；`recall_queries` 指定稠密召回或候选内筛选查询，校验与多查询 HNSW 一致。`None` 使用主查询；空序列下 HRRF、SemanticRRF 仅保留稀疏候选，SD Cascade 返回空列表。所有方法候选不足时返回较少结果。

### HRRFRetriever

Hierarchical Reciprocal Rank Fusion，分层倒数排名融合：

1. 稀疏侧召回最多 `limit` 篇；稠密侧各查询召回、合并后，按主查询内积保留前 `limit` 篇。
2. 在每路候选内分别按 BM25 升序和主查询内积降序排名，通过等权 RRF 得到该路新排序；缺少的稀疏分数通过公开评分接口补齐。
3. 对两份新排序再执行一次等权 RRF，取前 `limit` 篇。

两阶段均使用一基名次及常量 60：

```text
RRF(d) = Σ 1 / (60 + rank(d))
```

缺席某路贡献为零，交集论文获得两路贡献；不为交集预留位置。最终分数为第二阶段 RRF。

### SemanticRRFRetriever

Semantic-Weighted Reciprocal Rank Fusion，语义加权倒数排名融合。稀疏侧最多召回 `ceil(limit / 2)` 篇；稠密侧多查询召回后按主查询内积保留前 `limit` 篇。对两路列表计算一基名次的等权 RRF，再按逐篇主查询相似度加权：

```text
score(d) = RRF(d) × (cos(query, d) + 1)
```

相似度为 `None` 时权重为零。直接按加权分数从候选并集选取前 `limit` 篇，零分候选仍参与截取，不预留交集位置。

### SDCascadeRetriever

Sparse-to-Dense Cascade，稀疏到稠密级联检索。BM25 召回最多 `limit × 60` 篇，在这些候选内按每条召回查询的内积选取前 `limit` 篇并合并，再按主查询内积重排。

缺少向量的候选不参与语义选择；本方法不执行全库 HNSW 搜索，也不计算 RRF。无稀疏候选、没有召回查询或没有可评分候选时返回空列表。

## 检索创新：FacetCoverageRetriever

FC-HRRF 先用现有 HRRF 取得固定深度候选，再以每个召回查询作为一个语义分面，对
同一批候选计算精确内积排名。设 HRRF 排名百分位为 `P_H`，各分面排名百分位为
`P_j`，最终分数为：

```text
coverage(d) = min_j P_j(d)
score(d) = (1 - coverage_weight) * P_H(d) + coverage_weight * coverage(d)
```

缺少某一分面向量分数时，该分面的百分位为零。最小值使只匹配部分需求的论文受到
惩罚；`coverage_weight=0` 保留原 HRRF 顺序。接口沿用混合检索参数，并增加：

```text
candidate_limit: int | None = None  # 默认 max(50, limit)，最大 100
coverage_weight: float = 0.5        # 有限数值，范围 [0, 1]
```

默认覆盖权重 `0.5` 来自示例查询代理实验的开发集选择，不代表已经过人工相关性验证。
显式空 `recall_queries` 没有覆盖信号，此时保留 HRRF 顺序。检索器复用注入的稀疏、
稠密组件，不拥有这些资源。示例查询上的四方法代理实验见
[实验说明](../experiments/README.md)。

## SearchEngine

```text
SearchEngine(*, query_generator=None, sparse_retriever=None, dense_retriever=None)
search(request: str, method: str, limit=20, *, max_terms=12, max_queries=4)
```

引擎共享标题加权稀疏组件和多查询稠密组件，按需创建并跨调用复用。

| 方法 | 使用的数量参数 | limit | 每次搜索的 LLM 调用 |
| --- | --- | --- | --- |
| `BM25` | max_terms：5～20 | 正整数 | 1 次关键词生成 |
| `TITLE_WEIGHTED_BM25` | max_terms：5～20 | 正整数 | 1 次关键词生成 |
| `HNSW` | 无 | 正整数 | 1 次主查询生成 |
| `MULTI_QUERY_HNSW` | max_queries：正整数 | 正整数 | 召回查询、主查询各 1 次 |
| `HRRF` | max_terms：5～20；max_queries：正整数 | 1～100 | 关键词、召回查询、主查询各 1 次 |
| `SEMANTIC_RRF` | 同上 | 1～100 | 同上 |
| `SD_CASCADE` | 同上 | 1～100 | 同上 |
| `FACET_COVERAGE` | 同上 | 1～100 | 同上 |

多查询和混合路径在检索前完成上述生成，空召回不减少调用次数。方法名去掉首尾空白后转为大写；未知方法、空白请求及当前方法使用的无效数量参数在调用 LLM 前报错。数量参数拒绝布尔值和非整数，未使用的参数不参与校验。数据库、模型和生成异常直接传播。

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

两个方法各调用一次 LLM，生成 1 至指定上限的英文检索输入。`max_terms` 至少为 5，独立生成器没有上界；接入引擎的稀疏路径限制为 20。`max_queries` 为正整数，设为 1 时使用整合主题与约束的专用提示词。结构化模型按类型和上限缓存，最多 32 项，不缓存 LLM 响应；返回内容中的空白项报错。

生成器与推荐生成器共同使用 `StructuredGenerator`，配置在构造时读取：进程环境变量优先于根目录 `.env`，不修改进程环境。显式 `model` 覆盖配置中的模型；注入 `client` 时不创建或接管客户端。

| 环境变量 | 用途 |
| --- | --- |
| `LLM_API_KEY` | 客户端 API 凭证 |
| `LLM_API_URL` | 客户端 `base_url` |
| `LLM_MODEL` | 非空模型名称，构造时校验 |

当前调用为 `OpenAI.responses.parse`，传入 Pydantic `text_format`、`reasoning={"effort": "none"}` 和 `temperature=0`。服务与模型必须支持该组合；没有自动切换其他生成接口的逻辑。

查询生成不接收检索结果，也不生成推荐理由。自然语言日期、分类等约束只保留在提示词中，不构成元数据硬过滤。

## 资源生命周期

检索器、查询生成器和引擎支持 `with` 或手动 `close()`。关闭幂等，关闭后的实例不能继续操作或再次进入上下文。

自建连接与客户端由所属对象关闭，注入组件由调用方关闭。混合检索器初始化失败会释放已创建的资源；引擎的稀疏路径不会创建稠密组件或加载向量模型。模型和索引随引用生命周期释放，不强制清理全局 CUDA 缓存。

这些接口和算法尚未经过正式效果评估，评估方案见[实验与评估](evaluation.md)。
