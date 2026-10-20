# 检索 API

安装与快速开始见 [README](../README.md)，数据准备见[数据与构建](data.md)，模型参数见[向量化](vectorization.md)。

本文描述当前工作区的检索实现。课程最终目标还包括读取检索证据的推荐生成、引用和网页展示，这些部分尚未实现，见[项目目标与进度](project.md)。评估设计及缺口见[实验与评估](evaluation.md)；本文的算法说明不代表已验证某方法优于其他方法。

## 公开入口

```python
from arxiv_paper_recommend.retrieval import Retriever, SearchEngine, SearchResult
from arxiv_paper_recommend.retrieval.sparse import BM25Retriever
from arxiv_paper_recommend.retrieval.dense import HNSWRetriever
from arxiv_paper_recommend.retrieval.hybrid import (
    HybridRetriever,
    HRRFRetriever,
    SemanticRRFRetriever,
    SDCascadeRetriever,
)
from arxiv_paper_recommend.retrieval.query import QueryGenerator
```

底层检索器不调用 LLM。BM25 需要数据库；HNSW 和所有混合检索器还需要向量索引及 `vectorization` 可选依赖。使用 `SearchEngine` 或 `QueryGenerator` 时，需配置根目录 `.env` 中的 `LLM_API_KEY`、`LLM_API_URL`、`LLM_MODEL`。

## 共同接口

`Retriever[QueryT]` 是 sparse、dense 的泛型抽象基类，包含两个抽象方法：

```text
search(query: QueryT, limit: int = 20) -> list[SearchResult]
get_score(ids: list[int], query: QueryT) -> dict[int, float | None]
```

`BM25Retriever` 继承 `Retriever[list[str]]`，`HNSWRetriever` 继承 `Retriever[str]`。方法签名与返回结构一致，查询类型和分数含义由算法决定；评分保留 ID 首次出现的顺序，不可评分的 ID 返回 `None`。基类复用 `ManagedResource`，支持 `with` 和 `close()`。

`HybridRetriever` 是独立的混合检索抽象基类，仅规定 `search(terms, queries, score_query, limit=20)`；它不继承 `Retriever`，也不提供 `get_score`。三种混合算法保持现有候选选择和融合规则。

仅从 retrieval 根包导入 `Retriever` 或 `SearchResult` 不会加载 FAISS、向量化器或 LLM 客户端依赖；`SearchEngine` 继续延迟导出。

## 结果与分数

`SearchResult` 是数据类，字段如下：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | `int` | 内部数据库主键，也是 FAISS 向量 ID，供评分接口使用 |
| `doc_id` | `str` | 去掉 `arxiv:` 前缀、保留版本号的论文标识 |
| `title` | `str` | 标题 |
| `abstract` | `str \| None` | 摘要 |
| `score` | `float \| None` | 当前方法的分数，默认值为 `None` |

BM25 原始分数越小越相关；内积、RRF 和加权 RRF 越大越相关。不同方法的分数不具有统一尺度。归一化向量的内积对应余弦相似度。

结果不含作者、日期、来源 URL、推荐理由或引用证据；部分元数据仅在原始 JSONL 中保留。最终生成模块仍需消费这些检索结果并建立理由与来源的对应关系。

引擎重排和混合检索创建新的结果对象，不改写底层检索器返回的对象。公共排序逻辑将 `None` 放在末尾，同分按内部 ID 升序；底层 BM25、HNSW 搜索分别保留 SQLite、FAISS 的返回顺序。

## BM25Retriever

构造器：`BM25Retriever(database_path=DATABASE_PATH)`。

| 方法 | 返回值 | 行为 |
| --- | --- | --- |
| `search(query, limit=20)` | `list[SearchResult]` | 匹配任意关键词／短语，按 BM25 升序返回最多 limit 篇 |
| `get_score(ids, query)` | `dict[int, float \| None]` | 为指定内部 ID 评分，无结果数量和查询长度限制 |

`query` 是关键词字符串列表。统一连续空白、转小写、过滤空项并按首次出现顺序去重后，需有 1～20 项；每项必须含可检索字符，多词项作为连续短语匹配，内嵌双引号会转义。搜索使用单次 OR 查询，最终查询长度最多 1,347 字符。

`limit` 必须为正整数，不设数量上限。`get_score` 保留 ID 首次出现的顺序；已有但不匹配的论文为 `0.0`，不存在的 ID 为 `None`。关键词合法时，空 ID 列表返回空字典。

```python
with BM25Retriever() as retriever:
    terms = ["retrieval augmented generation", "reasoning"]
    results = retriever.search(query=terms, limit=10)
    scores = retriever.get_score([result.id for result in results], terms)
```

## HNSWRetriever

构造器：`HNSWRetriever(index_path=INDEX_PATH, database_path=DATABASE_PATH, *, vectorizer=None)`。初始化加载索引并打开只读数据库；未注入向量化器时，在首次需要编码时创建并复用默认实例。

| 方法 | 返回值 | 行为 |
| --- | --- | --- |
| `search(query, limit=20)` | `list[SearchResult]` | 对单个字符串查询执行近似召回，返回最多 limit 篇 |
| `get_score(ids, query)` | `dict[int, float \| None]` | 为单个字符串查询指定的 ID 计算精确内积 |
| `batch_search(query, limit=5)` | `dict[str, list[SearchResult]]` | 接受字符串或字符串集合，每条查询独立召回最多 limit 篇 |
| `batch_get_score(query_ids)` | `dict[str, dict[int, float \| None]]` | 为每条查询实际指定的 ID 计算精确内积 |

`search`、`get_score` 的 `query` 仅接受单个字符串，分别复用批量搜索和批量评分。`batch_search` 的 `query` 接受 `str | set[str]`，单条查询也返回以原始字符串为键的字典；集合排序后批量编码并搜索。`limit` 为正整数，不设数量上限，实际召回数不会超过索引大小。查询必须为非空白字符串；向量形状必须与索引匹配。

`batch_search` 的空集合返回 `{}`；空索引时，`search` 返回 `[]`，`batch_search` 为每条合法查询返回 `[]`，均不加载向量化模型。无效 FAISS ID 或数据库中缺失的论文会被过滤，因此结果可能少于 limit。

`batch_get_score` 的 `query_ids` 是 `dict[str, list[int]]`，空映射返回 `{}`。两种评分方法均要求 `IndexIDMap2` 内积索引，保留查询顺序及每条查询中 ID 首次出现的顺序。索引 ID 集合在首次评分时缓存；仅查询数据库 ID，批量重建有效论文向量，再执行矩阵评分。不在数据库或索引中的 ID 为 `None`；空 ID 列表或全部缺失的 ID 无需编码。索引在实例生命周期内应保持不变。

```python
query = "How can retrieval improve language model reasoning?"
with HNSWRetriever() as retriever:
    results = retriever.search(query, limit=10)
    scores = retriever.get_score([result.id for result in results], query)

    grouped_results = retriever.batch_search({query, "Evidence for multi-step reasoning"}, limit=10)
    grouped_scores = retriever.batch_get_score({
        text: [result.id for result in matches]
        for text, matches in grouped_results.items()
    })
```

## 三种混合检索器

构造器均接受 `(sparse_retriever=None, dense_retriever=None)`。未提供的组件在初始化时创建；初始化失败会关闭已创建的组件。

三种检索器共同继承 `HybridRetriever`，统一方法：

```text
search(terms, queries, score_query, limit=20) -> list[SearchResult]
```

`terms` 遵循 BM25 规则；`queries` 为非空白字符串或非空字符串集合，`score_query` 为非空白字符串；最终 `limit` 为 1～100 的整数。

```python
with HRRFRetriever() as retriever:
    results = retriever.search(
        terms=["retrieval augmented generation", "reasoning"],
        queries={
            "How can retrieval improve language model reasoning?",
            "How can retrieved evidence support multi-step reasoning?",
        },
        score_query="Retrieval methods that improve language model reasoning with evidence",
        limit=10,
    )
```

将类名替换为另外两种检索器即可使用相同输入。三种方法的每次搜索都使用独立评分缓存，只补齐缺失的查询／论文组合；已缓存的 `None` 不重复评分。

### HRRFRetriever

全名为 **Hierarchical Reciprocal Rank Fusion**（分层倒数排名融合），代码类名为 `HRRFRetriever`。

1. 稀疏侧召回最多 limit 篇；稠密侧每查询召回最多 limit 篇，合并后按评分查询相似度保留前 limit 篇。
2. 两路各自在完整候选列表内，按 BM25 升序和评分查询相似度降序分别排名，通过等权 RRF 得到该路新排序。
3. 用两份新排序的位置计算最终等权 RRF，返回前 limit 篇；不直接相加局部 RRF 分数，也不为交集预留位置。

两阶段均使用一基排名和常量 60：

```text
RRF(d) = Σ 1 / (60 + rank(d))
```

缺席某路贡献为 0，交集论文获得两路贡献。若两路新排序为 `[A, B]`、`[B, C]` 且 limit=2，在 A 的 ID 小于 C 时返回 `[B, A]`。最终 `score` 为最终 RRF。

### SemanticRRFRetriever

全名为 **Semantic-Weighted Reciprocal Rank Fusion**（语义加权倒数排名融合），代码类名为 `SemanticRRFRetriever`。此处按每篇论文的语义相似度加权。

稀疏侧召回最多 `ceil(limit / 2)` 篇；稠密侧各查询召回后合并，按评分查询相似度保留前 limit 篇。

使用两路列表的一基名次计算等权 RRF，再按每篇论文的语义相似度加权：

```text
score(d) = RRF(d) × (cos(score_query, d) + 1)
```

相似度为 `None` 时权重为 0。直接从候选并集中按加权分数取前 limit 篇，不预留交集、不重新计算名次；零分候选仍参与截取。`score` 为语义加权 RRF。

### SDCascadeRetriever

全名为 **Sparse-to-Dense Cascade**（稀疏到稠密级联检索），代码类名为 `SDCascadeRetriever`。

BM25 先召回最多 `limit × 60` 篇，再仅对这些候选计算各语义查询的内积，每查询取前 limit 篇并合并去重；最后按 `score_query` 重排，返回前 limit 篇。

缺少向量的候选不参与语义召回。本方法不执行全库 HNSW 搜索，也不计算 RRF；最终 `score` 为评分查询内积。BM25 无候选或没有可评分候选时返回 `[]`。

三种方法候选不足时返回较少结果，检索或初始化异常直接传播。

## SearchEngine

构造器：`SearchEngine(*, query_generator=None, sparse_retriever=None, dense_retriever=None)`。

```text
search(request, method, limit=20, *, max_terms=12, max_queries=4) -> list[SearchResult]
```

| method | 使用的参数 | 最终 limit | LLM 调用 |
| --- | --- | --- | --- |
| `BM25` | max_terms：5～20 | 正整数 | 1 次关键词生成 |
| `HNSW` | max_queries：正整数 | 正整数 | 查询生成 + 单条整合评分查询；无候选时仅 1 次 |
| `HRRF` | max_terms：5～20；max_queries：正整数 | 1～100 | 关键词、查询、评分查询各 1 次 |
| `SEMANTIC_RRF` | 同上 | 1～100 | 同上 |
| `SD_CASCADE` | 同上 | 1～100 | 同上 |

BM25 保留原始结果顺序和分数。HNSW 多查询各召回最多 limit 篇，合并去重后，用额外生成的整合查询评分重排，再截取 limit 篇。三种混合方法交给对应独立检索器执行。

方法名去除首尾空白并转为大写。空白请求、未知方法、无效数量参数在调用 LLM 前报错；所有数量参数拒绝布尔值和非整数。BM25 不使用 max_queries，HNSW 不使用 max_terms。模型、数据库、索引及生成异常直接传播。

`engine.available_methods` 返回按注册顺序排列的方法名称元组。扩展方法使用 `engine.register_method(name, handler)`，处理器遵循：

```text
handler(request: str, limit: int, *, max_terms: int, max_queries: int) -> list[SearchResult]
```

处理器负责校验自己的专属参数和管理外部资源。空白名称、重复名称及不可调用的处理器会被拒绝。

## QueryGenerator

`generate_terms(request, max_terms=12)` 返回英文关键词列表，上限必须为至少 5 的整数；`generate_queries(request, max_queries=4)` 返回英文语义查询列表，上限为正整数，设为 1 时使用整合主题与约束的专用提示词。

每个方法调用一次 LLM，通过 Pydantic 结构化模型限制返回数量为 1 至指定上限。模型定义按生成类型与上限缓存，最多 32 项；LLM 响应不缓存。独立生成器不限制 max_terms 的上界，接入 BM25 的引擎路径限制为 20。

配置来自项目根目录 `.env`，或在进程中预先设置的同名环境变量：

| 配置 | 用途 |
| --- | --- |
| `LLM_API_KEY` | LLM 服务的 API 凭证 |
| `LLM_API_URL` | 传入客户端的 `base_url`；需为支持当前接口的服务地址 |
| `LLM_MODEL` | 非空模型名称，初始化时校验 |

当前代码使用 `OpenAI(...).responses.parse`，传入 Pydantic `text_format`、`reasoning={"effort": "none"}` 和 `temperature=0`。服务和模型必须支持这一接口与参数组合，不能仅凭“兼容 OpenAI”就假定可用。源码没有自动改用 Chat Completions 或本地生成服务的逻辑。

生成器只接收研究请求，不接收已检索论文。其输出是英文检索输入，不是推荐理由；也不执行日期、分类等元数据硬过滤。自然语言约束会保留在生成提示词中，但不等于查询结果必然满足每项约束。

## 资源生命周期

上述检索器、生成器和引擎均支持 `with` 或手动 `close()`，关闭幂等，关闭后不能继续操作或重新进入上下文。

自建连接与客户端由所属对象关闭，注入组件由调用方关闭；混合检索器和引擎不会关闭借用的检索器，HNSW 不接管注入的向量化器。关闭操作不强制清理全局 CUDA 缓存；模型和索引对象随其引用生命周期释放。

引擎组件按需创建：BM25 路径不会创建 HNSW 检索器或实例化向量化器。初始化失败时，混合检索器释放已创建的连接。

## 导入迁移

本次底层 API 直接迁移，不提供旧调用方式或参数别名：

| 原调用 | 当前调用 |
| --- | --- |
| BM25 旧关键词搜索 `term_search(terms, limit)` | `search(query=terms, limit=limit)` |
| BM25 `get_score(ids, terms=terms)` | `get_score(ids, query=terms)`；位置参数调用不变 |
| HNSW 单查询搜索后读取 `results[query]` | `search(query, limit)` 直接返回列表，默认 limit 从 5 改为 20 |
| HNSW 分组搜索 `search(query_or_queries, limit)` | `batch_search(query_or_queries, limit)`，默认 limit 仍为 5 |
| HNSW `get_score(query_ids)` | `batch_get_score(query_ids)`；单查询可用 `get_score(ids, query)` |

`SearchEngine.search` 和三种混合检索器的 `search` 参数及返回结构保持不变。

模块文件已统一为 snake_case，旧模块路径不提供转发文件。推荐改用本页的包级导入。

| 原模块名 | 当前模块名 |
| --- | --- |
| `SearchEngine` / `SearchResult` | `search_engine` / `search_result` |
| `sparse.BM25Retriever` | `sparse.bm25_retriever` |
| `dense.HNSWRetriever` | `dense.hnsw_retriever` |
| `query.Generator` | `query.generator` |
| `vectorization.Qwen3Vectorizer` | `vectorization.qwen3_vectorizer` |

混合检索器类名统一使用 `HRRFRetriever`、`SemanticRRFRetriever` 和 `SDCascadeRetriever`，保留 `Retriever` 后缀；引擎方法标识分别为 `HRRF`、`SEMANTIC_RRF` 和 `SD_CASCADE`，不添加 `HYBRID_` 前缀。调用方需同步更新类名、模块路径和方法标识，不提供旧名称别名；推荐使用本文开头的包级导入，模块路径见[项目结构](../structure.md)。

旧有调用方无需再访问 `retriever.conn.close()`、`generator.client.close()` 或逐个关闭混合检索器的内部组件，统一使用对象的 `close()` 或上下文管理。
