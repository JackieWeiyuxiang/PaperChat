# 项目结构与模块职责

本文面向维护者，说明当前工作区目录、职责边界和数据流。安装与快速开始见 [README](README.md)，课程目标与实现状态见[项目目标与进度](docs/project.md)，接口与算法见[检索 API](docs/retrieval.md)，模型配置见[向量化](docs/vectorization.md)，存储格式见[数据与构建](docs/data.md)，评估设计见[实验与评估](docs/evaluation.md)。

## 目录

```text
arxiv_paper_recommend/
├── README.md
├── structure.md
├── AGENTS.md
├── pyproject.toml / uv.lock          # 依赖、Python 版本和命令入口
├── .env.example                     # LLM 配置示例
├── docs/
│   ├── retrieval.md
│   ├── vectorization.md
│   ├── data.md
│   ├── project.md                   # 课程要求、方案差异与实现进度
│   ├── evaluation.md                # 待执行评估设计与交付要求
│   └── retrieval_overview.pdf       # 检索技术概览
├── data.db                         # 历史文件，不用于当前默认接口
├── data/
│   ├── raw/papers.jsonl
│   ├── raw/vectors.npz
│   ├── documents.db
│   ├── documents.faiss
│   ├── example_queries.json         # 40 条中英文研究请求，尚无相关性标注
│   └── statistics.md                # 本地报告，被 Git 忽略
└── src/arxiv_paper_recommend/
    ├── __init__.py                  # main()，当前仅打印问候信息
    ├── config.py
    ├── vectorization/
    │   ├── __init__.py
    │   └── qwen3_vectorizer.py
    ├── retrieval/
    │   ├── __init__.py              # Retriever、SearchResult；延迟导出 SearchEngine
    │   ├── base.py                  # sparse / dense 的泛型抽象接口
    │   ├── search_result.py
    │   ├── search_engine.py
    │   ├── _lifecycle.py
    │   ├── _validation.py
    │   ├── _ranking.py
    │   ├── sparse/
    │   │   ├── __init__.py
    │   │   ├── bm25.py
    │   │   ├── title_aware_bm25.py   # 可配置标题权重的 BM25 子类
    │   │   └── build_database.py
    │   ├── dense/
    │   │   ├── __init__.py
    │   │   ├── hnsw.py    # 单查询召回与精确评分
    │   │   ├── divisional_hnsw.py # 分路召回、重排与内部批量计算
    │   │   └── build_index.py
    │   ├── hybrid/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── constants.py
    │   │   ├── hrrf.py
    │   │   ├── semantic_rrf.py
    │   │   └── sd_cascade.py
    │   ├── query/
    │   │   ├── __init__.py
    │   │   └── generator.py
    │   └── innovation/              # 占位
    ├── corpus/                      # 占位
    ├── recommendation/              # 占位
    │   ├── citations/
    │   ├── generation/
    │   └── innovation/
    └── web/                         # 占位
```

## 职责边界

| 模块 | 负责 | 边界 |
| --- | --- | --- |
| `config` | 从根目录 `.env` 加载 LLM 配置，定义默认数据路径 | 不创建客户端或读取语料 |
| `vectorization` | 模型加载、文本编码、池化与归一化 | 返回向量，不保存归档、不构建索引 |
| `retrieval.base` | sparse / dense 的泛型 `search`、`get_score` 抽象接口 | 复用资源管理，不统一查询类型或分数尺度 |
| `retrieval.search_result` | 统一结果数据类 | 不执行检索或评分 |
| `retrieval.sparse` | 数据库导入、FTS5 构建、关键词召回、普通及标题加权 BM25 评分 | 保留各自的原始 BM25 分数，复用同一索引 |
| `retrieval.dense` | 向量索引构建、HNSW 单查询与分路召回、去重重排、批量操作及指定 ID 精确评分 | 不生成 LLM 查询；评分核对数据库 ID，不读取标题和摘要 |
| `retrieval.query` | LLM 输入生成、结构化响应与生成模型缓存 | 不执行向量化或检索 |
| `retrieval.hybrid` | 独立的 `HybridRetriever` 检索接口及三种策略的候选选择、重排与融合 | 仅规定 `search`，不提供 `get_score`；复用每次搜索的评分缓存 |
| `retrieval.search_engine` | 方法注册、请求生成、组件复用和方法分派 | BM25 / TITLE_BM25 保留原分数；HNSW 单查询召回，DIVISIONAL_HNSW 与混合算法委托对应检索器 |
| `retrieval._lifecycle / _validation / _ranking` | 资源所有权、公共校验和排序／RRF | 内部模块，不作为稳定公开 API |
| `corpus` | 预留采集、清洗、分块及语料统计职责 | 当前为空包；现有数据快照不代表该流程已迁入源码 |
| `recommendation.generation / citations / innovation` | 预留证据驱动的推荐生成、引用与下游创新职责 | 当前均为空包；查询生成器不能代替下游生成 |
| `web` | 预留用户研究需求输入和推荐展示职责 | 当前为空包；没有可启动的 Web 服务 |

`BM25Retriever` 和 `HNSWRetriever` 分别继承 `Retriever[list[str]]`、`Retriever[str]`，统一 `search(query, limit=20)` 和 `get_score(ids, query)`。`HybridRetriever` 独立于该泛型基类，管理默认检索器和公共校验，稀疏／稠密组件分别接受并默认创建 `TitleAwareBM25Retriever`、`DivisionalHNSWRetriever`。混合检索仅调用检索器公开的 `search` 和 `get_score`；HRRF / SemanticRRF 委托稠密组件完成分路召回及重排，SD Cascade 在稀疏候选内按查询评分筛选。`DenseScoreCache` 只在单次搜索内缓存分数，包括不可评分的 `None`。自建资源通过 `ManagedResource` 注册关闭操作，注入组件由调用方管理。

三种 hybrid 的接口均为 `search(terms, query, limit=20, *, title_weight=None, divisional_queries=None)`，不再接受 `queries` 参数。标题权重用于稀疏召回和 HRRF 补评分；可选参数直接透传，组件收到 `None` 时回退到父类基础逻辑。SD Cascade 在未提供分路时使用 `query` 进行候选内筛选，提供时使用分路查询，最终始终按 `query` 进行语义重排；空集合不回退到评分查询。引擎通过 `divisional_queries` 传入生成的分路查询，省略标题权重。

`DivisionalHNSWRetriever` 继承 `HNSWRetriever`，沿用构造器、精确评分与资源管理；`search(query, limit=20, *, divisional_queries=None)` 在未提供分路查询时回退到父类单查询，提供时各分路召回、合并去重，再用 query 评分重排。分路召回的批量计算仅在类内使用。引擎的 `dense_retriever` 使用该子类，并向混合检索器注入同一实例，让引擎各条稠密路径共享资源。

`TitleAwareBM25Retriever` 继承 `BM25Retriever`，由 sparse 包公开导出，沿用父类构造器和资源管理。`search(query, limit=20, *, title_weight=None)` 与 `get_score(ids, query, *, title_weight=None)` 在省略权重时回退到父类；显式传入有限非负权重时，内部接口仅接收 `title_weight`，摘要权重在 SQL 中固定为 `1.0`，doc_id 为未索引字段，不参与评分。搜索和指定 ID 评分复用父类私有方法，权重仅作用于本次调用，不保存在实例上。引擎的 `sparse_retriever` 使用该子类，按需创建或通过同名参数注入，供 BM25、TITLE_BM25 和混合检索共享同一实例。TITLE_BM25 每次显式传入 `title_weight=3.0`；BM25 和混合检索不传权重，使用普通 BM25。

## 数据流

离线构建：

```text
papers.jsonl ── DatabaseBuilder ──> documents.db
vectors.npz + documents.db ── IndexBuilder ──> documents.faiss
```

在线检索：

```text
研究请求 ── SearchEngine ── QueryGenerator ── 关键词 / 语义查询 / 评分查询
                    └── 方法分派
                        ├── BM25 / TITLE_BM25 ── TitleAwareBM25Retriever ── documents.db
                        ├── HNSW / DIVISIONAL_HNSW ── DivisionalHNSWRetriever
                        │                              └── Qwen3Vectorizer + 索引 + 数据库
                        └── 混合检索器 ── 复用 TitleAwareBM25Retriever 与 DivisionalHNSWRetriever
                                      ↓
                               list[SearchResult]
```

底层检索器也可独立调用，`search` 均返回结果列表，`get_score` 返回单查询的 ID 分数映射。分路检索在类内批量召回；混合检索通过公开方法委托召回或按查询评分，不访问其他检索器的私有方法。引擎 HNSW 生成一条整合查询；DIVISIONAL_HNSW 在检索前分别生成分路查询和整合评分查询，因此空召回时也会调用两次 LLM。

目标中的下游流程尚未实现：

```text
list[SearchResult] ──> recommendation（理由生成 + 来源引用） ──> web 展示
                           当前为占位                         当前为占位
```

原始 JSONL 保留作者、来源链接、日期和分类，但当前数据库与 `SearchResult` 只保留检索所需字段。实现推荐与网页时，需要在既定职责内补齐元数据读取与证据对应关系，不能假设现有结果已包含这些信息。

## 共享约定

- `documents.id`、FTS5 的 `rowid`、FAISS 向量 ID 和 `SearchResult.id` 对应；论文 `doc_id` 保留版本号，仅移除 `arxiv:` 前缀。
- 模型、连接和索引封装在可复用对象中。关闭检索器或引擎只关闭其自建资源；注入组件的生命周期由调用方控制。
- 构建操作只能由显式调用 `build()` 或模块命令触发；导入构建模块不读取语料或写入索引。
- PyTorch 和 Transformers 仅在向量化器实例化时导入；检索根包不会因导入 `Retriever` 或 `SearchResult` 而加载引擎依赖。
- 模块文件使用 `snake_case`，类名使用 `PascalCase`；包级导入为推荐公开入口。
- 结构变化时同步更新本文。推荐和 Web 功能分别扩展到现有占位包，不混入检索或向量化模块。
