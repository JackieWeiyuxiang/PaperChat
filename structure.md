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
│   └── statistics.md                # 本地报告，被 Git 忽略
└── src/arxiv_paper_recommend/
    ├── __init__.py                  # main()，当前仅打印问候信息
    ├── config.py
    ├── vectorization/
    │   ├── __init__.py
    │   └── qwen3_vectorizer.py
    ├── retrieval/
    │   ├── __init__.py              # SearchResult；延迟导出 SearchEngine
    │   ├── search_result.py
    │   ├── search_engine.py
    │   ├── _lifecycle.py
    │   ├── _validation.py
    │   ├── _ranking.py
    │   ├── sparse/
    │   │   ├── __init__.py
    │   │   ├── bm25_retriever.py
    │   │   └── build_database.py
    │   ├── dense/
    │   │   ├── __init__.py
    │   │   ├── hnsw_retriever.py
    │   │   └── build_index.py
    │   ├── hybrid/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── constants.py
    │   │   ├── hrrf_retriever.py
    │   │   ├── semantic_rrf_retriever.py
    │   │   └── sd_cascade_retriever.py
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
| `retrieval.search_result` | 统一结果数据类 | 不执行检索或评分 |
| `retrieval.sparse` | 数据库导入、FTS5 构建、关键词召回和 BM25 评分 | 保留原始 BM25 分数 |
| `retrieval.dense` | 向量索引构建、HNSW 召回、指定 ID 精确评分及结果读取 | 不生成 LLM 查询；评分仅检查数据库 ID |
| `retrieval.query` | LLM 输入生成、结构化响应与生成模型缓存 | 不执行向量化或检索 |
| `retrieval.hybrid` | 跨检索器候选选择、重排与融合 | 三种策略各自实现算法，复用基类和每次搜索的评分缓存 |
| `retrieval.search_engine` | 方法注册、请求生成、组件复用和方法分派 | BM25 保留原分数；HNSW 合并后重排；混合算法交给对应检索器 |
| `retrieval._lifecycle / _validation / _ranking` | 资源所有权、公共校验和排序／RRF | 内部模块，不作为稳定公开 API |
| `corpus` | 预留采集、清洗、分块及语料统计职责 | 当前为空包；现有数据快照不代表该流程已迁入源码 |
| `recommendation.generation / citations / innovation` | 预留证据驱动的推荐生成、引用与下游创新职责 | 当前均为空包；查询生成器不能代替下游生成 |
| `web` | 预留用户研究需求输入和推荐展示职责 | 当前为空包；没有可启动的 Web 服务 |

`BaseHybridRetriever` 管理默认检索器和公共校验，`DenseScoreCache` 只在单次搜索内缓存评分。自建资源通过 `ManagedResource` 注册关闭操作，注入组件由调用方管理。

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
                        ├── BM25Retriever ── documents.db
                        ├── HNSWRetriever ── Qwen3Vectorizer + 索引 + 数据库
                        └── 混合检索器 ── 复用 BM25 与 HNSW
                                      ↓
                               list[SearchResult]
```

底层检索器也可独立调用；HNSW 底层返回按查询分组的结果字典，引擎统一返回结果列表。

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
- PyTorch 和 Transformers 仅在向量化器实例化时导入；检索根包不会因导入 `SearchResult` 而加载引擎依赖。
- 模块文件使用 `snake_case`，类名使用 `PascalCase`；包级导入为推荐公开入口。
- 结构变化时同步更新本文。推荐和 Web 功能分别扩展到现有占位包，不混入检索或向量化模块。
