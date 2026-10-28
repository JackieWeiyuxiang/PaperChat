# 项目结构与模块职责

本文面向维护者，定义模块职责、依赖方向和数据流。安装使用见 [README](README.md)，公开契约见[检索 API](docs/retrieval.md)、[推荐生成](docs/recommendation.md)与[证据生成](docs/evidence_grounded_generation.md)。结构变更时同步更新本文。

## 目录

```text
arxiv_paper_recommend/
├── README.md / structure.md / AGENTS.md
├── pyproject.toml / uv.lock / .env.example
├── docs/                           # API、数据、模型、迁移、课程与评估说明
├── experiments/generation/         # 可复现的生成对比脚本与说明
│   ├── README.md
│   ├── prepare_candidates.py
│   └── run_comparison.py
├── data.db                         # 历史数据，不属于当前默认数据流
├── data/
│   ├── raw/papers.jsonl
│   ├── raw/vectors.npz
│   ├── documents.db
│   ├── documents.faiss
│   └── example_queries.json
└── src/arxiv_paper_recommend/
    ├── __init__.py                 # 包说明，无命令入口
    ├── config.py                  # 默认路径、按需读取 LLMSettings
    ├── common/
    │   ├── lifecycle.py
    │   └── validation.py
    ├── llm.py                     # 共享结构化生成及客户端所有权
    ├── corpus/
    │   ├── build_database.py
    │   └── build_index.py
    ├── vectorization/
    │   └── qwen3_vectorizer.py
    ├── retrieval/
    │   ├── base.py / constants.py / search_result.py
    │   ├── search_engine.py / _ranking.py
    │   ├── sparse/
    │   │   ├── bm25.py
    │   │   └── title_weighted_bm25.py
    │   ├── dense/
    │   │   ├── hnsw.py
    │   │   └── multi_query_hnsw.py
    │   ├── hybrid/
    │   │   ├── base.py / constants.py
    │   │   └── hrrf.py / semantic_rrf.py / sd_cascade.py
    │   ├── query/generator.py
    │   └── innovation/            # 规划占位
    ├── recommendation/
    │   ├── models.py
    │   ├── generation/generator.py
    │   ├── citations/             # 规划占位
    │   └── innovation/
    │       ├── models.py          # 带原文证据的推荐记录
    │       └── evidence_generator.py
    └── web/                       # 规划占位
```

各包包含相应的 `__init__.py`。复数形式的 `experiments/` 保存纳入 Git 的实验代码；单数形式的本地 `experiment/` 保存默认实验输出，与 `data/statistics.md` 一同被 Git 忽略，不能作为仓库已有正式评估结果的依据。

## 职责与依赖

| 模块 | 职责 | 依赖边界 |
| --- | --- | --- |
| `common` | 资源所有权、上下文管理、基本输入校验 | 仅依赖标准库 |
| `config` | 默认数据路径、构造时读取 LLM 配置 | 导入时不读取 `.env`、不创建客户端 |
| `llm` | 共享结构化调用、解析结果检查及客户端关闭 | 依赖 `common`、`config`；不包含业务提示词 |
| `corpus` | 导入准备好的语料、构建 FTS5 和 FAISS 索引 | 不执行在线检索；采集清洗和向量归档流程尚未实现 |
| `vectorization` | 模型加载、批量编码、池化与归一化 | 不保存归档或构建索引 |
| `retrieval.sparse / dense` | 召回和指定 ID 评分 | 不调用 LLM；查询类型和原始分数尺度由算法决定 |
| `retrieval.hybrid` | 候选选择、重排和融合 | 通过稀疏／稠密组件公开接口协作；缓存仅限单次搜索 |
| `retrieval.query` | 从研究请求生成关键词及语义查询 | 依赖共享 LLM 层，不执行检索或向量化 |
| `retrieval.search_engine` | 注册方法、生成输入、延迟创建和复用组件 | 不承担算法内部排名逻辑 |
| `recommendation.models` | 推荐记录及原始检索结果关联 | 不依赖 LLM 模块 |
| `recommendation.generation` | 基于候选标题和摘要生成理由、校验论文与响应 | 依赖共享 LLM 层及公开检索结果模型，不访问检索内部工具 |
| `recommendation.innovation.models` | 带原文证据的推荐记录及原始检索结果关联 | 模型定义不依赖 LLM；经 `innovation` 包导入时会同时加载生成器 |
| `recommendation.innovation.evidence_generator` | 生成理由和原文证据，校验候选、重复项与精确证据片段 | 复用共享 LLM 层及公开检索结果模型；不自动验证理由的语义支持性 |
| `experiments/generation` | 固定检索候选导出、基础与证据生成的配对运行、状态与耗时记录 | 通过公开 API 调用检索和生成；不实现人工标注或质量指标汇总 |
| `recommendation.citations / retrieval.innovation / web` | 保留完整引用、检索创新和界面扩展边界 | 当前为空包，不代表已有功能 |

`Retriever` 统一稀疏／稠密的搜索与评分契约；`HybridRetriever` 独立定义混合搜索，不提供单独评分接口。共享限值位于 `retrieval.constants`，混合算法专用预算位于 `retrieval.hybrid.constants`，排序与 RRF 位于检索内部模块 `_ranking`。

## 数据流

离线构建需显式调用 `build()` 或对应模块命令：

```text
papers.jsonl ── DatabaseBuilder ──> documents.db
vectors.npz + documents.db ── IndexBuilder ──> documents.faiss
```

构建器分别位于 `corpus.build_database`、`corpus.build_index`，不在包根导出。导入构建模块不会读取语料或写入数据。

```text
研究请求 ── SearchEngine ── QueryGenerator ── terms / recall_queries / query
                    └── 方法分派
                        ├── 稀疏检索 ── documents.db
                        ├── 稠密检索 ── Qwen3Vectorizer + documents.faiss + documents.db
                        └── 混合检索 ── 共享稀疏与稠密组件
                                      ↓
                               list[SearchResult]
研究请求 + 检索结果 ── RecommendationGenerator ── list[PaperRecommendation]
                 └── EvidenceGroundedRecommendationGenerator
                       └── 原文证据校验 ── list[EvidenceGroundedRecommendation]
```

下游生成由调用方显式衔接。基础方法返回推荐理由；证据生成方法额外返回标题或摘要中的精确连续原文。精确匹配不保证理由在语义上被证据支持，当前结果均不补充作者、日期、URL 或完整引用元数据。

[生成对比实验](experiments/generation/README.md)先通过 `prepare_candidates.py` 将检索候选保存为 JSONL，再通过 `run_comparison.py` 让两种生成器读取相同候选并记录输出。默认文件为本地 `experiment/generation/candidates.jsonl` 与 `comparison.jsonl`；脚本不属于库的在线调用链。

## 维护约定

- 存储层 `documents.id`、FTS5 rowid、FAISS ID 与 `SearchResult.document_id` 对应；存储层 `doc_id` 映射为 `SearchResult.arxiv_id`，保留论文版本号。
- 仅关闭自建资源；注入组件由调用方管理。关闭幂等，关闭后的对象不能继续操作。
- 包级导入为业务 API 推荐入口；检索根包延迟导出引擎，推荐根包延迟导出基础生成器。导入 `SearchResult` 或基础推荐模型 `PaperRecommendation` 不加载 FAISS 或 LLM 客户端模块；`recommendation.innovation` 当前直接导出模型与生成器，会加载共享 LLM 模块。
- PyTorch、Transformers 仅在实例化向量化器时导入。模型和索引跨查询复用。
- 文件使用 snake_case，类使用 PascalCase；Python 注释与文档字符串以清晰英文为主，使用文档以中文为主；编辑后执行 `ruff format`。
