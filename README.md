# arxiv-paper-recommend

基于 arXiv 标题与摘要的检索增强论文推荐项目，用于 AI6127（2026）课程作业。目标是根据自然语言研究需求生成 Top-10 论文推荐、简短推荐理由和来源引用。

当前已实现检索阶段的 Python API：BM25、标题加权 BM25、HNSW、三种混合检索策略、Qwen3 本地向量化，以及 LLM 查询生成；另有基于检索结果的 Generation Baseline，可从候选论文中筛选推荐并生成简短理由。推荐理由保留对应的原始检索结果以便追溯，但当前版本不做事实支持性验证，也不生成来源引用。

## 当前进度

以下代码状态依据 2026-10-20 的工作区整理，数据已于 2026-10-19 从 `data-preprocessing` 分支同步，不计未纳入 Git 的本地实验材料。

| 部分           | 当前情况                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------ |
| 知识库         | 10,000 篇论文、10,000 个段落、1,917,315 个空白分隔词；已达到作业的语料规模要求 |
| 检索与向量化   | 两种基础检索、三种混合策略、标题加权 BM25 创新、查询生成、数据库及索引构建已实现          |
| 评估           | 固定输入实验与指标脚本已实现；P2 结果、人工标注及正式指标仍缺失                         |
| 推荐与引用     | `recommendation/generation` 已有结构化推荐与理由生成 baseline；来源引用与支持性验证仍待实现 |
| 数据处理与界面 | `corpus/`、`web/` 为占位；现有数据已准备，但采集清洗流程未包含在当前源码中                 |
| 命令入口       | `arxiv-paper-recommend` 仅打印问候信息，不能用于检索或启动网页                             |

完整的作业要求、提案与实际实现差异见[项目目标与进度](docs/project.md)；待完成的评估见[实验与评估](docs/evaluation.md)。三种混合策略的效果尚无人工评估支持，不预设哪种方法更优。

## 安装

要求 Python >= 3.14，使用 uv 管理依赖；SQLite 需支持 FTS5。在项目根目录运行：

```powershell
uv sync
```

HNSW 文本查询、所有混合检索和本地向量化还需要模型依赖：

```powershell
uv sync --extra vectorization
```

## 准备数据

默认输入为 `data/raw/papers.jsonl` 和 `data/raw/vectors.npz`。当前仓库包含这两个快照及 `data/documents.db`、`data/documents.faiss`。若需要从原始快照重建，依次构建数据库和向量索引；已有匹配的数据库和索引时可跳过。

```powershell
uv run python -m arxiv_paper_recommend.retrieval.sparse.build_database
uv run python -m arxiv_paper_recommend.retrieval.dense.build_index
```

默认输出为 `data/documents.db`、`data/documents.faiss`。自定义路径、数据格式及更新规则见[数据与构建说明](docs/data.md)。

[查询示例](data/example_queries.json)包含 40 条中英文研究请求，覆盖十个主题，可用于试用检索；这些示例尚无人工相关性标注。

## 快速开始

直接使用 BM25 无需模型依赖或 LLM 配置：

```python
from arxiv_paper_recommend.retrieval.sparse import BM25Retriever

with BM25Retriever() as retriever:
    results = retriever.term_search(
        ["retrieval augmented generation", "reasoning"], limit=10,
    )
    for result in results:
        print(result.doc_id, result.title, result.score)
```

从自然语言研究请求检索时，先将 [.env.example](.env.example) 复制为根目录 `.env`，配置 `LLM_API_KEY`、`LLM_API_URL` 和 `LLM_MODEL`。查询生成调用 Responses API 的结构化输出，服务与模型需支持当前调用参数，详见[查询生成配置](docs/retrieval.md#querygenerator)。

```powershell
Copy-Item .env.example .env
```

已有 `.env` 时直接编辑现有配置。安装向量化依赖后可使用：

```python
from arxiv_paper_recommend.retrieval import SearchEngine

with SearchEngine() as engine:
    results = engine.search(
        "How can retrieval improve language model reasoning?",
        method="HRRF",
        limit=10,
    )
    for result in results:
        print(result.doc_id, result.title, result.score)
```

`SearchEngine` 按需创建并复用组件。直接调用底层检索器无需 LLM；HNSW 与混合检索仍需要向量化模型。

检索后可用基础生成器从候选论文中选择并排序最多 10 篇，按标题和摘要生成简短推荐理由（需配置 `.env` 中的 LLM 参数）。生成器只接收检索候选的 `doc_id`、标题和摘要，并校验返回的论文 ID 属于候选且没有重复；这能保证推荐对象可追溯，但不能证明理由中的每个事实都由摘要支持。

```python
from arxiv_paper_recommend.recommendation.generation import RecommendationGenerator

with SearchEngine() as engine, RecommendationGenerator() as generator:
    papers = engine.search("RAG citation reliability", method="HRRF", limit=20)
    recommendations = generator.generate(
        "Find research on whether citations support claims in RAG answers.",
        papers,
        limit=10,
    )
    for item in recommendations:
        print(item.rank, item.result.doc_id, item.result.title, item.reason)
```

生成器实现位于 [`baseline_generator.py`](src/arxiv_paper_recommend/recommendation/generation/baseline_generator.py)，包级入口位于 [`generation/__init__.py`](src/arxiv_paper_recommend/recommendation/generation/__init__.py)。它使用配置的 Responses API 模型和结构化 Pydantic 输出；推荐结果包含 `rank`、原始 `SearchResult` 和 `reason`。模型未配置时不会产生生成结果。完整设计、输入输出和限制见 [P3 中文报告](docs/p3_report.md)。

## 选择检索方法

三种混合方法分别为 HRRF（Hierarchical Reciprocal Rank Fusion，分层倒数排名融合）、SemanticRRF（Semantic-Weighted Reciprocal Rank Fusion，语义加权倒数排名融合）和 SDCascade（Sparse-to-Dense Cascade，稀疏到稠密级联检索）。代码类名使用简称并保留 `Retriever` 后缀。

| SearchEngine 方法 | 检索流程                              | 最终分数               |
| ----------------- | ------------------------------------- | ---------------------- |
| `BM25`            | 关键词／短语匹配                      | 原始 BM25，越小越相关  |
| `TITLE_BM25`      | BM25，标题字段权重为摘要的 3 倍         | 加权 BM25，越小越相关  |
| `HNSW`            | 多查询近似召回，合并后按整合查询重排  | 内积相似度，越大越相关 |
| `HRRF`            | 两路候选分别重排，再融合新排名        | 等权 RRF               |
| `SEMANTIC_RRF`    | 两路召回，按逐篇语义相似度加权        | 语义加权 RRF           |
| `SD_CASCADE`      | BM25 筛选候选，候选内多查询选择与重排 | 内积相似度             |

不同方法的分数不能直接比较。方法名不区分大小写；完整参数、算法规则和独立检索器示例见[检索 API](docs/retrieval.md)。

`TITLE_BM25` 是本项目新增的检索创新：利用论文标题更集中表达主题的特点，提高标题匹配的排序权重。实现位于 [`title_aware_bm25.py`](src/arxiv_paper_recommend/retrieval/innovation/title_aware_bm25.py)，统一接口分派位于 [`search_engine.py`](src/arxiv_paper_recommend/retrieval/search_engine.py)。3:1 是待验证的初始权重；目前只有探索性排名案例，尚无人工相关性标注支持效果结论。实现细节、案例和评估限制见 [P3 中文报告](docs/p3_report.md)。

## 文档导航

| 文档                                               | 内容                                               |
| -------------------------------------------------- | -------------------------------------------------- |
| [项目目标与进度](docs/project.md)                  | 作业要求、方案目标、实现状态、差异与剩余交付       |
| [检索 API](docs/retrieval.md)                      | 公开接口、排名规则、分数含义、资源管理和导入迁移   |
| [向量化](docs/vectorization.md)                    | 模型配置、编码接口、设备与批处理                   |
| [数据与构建](docs/data.md)                         | 实际来源、语料统计、数据格式、归档元数据与构建更新 |
| [实验与评估](docs/evaluation.md)                   | 人工标注方案、指标、实验记录与交付要求             |
| [P3 中文报告](docs/p3_report.md)                    | 检索基线问题、标题加权 BM25 创新、Generation Baseline 实现与评估边界 |
| [检索方案概览（PDF）](docs/retrieval_overview.pdf) | 现有检索架构与五种方法的图文简介                   |
| [项目结构](structure.md)                           | 模块职责、依赖边界和数据流                         |
