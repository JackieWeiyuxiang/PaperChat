# arxiv-paper-recommend

基于 arXiv 标题与摘要的检索增强论文推荐项目，用于 AI6127（2026）课程作业。目标是根据自然语言研究需求生成 Top-10 论文推荐、简短推荐理由和来源引用。

当前已实现检索阶段的 Python API：BM25、HNSW、三种混合检索策略、Qwen3 本地向量化，以及 LLM 查询生成。可直接提供关键词／语义查询，也可通过 `SearchEngine` 从研究请求生成检索输入。返回的是论文及检索分数，尚不生成推荐理由或证据引用。

## 当前进度

以下状态依据 2026-10-16 的工作区代码与仓库数据整理，不计未纳入 Git 的本地实验材料。

| 部分           | 当前情况                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------ |
| 知识库         | 6,000 篇论文、6,000 个段落、1,145,234 个空白分隔词；尚未达到作业至少 10,000 篇／段落的要求 |
| 检索与向量化   | 两种基础检索、三种混合策略、查询生成、数据库及索引构建已实现                               |
| 评估           | 仓库未提供人工标注集、五查询结果与耗时、检索指标、生成质量或消融结果                       |
| 推荐与引用     | `recommendation/` 及其子包为占位，待实现基于检索证据的生成                                 |
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

## 选择检索方法

三种混合方法分别为 HRRF（Hierarchical Reciprocal Rank Fusion，分层倒数排名融合）、SemanticRRF（Semantic-Weighted Reciprocal Rank Fusion，语义加权倒数排名融合）和 SDCascade（Sparse-to-Dense Cascade，稀疏到稠密级联检索）。代码类名使用简称并保留 `Retriever` 后缀。

| SearchEngine 方法 | 检索流程                              | 最终分数               |
| ----------------- | ------------------------------------- | ---------------------- |
| `BM25`            | 关键词／短语匹配                      | 原始 BM25，越小越相关  |
| `HNSW`            | 多查询近似召回，合并后按整合查询重排  | 内积相似度，越大越相关 |
| `HRRF`            | 两路候选分别重排，再融合新排名        | 等权 RRF               |
| `SEMANTIC_RRF`    | 两路召回，按逐篇语义相似度加权        | 语义加权 RRF           |
| `SD_CASCADE`      | BM25 筛选候选，候选内多查询选择与重排 | 内积相似度             |

不同方法的分数不能直接比较。方法名不区分大小写；完整参数、算法规则和独立检索器示例见[检索 API](docs/retrieval.md)。

## 文档导航

| 文档                                               | 内容                                               |
| -------------------------------------------------- | -------------------------------------------------- |
| [项目目标与进度](docs/project.md)                  | 作业要求、方案目标、实现状态、差异与剩余交付       |
| [检索 API](docs/retrieval.md)                      | 公开接口、排名规则、分数含义、资源管理和导入迁移   |
| [向量化](docs/vectorization.md)                    | 模型配置、编码接口、设备与批处理                   |
| [数据与构建](docs/data.md)                         | 实际来源、语料统计、数据格式、归档元数据与构建更新 |
| [实验与评估](docs/evaluation.md)                   | 人工标注方案、指标、实验记录与交付要求             |
| [检索方案概览（PDF）](docs/retrieval_overview.pdf) | 现有检索架构与五种方法的图文简介                   |
| [项目结构](structure.md)                           | 模块职责、依赖边界和数据流                         |
