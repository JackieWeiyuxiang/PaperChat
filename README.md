# arxiv-paper-recommend

面向 arXiv 论文的检索项目，提供 BM25、HNSW、三种混合检索策略，以及 Qwen3 本地文本向量化。可直接提供关键词／语义查询，也可通过 `SearchEngine` 从研究请求生成输入并检索。

当前主要提供 Python API。推荐、语料处理和 Web 包仍为占位模块，`arxiv-paper-recommend` 命令目前仅打印问候信息。

## 安装

要求 Python >= 3.14，使用 uv 管理依赖。在项目根目录运行：

```powershell
uv sync
```

HNSW 文本查询、所有混合检索和本地向量化还需要模型依赖：

```powershell
uv sync --extra vectorization
```

## 准备数据

默认输入为 `data/raw/papers.jsonl` 和 `data/raw/vectors.npz`。首次使用时依次构建数据库、向量索引；已有匹配的数据库和索引时可跳过。

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

从自然语言研究请求检索时，先将 [.env.example](.env.example) 复制为根目录 `.env`，配置 `LLM_API_KEY`、`LLM_API_URL` 和 `LLM_MODEL`：

```python
from arxiv_paper_recommend.retrieval import SearchEngine

with SearchEngine() as engine:
    results = engine.search(
        "How can retrieval improve language model reasoning?",
        method="HYBRID_CANDIDATE_FUSION",
        limit=10,
    )
    for result in results:
        print(result.doc_id, result.title, result.score)
```

`SearchEngine` 按需创建并复用组件。直接调用底层检索器无需 LLM；HNSW 与混合检索仍需要向量化模型。

## 选择检索方法

| SearchEngine 方法 | 检索流程 | 最终分数 |
| --- | --- | --- |
| `BM25` | 关键词／短语匹配 | 原始 BM25，越小越相关 |
| `HNSW` | 多查询近似召回，合并后按整合查询重排 | 内积相似度，越大越相关 |
| `HYBRID_CANDIDATE_FUSION` | 两路候选分别重排，再融合新排名 | 等权 RRF |
| `HYBRID_SEMANTIC_WEIGHTED` | 两路召回，按逐篇语义相似度加权 | 语义加权 RRF |
| `HYBRID_SEQUENTIAL` | BM25 筛选候选，候选内多查询选择与重排 | 内积相似度 |

不同方法的分数不能直接比较。方法名不区分大小写；完整参数、算法规则和独立检索器示例见[检索 API](docs/retrieval.md)。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [检索 API](docs/retrieval.md) | 公开接口、排名规则、分数含义、资源管理和导入迁移 |
| [向量化](docs/vectorization.md) | 模型配置、编码接口、设备与批处理 |
| [数据与构建](docs/data.md) | 数据格式、归档元数据、数据库与索引构建更新 |
| [项目结构](structure.md) | 模块职责、依赖边界和数据流 |
