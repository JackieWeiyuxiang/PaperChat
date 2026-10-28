# arxiv-paper-recommend

基于 arXiv 标题与摘要的检索增强论文推荐项目，用于 AI6127（2026）课程作业。通过 Python API 检索论文，并基于候选内容生成最多 10 条推荐及简短理由。

## 项目能力

已有 10,000 篇论文的数据快照、SQLite FTS5 数据库和 FAISS 向量索引。支持关键词检索、本地 Qwen3 向量化、LLM 查询生成及推荐理由生成。下游生成提供基础方法和返回标题／摘要原文证据的 Evidence-Grounded 方法。

| 检索方法 | 流程 | 分数方向 |
| --- | --- | --- |
| `BM25` | 关键词与短语匹配 | 越小越相关 |
| `TITLE_WEIGHTED_BM25` | 标题权重为 3、摘要权重为 1 的 BM25 | 越小越相关 |
| `HNSW` | 单查询近似向量召回 | 越大越相关 |
| `MULTI_QUERY_HNSW` | 多查询召回，合并后按主查询重排 | 越大越相关 |
| `HRRF` | 两路候选分别重排，再融合排名 | 越大越相关 |
| `SEMANTIC_RRF` | 按逐篇语义相似度加权 RRF | 越大越相关 |
| `SD_CASCADE` | BM25 候选内执行多查询选择与重排 | 越大越相关 |

不同方法的分数不能直接比较。目前没有正式质量或性能评估结论。独立来源引用、Web 界面和完整采集清洗流程仍待实现，详见[项目目标与进度](docs/project.md)。

## 安装

要求 Python >= 3.14、uv，以及支持 FTS5 的 SQLite。在项目根目录运行：

```powershell
uv sync
```

HNSW、混合检索及本地向量化需要额外模型依赖：

```powershell
uv sync --extra vectorization
```

仓库已有匹配的 `data/documents.db` 和 `data/documents.faiss`，可直接使用。需要从原始快照重建时，依次运行：

```powershell
uv run python -m arxiv_paper_recommend.corpus.build_database
uv run python -m arxiv_paper_recommend.corpus.build_index
```

自定义路径和更新规则见[数据与索引构建](docs/data.md)。项目当前使用 Python API，没有检索命令或 Web 启动入口。

## 最小检索示例

直接调用 BM25 无需配置 LLM，也无需安装模型依赖：

```python
from arxiv_paper_recommend.retrieval.sparse import BM25Retriever

with BM25Retriever() as retriever:
    papers = retriever.search(
        ["retrieval augmented generation", "reasoning"], limit=10,
    )
    for paper in papers:
        print(paper.arxiv_id, paper.title, paper.score)
```

底层检索器还提供 `score(query, document_ids)`，为指定数据库整数 ID 评分。完整接口及分数含义见[检索 API](docs/retrieval.md)。

## 自然语言检索与推荐

首次使用时，将 [.env.example](.env.example) 复制为根目录 `.env`，填写 `LLM_API_KEY`、`LLM_API_URL` 和 `LLM_MODEL`；已有文件时直接编辑。

```powershell
Copy-Item .env.example .env
```

服务与模型需支持代码所用的结构化输出接口及参数，配置规则见[查询生成](docs/retrieval.md#querygenerator)。以下示例还需要向量化依赖；首次创建向量化器会加载模型，远程模型尚未缓存时会下载文件。

```python
from arxiv_paper_recommend.recommendation import RecommendationGenerator
from arxiv_paper_recommend.retrieval import SearchEngine

request = "How can retrieval improve language model reasoning?"
with SearchEngine() as engine, RecommendationGenerator() as generator:
    papers = engine.search(request, method="HRRF", limit=30)
    recommendations = generator.generate(request, papers, limit=10)
    for recommendation in recommendations:
        print(
            recommendation.rank,
            recommendation.paper.arxiv_id,
            recommendation.reason,
        )
```

`SearchEngine` 按需创建并复用组件；方法名不区分大小写。推荐生成也可消费直接调用 BM25 得到的候选。基础方法返回理由和原始检索对象；候选限制与输出校验见[推荐生成](docs/recommendation.md)。

需要同时核查推荐证据时，可改用 `arxiv_paper_recommend.recommendation.innovation` 中的 `EvidenceGroundedRecommendationGenerator`。它使用相同的 `generate(request, papers, limit=10)` 调用方式，每项结果额外包含 `evidence`，程序检查该证据是否为对应标题或摘要的精确连续原文。原文匹配不保证理由在语义上被证据支持；作者、日期、来源 URL 和完整引用系统仍待补充。接口与示例见[证据生成](docs/evidence_grounded_generation.md)。

已提供[生成对比实验脚本](experiments/generation/README.md)，用于保存固定检索候选，再配对运行基础方法与证据生成方法。正式质量、延迟与成本结论仍待评估。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [项目结构](structure.md) | 模块职责、依赖方向与数据流 |
| [检索 API](docs/retrieval.md) | 检索接口、七种方法、分数与资源管理 |
| [推荐生成](docs/recommendation.md) | 生成接口、候选与响应校验 |
| [证据生成](docs/evidence_grounded_generation.md) | 原文证据接口、精确匹配与语义验证边界 |
| [生成对比实验](experiments/generation/README.md) | 固定候选、配对生成与 JSONL 输出 |
| [向量化](docs/vectorization.md) | 模型、设备、编码和查询指令 |
| [数据与索引构建](docs/data.md) | 来源、统计、存储格式和离线构建 |
| [接口迁移](docs/migration.md) | 不兼容变更及历史迁移记录 |
| [项目目标与进度](docs/project.md) | 课程要求、完成情况与剩余交付 |
| [实验与评估](docs/evaluation.md) | 人工标注、指标与待执行实验 |

[查询示例](data/example_queries.json)包含 40 条中英文研究请求，覆盖十个主题，尚无人工相关性标注。
