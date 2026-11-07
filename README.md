# arxiv-paper-recommend

基于 arXiv 标题与摘要的检索增强论文推荐项目，用于 AI6127（2026）课程作业。通过 Python API 检索论文，并基于候选内容生成最多 10 条推荐及简短理由。

## 项目能力

已有 10,000 篇论文的数据快照、SQLite FTS5 数据库和 FAISS 向量索引。支持关键词检索、本地 Qwen3 向量化、LLM 查询生成及推荐理由生成。下游生成提供基础方法和 Evidence-Grounded 方法，两者都先逐篇判断，再集中筛选排序。基础方法依据完整标题和摘要生成理由；Evidence-Grounded 方法要求理由包含可核查的标题／摘要原文引用。

| 检索方法 | 流程 | 分数方向 |
| --- | --- | --- |
| `BM25` | 标题、摘要分别计算 BM25，按 `1:1` 相加 | 越小越相关 |
| `TITLE_WEIGHTED_BM25` | 对标题侧最终 BM25 加权后与摘要分数相加，引擎默认 `1:1` | 越小越相关 |
| `HNSW` | 单查询近似向量召回 | 越大越相关 |
| `MULTI_QUERY_HNSW` | 多查询召回，合并后按主查询重排 | 越大越相关 |
| `FC_HNSW`（默认） | 完整多查询召回并集按主查询内积与分面非负内积几何平均加权重排 | 越大越相关 |
| `HRRF` | 两路候选分别重排，再融合排名 | 越大越相关 |
| `SEMANTIC_RRF` | 按逐篇语义相似度加权 RRF | 越大越相关 |
| `SD_CASCADE` | BM25 召回后对全部候选计算稠密 `score()`，降序返回前 `limit` 篇，默认使用 FC 分数 | 越大越相关 |
| `HPRF` | 沿用 HRRF 候选，分别按 BM25 与语义百分位重排两路，再加权融合两路排名百分位 | 越大越相关 |

不同方法的分数不能直接比较。目前没有正式质量或性能评估结论。独立来源引用、Web 界面和完整采集清洗流程仍待实现，详见[项目目标与进度](docs/project.md)。

稀疏检索为标题、摘要分别建立单字段 FTS5 索引，各自使用独立的长度统计和 IDF。总分为 `title_weight × 标题 BM25 + 摘要 BM25`；直接调用标题加权检索器可在每次调用中指定权重，详见[检索 API](docs/retrieval.md#titleweightedbm25retriever)。

## 安装

要求 Python >= 3.14、uv，以及支持 FTS5 的 SQLite >= 3.35（评分使用物化 CTE）。在项目根目录运行：

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

从旧版双字段 FTS5 数据库升级时，只需运行上述 `build_database` 命令，即可在保留已有论文整数 ID 的同时重建标题、摘要索引并移除旧索引；本次仅索引结构与评分方式变化，无需重建 FAISS。检索器只读打开数据库，不会自动升级旧索引。

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

默认 dense 实现为独立的 `FCHNSWRetriever`，位于 `retrieval.innovation`，可直接使用而无需 LLM：

```python
from arxiv_paper_recommend.retrieval.innovation import FCHNSWRetriever

with FCHNSWRetriever() as retriever:
    papers = retriever.search(
        "retrieval for language model reasoning",
        limit=10,
        recall_queries=["evidence retrieval", "multi-step reasoning"],
    )
```

该示例需要向量化依赖。`search()`、`score()` 和 `recall()` 统一使用 FC 分数：
`coverage_weight × GM(max(0, 分面内积)) + (1 − coverage_weight) × 主查询内积`，
默认权重为 `0.5`。`score()` 和 `recall()` 同样支持 `recall_queries`、`coverage_weight`；
需要主查询原始内积或完整原始召回并集时，分别使用 `raw_score()`、`raw_recall()`。

## 自然语言检索与推荐

首次使用时，将 [.env.example](.env.example) 复制为根目录 `.env`，填写 `LLM_API_KEY`、`LLM_API_URL` 和 `LLM_MODEL`；已有文件时直接编辑。

```powershell
Copy-Item .env.example .env
```

服务与模型需支持代码所用的结构化输出接口及参数，配置规则见[查询生成](docs/retrieval.md#querygenerator)。以下示例还需要向量化依赖；首次创建向量化器会加载模型，远程模型尚未缓存时会下载文件。

默认向量模型固定为 `Qwen/Qwen3-Embedding-0.6B` 的指定版本，优先逐文件读取本地 HF 缓存，只匿名下载缺失文件，不使用本机 HF 登录凭据。首次使用需联网下载约 1.2 GB 的模型文件；文件完整后，向量模型加载及编码均可离线进行。LLM 查询与推荐生成仍需连接 `.env` 配置的服务。自定义模型来源保留原有加载及认证方式。

统一查询生成最多重试 5 次，含首次最多请求 6 次；旧独立关键词／查询生成、每篇论文判断和集中排序各自最多重试 3 次，含首次最多请求 4 次。网络/API 异常、结构解析、缺少结构化结果、理由非空及排序等输出校验失败共用各次调用的预算；引用标记与原文不再校验。SDK 内置重试已关闭，不存在嵌套重试。校验失败时附带最近一次错误反馈立即重新生成；网络/API 异常保留反馈，按共享尝试序号等待 1、2、4 秒，统一查询生成可继续等待 8、16 秒。耗尽后共享层抛出最后一次异常；单篇判断由上层打印错误并按不推荐处理，其他阶段继续传播异常。

### 统一全流程接口

`PaperRecommender` 将查询生成、`SD_CASCADE` 检索和 `EvidenceGroundedRecommendationGenerator` 的两阶段生成串联起来，默认检索 30 篇候选并生成最多 10 篇推荐。返回普通 Python 字典，可直接序列化为 JSON：

```python
import json

from arxiv_paper_recommend import PaperRecommender

with PaperRecommender() as recommender:
    result = recommender.recommend(
        "How can retrieval improve language model reasoning?",
        candidate_limit=30,
        limit=10,
    )

print(json.dumps(result, ensure_ascii=False, indent=2))
```

```text
PaperRecommender(*, search_engine=None, generator=None)

recommend(request, *, candidate_limit=30, limit=10, max_terms=12, max_queries=4)
    -> dict[str, object]
```

| 参数 | 默认值 | 约束与含义 |
| --- | --- | --- |
| `request` | 必填 | 研究需求字符串，去除首尾空白后不能为空 |
| `candidate_limit` | `30` | 1～30 的整数，进入生成阶段的检索结果上限 |
| `limit` | `10` | 1～10 的整数，最终推荐上限 |
| `max_terms` | `12` | 5～20 的整数，查询生成的关键词数量上限 |
| `max_queries` | `4` | 至少为 1 的整数，查询生成的分面数量上限 |

所有整数参数均拒绝布尔值；全部参数在创建资源和执行检索前校验。最终推荐数量为 `min(limit, 初筛推荐数)`，可以小于 `limit`。

返回结构与字段类型如下，`abstract` 和 `score` 的 `None` 在 JSON 中对应 `null`：

```text
{
    "request": str,
    "recommendations": [
        {
            "rank": int,
            "paper": {
                "document_id": int,
                "arxiv_id": str,
                "title": str,
                "abstract": str | None,
                "score": float | None
            },
            "reason": str
        }
    ]
}
```

`request` 为去除首尾空白后的需求，`recommendations` 按最终排名排列，`rank` 从 1 开始；`score` 保留检索分数，`reason` 完整保留 `[[cite]]...[[/cite]]` 原文引用标记。结果只包含最终推荐，不附带全部候选。

检索为空或初筛没有成功推荐项时，返回 `{"request": "...", "recommendations": []}`；检索为空时不会创建推荐生成器。单篇评估重试耗尽后打印论文 arXiv ID、异常类型与错误，按不推荐处理并继续其他候选。仅一篇通过时直接返回该篇并设为第 1 名，跳过 LLM 排序。输入错误、资源初始化、检索／查询生成异常和集中排序最终异常继续抛出。

同一个 `PaperRecommender` 可多次调用，按需创建并复用引擎及生成器，退出 `with` 或调用 `close()` 时关闭自建资源，重复关闭无额外效果；关闭后不能继续调用。通过构造参数注入的 `SearchEngine` 和 `EvidenceGroundedRecommendationGenerator` 由调用方管理生命周期。默认组件继续使用现有数据路径和 `.env`／环境变量配置。

### 分别调用检索与生成

也可继续使用底层接口，自行选择检索方法和生成器：

```python
from arxiv_paper_recommend.recommendation import RecommendationGenerator
from arxiv_paper_recommend.retrieval import SearchEngine

request = "How can retrieval improve language model reasoning?"
with SearchEngine() as engine, RecommendationGenerator() as generator:
    papers = engine.search(request, limit=30)  # 默认 FC_HNSW
    recommendations = generator.generate(request, papers, limit=10)
    for recommendation in recommendations:
        print(
            recommendation.rank,
            recommendation.paper.arxiv_id,
            recommendation.reason,
        )
```

`SearchEngine` 按需创建并复用组件；省略 `method` 时使用 `FC_HNSW`，方法名不区分大小写。显式指定 `HNSW` 或 `MULTI_QUERY_HNSW` 仍执行原检索算法；混合检索默认复用 FC-HNSW 组件，执行各自的候选选择与融合规则。多查询稠密和全部混合方法将关键词、分面查询与主查询合为一次结构化生成，使用 `thinking=high`、省略 `temperature`，最多重试 5 次（含首次最多 6 次请求）。BM25、标题加权 BM25 和单查询 HNSW 保留专用生成，继续关闭 thinking、设置 `temperature=0`，最多重试 3 次。

推荐生成也可消费直接调用 BM25 得到的候选。两种生成器均取前 30 条候选并去重，以最多 5 个并发请求逐篇判断；只有至少两篇通过时才调用一次 API 筛选排序，最终返回 `min(limit, 成功初筛推荐数)` 项。逐篇判断关闭 thinking，设置 `temperature=0`；集中排序使用 `thinking=high`，不发送 `temperature`。两阶段仍各自最多重试 3 次。候选限制与输出校验见[推荐生成](docs/recommendation.md)。

选择 `HRRF` 或 `HPRF` 时可分别配置稀疏候选内的语义占比、稠密候选内的语义占比和最终稠密路占比，三个权重默认均为 `0.5`，范围均为 `[0, 1]`：

```python
with SearchEngine() as engine:
    papers = engine.search(
        request, method="HRRF", limit=30,  # HPRF 支持相同权重
        sparse_semantic_weight=0.5,
        dense_semantic_weight=0.5,
        fusion_weight=0.5,
    )
```

底层入口 `retrieval.hybrid.HRRFRetriever.search()` 和 `retrieval.innovation.HPRFRetriever.search()` 支持相同三个权重，分别用于 RRF 和排名百分位融合；公式与边界规则见[检索 API](docs/retrieval.md#混合检索)。HRRF 默认排序保持不变，最终分数因权重归一化变为原来的一半。旧 `FACET_COVERAGE` 入口已移除，迁移方式见[接口迁移](docs/migration.md)。

两种生成器统一返回 `PaperRecommendation`。基础 `RecommendationGenerator` 的提示词只要求依据完整标题和摘要解释相关性，不提及引用。`arxiv_paper_recommend.recommendation.innovation` 中的 `EvidenceGroundedRecommendationGenerator` 使用相同的调用方式，提示词要求 `reason` 包含 1～2 处 `[[cite]]原文片段[[/cite]]` 引用，且理由中的论文具体主张由标注片段直接支持。程序保留模型生成的理由及其中可能出现的标记，不校验引用数量、配对、嵌套、引用内容非空或原文匹配；理由非空、结构与排序结果校验保留。引用格式、原文对应关系和语义支持性需单独评估；作者、日期、来源 URL 和完整引用系统仍待补充。原有独立 `evidence` 字段已移除，接口与示例见[证据生成](docs/evidence_grounded_generation.md)。

已提供[生成对比实验脚本](experiments/generation/README.md)，用于保存固定检索候选，再配对运行基础方法与证据生成方法。正式质量、延迟与成本结论仍待评估。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [项目结构](structure.md) | 模块职责、依赖方向与数据流 |
| [检索 API](docs/retrieval.md) | 检索接口、九种方法、分数与资源管理 |
| [推荐生成](docs/recommendation.md) | 生成接口、候选与响应校验 |
| [证据生成](docs/evidence_grounded_generation.md) | 原文证据接口、引用格式与语义验证边界 |
| [生成对比实验](experiments/generation/README.md) | 固定候选、配对生成与 JSONL 输出 |
| [向量化](docs/vectorization.md) | 模型、设备、编码和查询指令 |
| [数据与索引构建](docs/data.md) | 来源、统计、存储格式和离线构建 |
| [接口迁移](docs/migration.md) | 不兼容变更及历史迁移记录 |
| [项目目标与进度](docs/project.md) | 课程要求、完成情况与剩余交付 |
| [实验与评估](docs/evaluation.md) | 人工标注、指标与待执行实验 |

[查询示例](data/example_queries.json)包含 40 条中英文研究请求，覆盖十个主题，尚无人工相关性标注。
四方法代理实验的运行方式、结果文件和标签限制见[检索创新实验](experiments/README.md)。
