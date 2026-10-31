# 项目结构与模块职责

本文面向维护者，定义模块职责、依赖方向和数据流。安装使用见 [README](README.md)，公开契约见[检索 API](docs/retrieval.md)、[推荐生成](docs/recommendation.md)与[证据生成](docs/evidence_grounded_generation.md)。结构变更时同步更新本文。

## 目录

```text
arxiv_paper_recommend/
├── README.md / structure.md / AGENTS.md
├── pyproject.toml / uv.lock / .env.example
├── docs/                           # API、数据、模型、迁移、课程与评估说明
├── experiments/
│   ├── README.md                   # 历史检索实验材料与生成实验说明
│   ├── results/                    # 旧版冻结输入、运行结果、指标与实验报告
│   └── generation/                # 可复现的生成对比脚本与说明
│       ├── README.md
│       ├── prepare_candidates.py
│       └── run_comparison.py
├── data.db                         # 历史数据，不属于当前默认数据流
├── data/
│   ├── raw/papers.jsonl
│   ├── raw/vectors.npz
│   ├── documents.db
│   ├── documents.faiss
│   └── example_queries.json
└── src/arxiv_paper_recommend/
    ├── __init__.py                 # 延迟导出 PaperRecommender，无命令入口
    ├── pipeline.py                # SD Cascade + 证据生成的统一 Python API
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
    │   ├── search_engine.py / _ranking.py / _scoring.py
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
    │   └── innovation/
    │       ├── fc_hnsw.py         # 独立 FC-HNSW，默认 dense 组件
    │       ├── hprf.py            # 分层排名百分位加权混合检索
    │       └── _facet_coverage.py # 独立 FC-HNSW 原始覆盖分数
    ├── recommendation/
    │   ├── base.py                # 共同接口、逐篇并发判断、引用校验与集中排序
    │   ├── models.py              # 两种实现共用的推荐结果
    │   ├── generation/generator.py # 基于完整标题和摘要的判断提示词
    │   ├── citations/             # 规划占位
    │   └── innovation/
    │       ├── models.py          # 统一结果模型的兼容名称
    │       └── evidence_generator.py # 仅依据标注证据解释判断的提示词
    └── web/                       # 规划占位
```

各包包含相应的 `__init__.py`。复数形式的 `experiments/` 保存纳入 Git 的实验代码及 `experiments/results/` 中的检索代理实验材料。生成实验默认输出保存在单数形式的本地 `experiment/generation/`，与 `data/statistics.md` 一同被 Git 忽略，不能作为仓库已有正式评估结果的依据。

## 职责与依赖

| 模块 | 职责 | 依赖边界 |
| --- | --- | --- |
| `common` | 资源所有权、上下文管理、基本输入与有限权重校验 | 仅依赖标准库 |
| `config` | 默认数据路径、构造时读取 LLM 配置 | 导入时不读取 `.env`、不创建客户端 |
| `llm` | 共享结构化调用、按调用设置 reasoning、格式校验失败时最多调用三次纠正、解析结果检查及客户端关闭 | 依赖 `common`、`config`；不包含业务提示词 |
| `pipeline` | 校验研究需求与数量，串联 SD Cascade 检索和证据生成，返回 JSON 可序列化字典 | 通过 `SearchEngine` 和 `EvidenceGroundedRecommendationGenerator` 公开接口编排；延迟创建、复用并仅关闭自建组件，不实现排名或引用校验 |
| `corpus` | 导入准备好的语料、分别构建标题／摘要单字段 FTS5 索引，以及 FAISS 索引 | 不执行在线检索；采集清洗和向量归档流程尚未实现 |
| `vectorization` | 模型加载、批量编码、池化与归一化 | 不保存归档或构建索引 |
| `retrieval.sparse / dense` | 召回和指定 ID 评分 | 不调用 LLM；查询类型和原始分数尺度由算法决定 |
| `retrieval.hybrid` | 共享混合检索基类及 HRRF、SemanticRRF、SD Cascade 的候选选择、重排和融合，默认 dense 为 FC-HNSW | 通过稀疏／稠密组件公开接口协作；SD Cascade 使用 dense `score()`，原始内积通过 `raw_score()` 获取，缓存仅限单次搜索 |
| `retrieval.innovation` | 独立 FC-HNSW、原始覆盖评分及 HPRF 分层百分位混合检索 | FC-HNSW 继承多查询 HNSW，其原始覆盖评分不依赖 hybrid；HPRF 继承 HybridRetriever，包级延迟导出避免循环依赖 |
| `retrieval.query` | 从研究请求生成关键词及语义查询 | 依赖共享 LLM 层，不执行检索或向量化 |
| `retrieval.search_engine` | 注册方法、生成输入、延迟创建和复用组件 | 不承担算法内部排名逻辑 |
| `recommendation.models` | 统一推荐记录及原始检索结果关联，理由内含引用标记 | 不依赖 LLM 模块 |
| `recommendation.base` | 共同抽象接口、前 30 条去重、最多 5 个并发逐篇判断、精确引用校验及单篇最多三次纠正、集中筛选排序 | 依赖共享 LLM 层及公开检索结果模型；逐篇判断关闭 thinking 并设置 temperature=0，集中排序使用 thinking=high，不访问检索内部工具 |
| `recommendation.generation` | 基于完整标题和摘要解释相关性的判断提示词 | 实现共同基类，统一返回 `PaperRecommendation` |
| `recommendation.innovation.models` | `EvidenceGroundedRecommendation` 作为统一结果模型的别名 | 模型定义不依赖 LLM；经 `innovation` 包导入时会同时加载生成器 |
| `recommendation.innovation.evidence_generator` | 要求理由具体主张仅由标注原文支持的判断提示词 | 实现共同基类，复用相同的流程与校验；不自动验证语义支持性 |
| `experiments/results` | 保存旧版 BM25、HNSW、HRRF、FC-HRRF 的输入、结果与报告 | 旧脚本已删除；主题代理标签不能作为当前 HPRF 或人工相关性评估结论 |
| `experiments/generation` | 固定检索候选导出、基础与证据生成的配对运行、状态与耗时记录 | 通过公开 API 调用检索和生成；不实现人工标注或质量指标汇总 |
| `recommendation.citations / web` | 保留完整引用和界面扩展边界 | 当前为空包，不代表已有功能 |

`Retriever` 统一稀疏／稠密的搜索与评分契约；`HNSWRetriever.raw_score()` 提供指定 ID 的精确内积，`MultiQueryHNSWRetriever.raw_recall()` 提供完整多查询召回并集及主查询精确分数。原 HNSW 的 `score()`、多查询 HNSW 的 `recall()` 分别委托这两个原始入口，多查询 `search()` 每分面召回 `ceil(limit * sqrt(m))` 篇后按请求数量截取并使用原始内积，`m` 为去重后的分面数。`FCHNSWRetriever.search()`、`score()`、`recall()` 统一使用主查询原始内积与分面非负内积几何平均的加权分数，分数不依赖候选集合；`recall()` 保留完整并集，`search()` 默认每分面深度同为 `ceil(limit * sqrt(m))`，显式 `candidate_limit` 直接覆盖深度，独立 dense 无 100 条上限。`recall()`、`raw_recall()` 的 `limit` 直接表示每分面深度，不再次缩放。`HybridRetriever` 独立定义混合搜索，不提供单独评分接口。共享限值位于 `retrieval.constants`，混合算法专用预算位于 `retrieval.hybrid.constants`，排序、排名百分位与 RRF 位于 `_ranking`，原始评分缓存位于 `_scoring`，FC-HNSW 原始覆盖评分位于 `innovation._facet_coverage`，权重校验位于 `common.validation`，缓存仅限单次调用。

`SearchEngine.search()` 默认 `method="FC_HNSW"`，生成分面查询和主查询；默认 dense 组件也供混合检索复用。显式 `HNSW`、`MULTI_QUERY_HNSW` 在共享组件上分别调用原类的搜索实现，保留原算法；注入组件继续覆盖默认组件，资源由调用方管理。HPRF 沿用 HRRF 每路最多 `limit` 篇的候选预算，在各路内部以 BM25 与主查询原始内积的百分位加权重排，再对两路新排名的百分位加权。HRRF 对应的三处融合使用加权 RRF。两者的引擎和底层搜索均接受三个默认 `0.5` 的权重：`sparse_semantic_weight`、`dense_semantic_weight`、`fusion_weight`；HRRF 默认排序不变，最终分数为此前等权相加版本的一半。其他引擎方法忽略这些权重，自定义处理器沿用原有签名。

SD Cascade 先按 `max(1, ceil(sqrt(limit × N)))` 的预算执行 BM25 召回，`N` 为稀疏检索器的数据库文档总量；再对全部去重候选调用 dense `score()`，排除不可评分项，按分数降序、文档 ID 升序返回前 `limit` 篇。默认 FC-HNSW 接收分面查询并计算 FC 分数；注入原 HNSW 或多查询 HNSW 时使用主查询原始内积。空分面序列仅按主查询原始内积评分，不执行全库稠密召回。

## 数据流

离线构建需显式调用 `build()` 或对应模块命令：

```text
papers.jsonl ── DatabaseBuilder ──> documents.db
vectors.npz + documents.db ── IndexBuilder ──> documents.faiss
```

构建器分别位于 `corpus.build_database`、`corpus.build_index`，不在包根导出。导入构建模块不会读取语料或写入数据。

`documents.db` 保存论文表 `documents` 与两张外部内容 FTS5 索引：`documents_title_fts(title)`、`documents_abstract_fts(abstract)`。两张索引均以 `documents.id` 为 rowid，使用 `porter` 分词器，分别维护长度统计和 IDF。稀疏检索统一计算 `title_weight × 标题 BM25 + 摘要 BM25`，默认 `1:1`；搜索与补评分共用私有 SQL 构造逻辑，标题权重乘在标题最终分数上。搜索先合并全部匹配论文的分数，再排序和截取结果，同分按整数 ID 升序。

数据库构建在单一事务内导入、重建两张索引并删除旧 `documents_fts`。旧数据库需显式构建升级；在线连接保持只读。仅重建 FTS 且论文内容、ID 不变时无需重建 FAISS。

统一入口 `PaperRecommender.recommend()` 使用已有数据库和索引，执行以下在线流程：

```text
研究请求 ── PaperRecommender（先校验全部参数）
              ├── SearchEngine.search(method="SD_CASCADE", limit=candidate_limit)
              │      └── 查询生成 → BM25 召回 → 默认 FC 分数重排
              ├── EvidenceGroundedRecommendationGenerator.generate(limit=limit)
              │      └── 逐篇判断与引用校验 → 集中筛选排序
              └── dict：request + recommendations（rank / paper / reason）
```

统一入口默认最多向生成阶段提供 30 篇检索结果、返回最多 10 篇推荐，数量参数分别允许 1～30 与 1～10。检索为空时跳过推荐生成器的创建；初筛全部不推荐时同样返回空推荐列表。输出转换为普通字典及列表，保留论文信息、检索分数和理由中的引用标记；异常沿用底层行为直接传播。详细参数与使用示例见 [README](README.md#统一全流程接口)。

底层检索与生成接口仍可分别调用：

```text
研究请求 ── SearchEngine ── QueryGenerator ── terms / recall_queries / query
                    └── 方法分派
                        ├── 稀疏检索 ── documents.db
                        ├── 稠密检索 ── 默认 FC-HNSW，保留 HNSW / 多查询 HNSW
                        │              Qwen3Vectorizer + documents.faiss + documents.db
                        ├── 混合检索 ── 共享稀疏与稠密组件
                        └── HPRF ── HRRF 候选预算 + 原始分数补齐
                                   ── 两路内部百分位加权重排
                                   ── 两路新排名百分位加权
                                      ↓
                               list[SearchResult]
研究请求 + 检索结果 ── BaseRecommendationGenerator（共同编排）
                       ├── RecommendationGenerator（基础判断提示词）
                       └── EvidenceGroundedRecommendationGenerator（证据约束提示词）
                             ↓
                    前 30 条候选按 arXiv ID 去重
                             ↓
                    逐篇判断（最多并发 5 个，thinking=none，temperature=0）
                             ↓
                    bool + reason（内含 1～2 处引用并校验原文）
                             ↓
                    仅初筛推荐项 ── 一次集中筛选排序（thinking=high）
                             ↓
                    list[PaperRecommendation]，数量 min(limit, 初筛推荐数)
```

调用方可自行衔接下游生成，也可使用统一入口自动衔接 SD Cascade 与证据生成。两种生成器均返回统一推荐记录，`reason` 使用 `[[cite]]原文片段[[/cite]]` 标注 1～2 处标题或摘要中的精确连续原文。初筛推荐数为 0 时跳过排序；否则第二阶段只返回有序 ID，理由沿用初筛结果。精确匹配不保证理由在语义上被证据支持，当前结果均不补充作者、日期、URL 或完整引用元数据。

[历史检索实验材料](experiments/README.md)保留 BM25、HNSW、HRRF 与旧 FC-HRRF 的输入、结果和报告，位于 `experiments/results/`；旧实验脚本已删除，当前没有 HPRF 专用检索实验入口。历史材料属于旧版算法，不能作为 HPRF 的效果或延迟结论；主题代理标签仅用于主题级比较，不能替代人工相关性评估。

[生成对比实验](experiments/generation/README.md)先通过 `prepare_candidates.py` 将检索候选保存为 JSONL，再通过 `run_comparison.py` 让两种生成器读取相同候选并记录输出。默认文件为本地 `experiment/generation/candidates.jsonl` 与 `comparison.jsonl`；两类实验脚本均不属于库的在线调用链。

## 维护约定

- 存储层 `documents.id`、两张 FTS5 索引的 rowid、FAISS ID 与 `SearchResult.document_id` 对应；存储层 `doc_id` 映射为 `SearchResult.arxiv_id`，保留论文版本号。
- 仅关闭自建资源；注入组件由调用方管理。关闭幂等，关闭后的对象不能继续操作。
- 包级导入为业务 API 推荐入口；项目根包延迟导出 `PaperRecommender`，仅导入根包不会加载检索或 LLM 模块；检索根包延迟导出引擎，推荐根包延迟导出共同基类与基础生成器。导入 `SearchResult` 或统一推荐模型 `PaperRecommendation` 不加载 FAISS 或 LLM 客户端模块；`recommendation.innovation` 当前直接导出模型与生成器，会加载共享 LLM 模块。
- PyTorch、Transformers 仅在实例化向量化器时导入。模型和索引跨查询复用。
- 文件使用 snake_case，类使用 PascalCase；Python 注释与文档字符串以清晰英文为主，使用文档以中文为主；编辑后执行 `ruff format`。
