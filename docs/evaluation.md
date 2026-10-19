# 实验与评估

本文记录作业与提案要求对应的评估设计、实现和数据缺口。项目要求及完成情况见[项目目标与进度](project.md)，方法与分数定义见[检索 API](retrieval.md)。

## 仓库当前状态

仓库已实现 BM25、标题加权 BM25、HNSW、三种混合检索策略、查询生成和固定输入实验／指标脚本；但未提供 P2 的实验结果、人工标注集、相关性指标或生成质量结果。下文明确区分已实现工具和待提供数据，不依据缺失标签作出效果结论。

2026-10-19 从 `data-preprocessing` 同步的[查询示例](../data/example_queries.json)包含 40 条中英文研究请求，覆盖十个收录主题，可用于试用及后续评估选题；尚无相关性标签或目标论文集合，不能直接作为已有人工评估集。

## 人工评估集设计（待执行）

按提案，准备约 100 条不重复研究需求，对各方法候选取并集并按 `(query_id, doc_id)` 去重，再人工判断至少 1,000 个 query–paper 对。候选池应覆盖不同主题及具体约束，不能靠重复结果凑足标注量。用于改进方法的开发查询与用于最终报告的留出查询应分开，记录各自用途，避免在同一批标签上调参和报告结果。

相关性采用提案的三级标注：`0` 不相关、`1` 部分相关、`2` 高度相关。至少两名标注者独立阅读同一请求及论文标题、摘要；标注时隐藏检索方法、分数和排名。先记录原始标签并计算一致率，再对分歧进行讨论或仲裁，保存最终标签及说明。课程要求一致率至少 80%；建议另报告 Cohen's kappa 以补充反映偶然一致的影响。

下表给出各团队成员应共享的 JSONL 记录约定。固定输入、运行结果和候选池可由下方实验程序导出；人工标签仍需 P5 组织标注：

| 材料 | 建议字段 |
| --- | --- |
| 查询 | `query_id`、`request`、`split`、`topic`、`constraints` |
| 固定检索输入 | `query_id`、`terms`、`queries`、`score_query`、生成模型与提示词版本 |
| 各方法运行结果 | `run_id`、`query_id`、`method`、`rank`、`doc_id`、`score`、`score_type` |
| 原始标注 | `query_id`、`doc_id`、`annotator_id`、`relevance`、`note` |
| 仲裁后标签 | `query_id`、`doc_id`、`relevance`、`adjudication_note` |
| 推荐生成与证据 | `run_id`、`query_id`、推荐论文 ID、理由、引用摘要及支持性判断 |

检索相关性标签用于评估候选排序，不能单独替代对推荐理由的人工或模型辅助评估。生成完成后，仍需检查理由是否被摘要支持、推荐是否满足需求以及引用是否正确。

## 已实现的检索实验程序

`python -m arxiv_paper_recommend.evaluation.retrieval run` 接收 P2/P5 固定的检索输入，对相同的 `terms`、`queries` 和 `score_query` 运行 BM25、TITLE_BM25、HNSW、HRRF、SEMANTIC_RRF 与 SD_CASCADE，并导出完整排名及供标注使用的候选池。固定输入将检索排序比较与在线 LLM 改写分开；默认重复三次，先在计时外加载并预热向量模型。单次计时包含查询向量化和检索／重排，不包含数据库／索引／模型初始化或 LLM 查询生成。

`data/evaluation/retrieval_inputs.jsonl` 每行一个查询，字段如下：

```json
{"query_id":"heldout_001","request":"...","terms":["..."],"queries":["..."],"score_query":"..."}
```

运行示例：

```powershell
uv run python -m arxiv_paper_recommend.evaluation.retrieval run `
  --inputs data/evaluation/retrieval_inputs.jsonl `
  --runs data/evaluation/retrieval_runs.jsonl `
  --pool data/evaluation/candidate_pool.jsonl `
  --limit 20
```

Dense 与 hybrid 方法需要 `uv sync --extra vectorization` 和可用的本地 Qwen 模型。输入文件应在 held-out 查询上冻结，并与 P2 的 query/terms 生成流程一致；不能在看过其相关性标签后调整查询或 3:1 标题权重。

P5 将去重后的候选池分配人工标注，再提供最终 qrels，每行格式为 `{"query_id":"heldout_001","doc_id":"...","relevance":0}`，相关性为 0、1 或 2。使用下列命令生成 Precision@K、Recall@K、MRR@K、NDCG@K 及平均检索时间：

```powershell
uv run python -m arxiv_paper_recommend.evaluation.retrieval evaluate `
  --qrels data/evaluation/qrels.jsonl `
  --runs data/evaluation/retrieval_runs.jsonl `
  --output data/evaluation/retrieval_metrics.json
```

为保证比较公平，若某查询在任一方法的候选结果中仍有未标注论文，该查询会从所有方法的指标中一起排除，并在报告里计入 `unjudged_or_incomplete_query_count`；脚本不会默默把未标注论文当成不相关。正式报告前需确认被评方法、查询集和 K 一致，并补足完整候选池判断。

## P3 基础生成器

`RecommendationGenerator` 接收用户需求和 `SearchResult` 候选，最多输出 10 篇论文及各自一到两句的推荐理由。模型只能选择候选中的 `doc_id`，提示要求理由以论文标题和摘要为依据；返回值同时保留论文原始结果，供之后展示来源。该实现是下游 baseline，不包含 P4 负责的 self-check、引用支持性判定或 RAG 质量提升方法。

没有 LLM 凭证时不能实际生成理由或报告质量指标。生成评估应在 P5 选择的 held-out 查询及固定检索结果上运行，由 P4 与标注者判断 answer relevance、faithfulness 和 context relevance/precision；不能以结构化输出成功或模型自评替代人工质量结果。

## 检索指标与比较（待执行）

正式报告以 Top-10 目标为主，可同时报告 K=5、10、20。二值指标需事先固定相关阈值（例如标签 ≥1）；NDCG 使用完整的 0/1/2 分级标签，并明确增益定义。对同一批查询按查询平均，说明无相关论文或标签不足时的处理。

| 指标 | 报告口径 |
| --- | --- |
| Precision@K | 前 K 个位置中相关论文数除以 K；不足 K 篇时需明确缺失位置处理 |
| Recall@K | 前 K 篇相关论文数除以该查询候选池内已标注相关论文数；这是候选池召回率，不是全库或整个 arXiv 召回率 |
| MRR | 第一篇达到相关阈值的论文排名倒数；若只评估前 K 篇，应写为 MRR@K |
| NDCG@K | 用分级相关性衡量排序质量，理想排序基于共同标注候选池 |

未标注候选不应悄悄当作不相关；加入新方法或扩大返回深度后，应补齐所需标注或明确未判定比例及局限。候选池之外的相关论文未知，因此不能宣称已测得全库 Recall。

六种方法（BM25、TITLE_BM25、HNSW、HRRF、SEMANTIC_RRF、SD_CASCADE）可直接纳入比较；P3 的创新对照为 TITLE_BM25 与等权 BM25。多查询、整合查询重排、分层 RRF、语义权重和级联候选选择均已实现，尚不能把其组合收益归因于其中任一项。严格比较需固定语料、模型、检索输入和 K；不同方法当前的候选预算不同，需记录并分析这一影响。

还需区分两种实验：固定 `terms/queries/score_query` 比较检索与排序，以及包含实时 LLM 查询生成的端到端比较。当前 `SearchEngine` 的 HNSW 方法包含多查询合并与整合查询重排，不等同于底层 `HNSWRetriever.search` 的单查询近似召回。

## 推荐生成与性能评估（待执行）

固定检索结果、生成模型和基础提示词后，比较基础生成与下游创新，使用同一组研究需求和证据。若引入多项创新，分别记录单项和组合结果。推荐任务需说明采用哪些任务质量指标及其理由；没有参考推荐理由时，不应虚构参考答案并报告摘要指标。

生成评估至少覆盖理由的 faithfulness（每项主张是否有摘要依据）、answer relevance（是否回答研究需求）、context relevance/precision（上下文是否相关）及引用对应关系。可结合人工标注与 RAGAS 等工具，但评估实现和结果目前均未提供；若使用模型裁判，需保存裁判模型、版本和判定规则。

五查询的速度测量仍待完成。正式记录应包含运行时间、代码版本、依赖、数据／索引标识、向量模型 revision 与指令、LLM 配置、硬件、K、候选规模和缓存条件。分别统计模型加载等冷启动开销与复用组件后的查询耗时，区分 LLM 查询生成、查询编码、召回／重排和下游生成；报告重复次数、平均值或中位数及适当分位数，并保存每个查询的原始测量。

吞吐、成本与可扩展性需基于明确的并发、缓存和语料规模实验；本地模型的预计算时间不能从总成本中无说明地剔除，LLM token 与费用也不能仅按调用次数推断。仓库未提供这些测量，本页没有性能或质量结论。
