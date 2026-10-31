# 实验材料

## 历史检索代理实验

`results/` 保留旧版 BM25、HNSW、HRRF、FC-HRRF 四方法实验的冻结输入、结果、
指标和报告。旧 `run_facet_coverage.py` 脚本及 Facet Coverage 混合检索器已移除，
当前没有 HPRF 检索实验入口；需要复现历史实验时，应使用对应历史版本的代码和索引。

历史实验读取 `data/example_queries.json` 中的 40 条查询，按主题各取首条作为
10 条开发查询，其余 30 条作为测试集；开发集在 `0.1`～`0.5` 选择覆盖权重。
所保存结果使用旧版双字段 FTS5 BM25，FC-HRRF 在 HRRF 固定候选上以各分面
排名百分位的最小值作为覆盖度，原始公式及配置见[历史报告](results/report.md)。

当前稀疏检索分别计算标题、摘要 BM25 后加权相加；新的 `HPRF` 沿用 HRRF 候选，
分别对两路候选进行 BM25 与语义百分位加权重排，再融合两路新排名的百分位。
三个权重均默认 `0.5`，定义见[检索 API](../docs/retrieval.md#hprfretriever)。
HPRF、独立 FC-HNSW 及当前默认混合候选均未重新评估，历史结果不代表它们的效果或延迟。
本次未执行实验，也未改写历史结果和报告。

历史材料包括：

- `query_inputs.json`：冻结的关键词、整合查询和分面；
- `query_embeddings.npz`：预计算查询向量；
- `runs.jsonl`：四种方法每条查询最多 Top-20；
- `query_metrics.csv`、`summary.csv`：逐查询及汇总指标；
- `metadata.json`、`report.md`、`summary_report.md`：旧版配置、限制与实验报告。

示例查询没有人工相关性标签。历史实验将论文 `collection_topic` 与查询 `topic`
相同作为二值代理相关性，只能用于主题级比较，不能替代人工相关性评估。
历史检索延迟不包含 LLM 调用和查询编码，查询向量预计算耗时单独记录。

## 生成对比实验

现有[生成对比脚本](generation/README.md)可导出固定检索候选，再配对运行基础生成与
证据生成。它通过公开 `SearchEngine` 接口选择检索方法，仍可使用当前注册的 `HPRF`；
三个 HPRF 权重使用默认值，脚本未增加专属调参选项。
