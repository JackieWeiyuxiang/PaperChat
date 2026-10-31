# 检索创新实验

`run_facet_coverage.py`只比较 BM25、HNSW、HRRF 和 FC-HRRF 四种方法。它读取
`data/example_queries.json`，缓存 LLM 生成的关键词和分面，预计算查询向量，并在
10 条开发查询上选择覆盖权重，再对其余 30 条查询报告结果。

当前保存在 `experiments/results/` 的结果来自旧版双字段 FTS5 BM25。2026-10-31 起，稀疏检索改为标题、摘要分别使用独立统计计算 BM25 后加权相加；BM25 及依赖它的混合检索排名可能变化，已有结果、延迟和覆盖权重选择不能作为新实现的评估结论。本次保留历史结果，未重新运行实验。重新执行以下命令会使用新实现并写入结果目录，复现旧结果需使用对应旧代码和索引。

```powershell
uv sync --extra vectorization
uv run python experiments/run_facet_coverage.py --env-file .env
```

脚本默认写入 `experiments/results/`：

- `query_inputs.json`：冻结的关键词、整合查询和分面；
- `query_embeddings.npz`：与归档查询指令一致的预计算查询向量；
- `runs.jsonl`：四种方法每条查询最多 Top-20；
- `query_metrics.csv`、`summary.csv`：逐查询及汇总指标；
- `metadata.json`、`report.md`：实验配置、限制与可读报告。

示例查询没有人工相关性标签。当前脚本将论文 `collection_topic` 与查询 `topic`
相同作为二值代理相关性，所以结果只能用于验证实现和比较主题级检索能力，不能替代
正式的盲法双人0/1/2相关性标注。查询向量预计算耗时单独记录；检索延迟不包含 LLM
调用和查询编码。
