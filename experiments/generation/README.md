# Generation Comparison Experiment

本目录保存论文推荐生成模块的可复现实验代码，用于比较基础
`RecommendationGenerator` 与证据增强的
`EvidenceGroundedRecommendationGenerator`。

方法及验证边界见[证据生成](../../docs/evidence_grounded_generation.md)，正式评价设计见[实验与评估](../../docs/evaluation.md)。

## 实验流程

```text
共享研究需求
    ↓
固定检索方法产生候选论文
    ↓
将候选保存为 candidates.jsonl
    ↓
Baseline 与 Innovation 读取完全相同的候选
    ↓
将两套生成结果保存为 comparison.jsonl
    ↓
自动检查、人工盲评与指标计算

```

## 文件

- `prepare_candidates.py`：读取共享查询，使用运行时指定的检索方法并保存固定候选。
- `run_comparison.py`：让 Baseline 与 Innovation 使用完全相同的固定候选，并保存两套输出。

## 公平比较原则

Baseline 与 Innovation 必须固定相同的：

- 用户请求；
- 候选论文及顺序；
- 候选规模；
- 推荐数量上限；
- 生成模型和 API 配置。

唯一改变的是下游生成方法。

## 本地输出

运行结果默认保存在 `experiment/generation/`。

单数形式的 `experiment/` 已被 `.gitignore` 忽略，不上传候选摘要和模型输出。复数形式的 `experiments/` 保存可复现实验代码与说明，需要提交到 Git。

## 使用方式

正式运行前需要确认检索方法、query instruction 和生成模型。

在项目根目录完成 `uv sync` 并配置 `.env` 中的 `LLM_API_KEY`、`LLM_API_URL` 和 `LLM_MODEL`。下例使用 HRRF，还需 `uv sync --extra vectorization` 及匹配的数据、向量索引。即使选择 BM25，本脚本通过 `SearchEngine` 生成查询，仍需要 LLM；完整配置见[项目说明](../../README.md)。

准备固定候选：

```powershell
uv run python experiments/generation/prepare_candidates.py --method HRRF --query-limit 1
```

运行两种生成方法：

```powershell
uv run python experiments/generation/run_comparison.py --recommendation-limit 3
```

准备阶段默认读取 `data/example_queries.json` 的 `questions` 列表，使用每条记录的 `query_en`；通过 `--queries`、`--method`、`--candidate-limit` 和 `--output` 指定输入与导出设置。候选默认上限为 30，两种生成器均只消费前 30 个候选并按 arXiv ID 去重；导出更多候选不会扩大生成输入。

比较阶段通过 `--candidates` 读取固定候选，可用 `--model` 覆盖 `LLM_MODEL`，`--recommendation-limit` 接受 1～10。两阶段均可用 `--query-limit` 缩小运行规模；省略时处理全部输入。两脚本都默认拒绝覆盖已有输出，显式传入 `--overwrite` 才会替换。运行会调用所配置的 LLM API。

## 输出与错误处理

每个查询完成后追加一行 JSON 并刷新文件，默认路径为：

| 文件 | 主要内容 |
| --- | --- |
| `experiment/generation/candidates.jsonl` | `query_id`、`request`、可选问题元数据、`retrieval` 配置与耗时、按检索顺序排列的 `candidates` |
| `experiment/generation/comparison.jsonl` | 请求与检索元数据、`candidate_count`、生成模型与数量上限，以及 `generation.baseline` 和 `generation.evidence_grounded` 两套结果 |

每个生成结果包含 `status`、`elapsed_seconds` 和 `recommendations`；成功时另含 `returned_count`，捕获到 `RuntimeError` 时记录 `error_type` 和 `error_message`。推荐项保留论文 ID、标题、摘要、原候选排名、检索分数和理由；证据生成结果额外包含 `evidence`。`candidate_count` 是输入文件中的候选数量，不代表去重并截断后的实际模型输入数量。

比较脚本只捕获生成调用中的 `RuntimeError`；其他客户端或解析异常会直接传播并中断运行。逐条刷新可保留此前完成的查询，但当前不支持断点续跑；同一查询的两种方法全部运行结束后才写入比较记录。

证据精确匹配在生成器内部执行，校验失败会使整次生成报错，不保存失败响应的逐项内容。脚本不自动汇总成功率、证据通过率、质量指标或费用；正式统计需定义分母及失败处理，不能仅凭成功响应推断所有生成证据的通过率。

## 评价方向

自动记录或检查：

- Evidence 是否为标题或摘要中的连续原文；
- 生成成功率与错误类型；
- 返回推荐数量；
- 检索与生成耗时。

人工评价：

- 推荐论文与研究需求的相关性；
- 推荐理由的 faithfulness；
- 推荐理由的 answer relevance；
- Evidence 对 Reason 的支持程度。

## 当前状态

已完成：

- 固定候选导出框架；
- Baseline 与 Innovation 配对运行框架；
- JSONL 增量保存；
- 参数、语法和命令行接口检查。

尚未执行：

- 最终检索配置确认；
- 正式付费实验；
- 人工标注与一致率计算；
- 正式质量、延迟和成本结论。

在正式实验完成前，不应宣称 Innovation 已优于 Baseline。
