# Generation Comparison Experiment

本目录保存论文推荐生成模块的可复现实验代码，用于比较基础
`RecommendationGenerator` 与证据增强的
`EvidenceGroundedRecommendationGenerator`。

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

准备固定候选：

`uv run python experiments/generation/prepare_candidates.py --method HRRF --query-limit 1`

运行两种生成方法：

`uv run python experiments/generation/run_comparison.py --recommendation-limit 3`

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
