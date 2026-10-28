# 基于可验证证据的论文推荐生成

本文记录论文推荐生成阶段的下游创新、接口、验证边界及待执行评估。基础生成方法见[推荐生成](recommendation.md)，整体评估要求见[实验与评估](evaluation.md)，可复现实验脚本见[生成对比实验](../experiments/generation/README.md)。

## 研究动机

基础 `RecommendationGenerator` 从最多30篇检索候选中选择并排序论文，为每篇论文生成一至两句推荐理由。模型输入包含用户研究需求、论文 ID、标题和摘要。

基础方法要求理由仅依据候选论文的标题和摘要，但最终结果只返回 `reason`，没有同时提供能够让用户直接核查的原始证据。程序会检查论文 ID、重复项和空白理由，但不能验证理由中的具体说法是否真的存在于标题或摘要。

本项目因此实现 Evidence-Grounded Recommendation Generation。该方法要求生成模型为每篇推荐论文同时返回：

- 论文 ID；
- 标题或摘要中的连续原文证据；
- 基于该证据生成的推荐理由。

## 方法概览

整体流程如下：

用户研究需求
→ 固定检索方法返回候选论文
→ 截取前30篇并按照 arXiv ID 去重
→ LLM 选择并重新排序论文
→ 每篇论文复制一个连续原文 Evidence
→ 根据 Evidence 生成一句推荐理由
→ 程序检查 Evidence 是否确实存在于标题或摘要
→ 返回最多10篇推荐结果

生成模型只能选择输入候选中的论文，不允许使用外部知识或虚构论文。

## 公开接口

创新代码位于：

- `src/arxiv_paper_recommend/recommendation/innovation/models.py`
- `src/arxiv_paper_recommend/recommendation/innovation/evidence_generator.py`
- `src/arxiv_paper_recommend/recommendation/innovation/__init__.py`

公开类包括：

- `EvidenceGroundedRecommendation`
- `EvidenceGroundedRecommendationGenerator`

`EvidenceGroundedRecommendation` 包含以下字段：

- `rank`：从1开始的推荐排名；
- `paper`：原始 `SearchResult`，保留检索分数；
- `evidence`：标题或摘要中的连续原文；
- `reason`：解释论文与用户需求关系的推荐理由。

生成接口接收非空研究需求、`SearchResult` 列表和1至10之间的推荐数量。候选为空时直接返回空列表，不调用 LLM。

```python
from arxiv_paper_recommend.recommendation.innovation import (
    EvidenceGroundedRecommendation,
    EvidenceGroundedRecommendationGenerator,
)
```

```text
EvidenceGroundedRecommendationGenerator(*, model=None, client=None)
generate(request: str, papers: list[SearchResult], limit=10) -> list[EvidenceGroundedRecommendation]
```

构造参数、LLM 配置和客户端所有权沿用[基础生成器](recommendation.md)：显式 `model` 覆盖配置；注入客户端由调用方关闭，自建客户端通过生成器的上下文管理或 `close()` 关闭。关闭后不能继续生成。结果为冻结数据类，`paper` 保留原始候选引用。当前 `innovation` 包直接导出生成器，导入该包时会加载共享 LLM 模块。

## 使用示例

以下先用 BM25 检索，无需向量化依赖；生成阶段仍需有效 LLM 配置。安装及 `.env` 配置见 [README](../README.md)。

```python
from arxiv_paper_recommend.recommendation.innovation import (
    EvidenceGroundedRecommendationGenerator,
)
from arxiv_paper_recommend.retrieval.sparse import BM25Retriever

request = "How can retrieval improve language model reasoning?"
with BM25Retriever() as retriever, EvidenceGroundedRecommendationGenerator() as generator:
    papers = retriever.search(
        ["retrieval augmented generation", "reasoning"], limit=30,
    )
    recommendations = generator.generate(request, papers, limit=10)
    for item in recommendations:
        print(item.rank, item.paper.arxiv_id)
        print(item.evidence)
        print(item.reason)
```

## 结构化生成

LLM 的结构化输出中，每个推荐项必须包含：

- 非空 `doc_id`；
- 长度为20至600个字符的 `evidence`；
- 非空 `reason`。

最多允许返回10个推荐项。额外字段会被 Pydantic 拒绝。

提示词要求 Evidence：

- 必须逐字复制自对应论文的标题或摘要；
- 必须是单个连续文本片段；
- 不得改写或拼接不连续片段；
- 不得自行添加引号或省略号；
- 应包含足够信息支持推荐理由。

提示词要求 Reason 为一句简洁说明，不应引入 Evidence 中没有明确出现的论文方法、结果、能力或细节。如果较短证据不足以支持理由，模型应选择更长的连续证据；如果没有合适证据，应省略该论文。程序不自动检查句子数量或语义支持性。

## 程序验证

生成完成后，程序执行以下检查：

- 返回的论文 ID 必须属于输入候选；
- 同一论文不得重复推荐；
- Reason 不得为空；
- Evidence 不得为空；
- Evidence 必须是该论文标题或摘要中的精确连续子字符串。

如果 Evidence 不满足精确原文检查，生成器会抛出 `RuntimeError`，不会把该结果作为有效推荐返回。

程序先去掉 Evidence 和 Reason 的首尾空白，对结构化返回的整个推荐列表进行检查，然后截取前 `limit` 项。任何一项校验失败都会使本次调用报错，即使该项位于请求数量之外；不会跳过错误项或自动重试。无结构化结果时同样报 `RuntimeError`；输入类型和取值错误、客户端及结构化解析异常会直接传播。

## 能够保证的性质

对于成功返回的结果，程序能够保证：

- 论文来自提供的检索候选；
- 同一论文不会重复出现；
- Evidence 不是空白文本；
- Evidence 确实逐字存在于论文标题或摘要；
- Reason 不是空白文本。

## 不能自动保证的性质

精确字符串验证不能证明：

- Evidence 一定与用户需求相关；
- Evidence 一定完整表达论文结论；
- Reason 在语义上完全被 Evidence 支持；
- 推荐论文一定是候选中最有用的论文；
- 摘要中的原始陈述本身一定正确。

这些性质仍需通过人工标注或明确记录的模型裁判进行评价。

该实现提供的是可核查的标题或摘要证据，不是完整引用系统。目前不补充作者、URL、发表日期、BibTeX 或论文全文证据。

## 公平对比设计

基础方法与创新方法必须使用完全相同的：

- 用户研究需求；
- 固定检索候选及顺序；
- 候选数量；
- 推荐数量上限；
- 生成模型和 API 配置。

唯一改变的是下游生成方法。

`experiments/generation/prepare_candidates.py` 负责检索一次并保存固定候选。`experiments/generation/run_comparison.py` 读取相同候选，分别运行 Baseline 与 Innovation，并保存结果、状态和耗时。

## 评价指标

生成器在返回结果前执行精确原文校验，实验脚本负责记录成功输出、捕获到的 `RuntimeError`、返回数量和耗时。尚需执行正式实验并基于运行记录计算汇总指标，先明确以下指标的分母和错误处理口径：

- Evidence 精确原文验证通过率；
- 生成成功率和错误类型分布；
- 每条查询的返回数量；
- 检索与生成耗时。

当前校验失败时整次生成报错，实验输出不保留失败响应中的逐项证据；不能用成功输出中证据全部通过，直接推导所有模型生成证据的逐项通过率。需区分查询级成功率与证据项级通过率，后者还需要额外记录失败响应。

人工评价至少包括：

- Context relevance：推荐论文是否与研究需求相关；
- Faithfulness：Reason 中的主张是否被标题或摘要支持；
- Answer relevance：Reason 是否解释论文与用户需求的联系；
- Evidence support：Evidence 是否在语义上支持对应 Reason。

人工评价应隐藏系统名称，并由至少两名标注者独立完成重叠样本。正式结果应报告原始一致率；条件允许时同时报告 Cohen’s kappa。分歧应在保留原始标签后进行讨论或仲裁。

## 当前状态

已经完成：

- 基础生成端到端运行验证；
- Evidence-Grounded Generator 实现；
- 精确 Evidence 验证；
- 一条查询上的功能性 API 测试；
- 固定候选导出框架；
- Baseline 与 Innovation 配对运行框架。

尚未完成：

- 最终检索方法和 query instruction 确认；
- 最终生成模型确认；
- 全量正式实验；
- 人工生成质量标注；
- 标注者一致率计算；
- 正式延迟、吞吐和费用统计；
- 统计结果与显著性分析。

在正式评估完成前，不能据此宣称 Innovation 已优于 Baseline。
