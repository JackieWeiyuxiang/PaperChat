# 基于可验证证据的论文推荐生成

本文说明证据增强版的判断约束及验证边界。两阶段编排、输入校验、结果模型和错误行为与[基础生成器](recommendation.md)一致；引用格式与证据支持要求由证据增强版专属提示词规定。整体评估见[实验与评估](evaluation.md)，可复现实验脚本见[生成对比实验](../experiments/generation/README.md)。

## 方法与基础版的区别

两种生成器都实现 `BaseRecommendationGenerator`，共享以下流程：

```text
研究需求 + 检索候选
→ 取前 30 条，按 arXiv ID 去重
→ 每篇独立判断（最多 5 个并发请求，thinking=none，temperature=0）
→ recommended: bool + reason: str（仅证据增强版提示词要求内嵌 1～2 处原文引用）
→ 检查结构与理由非空，逐篇失败耗尽重试后打印错误并按不推荐处理
→ 至少两篇通过时，将推荐项交给 API 筛选排序（thinking=high）
→ 返回 min(limit, 初筛推荐数) 篇及原有理由
```

初筛推荐数为 0 时直接返回空列表；为 1 时直接返回该篇并设为第 1 名，跳过排序。
至少两篇通过时才调用第二阶段，即使 `limit=1` 仍由 LLM 选择。
初筛不推荐项和评估失败项不会交给第二阶段；第二阶段仅返回有序论文 ID，不重写理由或引用。

基础版依据完整标题和摘要解释相关性，提示词不提及引用。证据增强版要求理由内嵌 1～2 处带标记的原文引用，且理由中的每项论文具体方法、结果、能力或细节，都必须被标注的原文片段直接支持，不能依赖未引用的标题或摘要内容补充主张。短片段不足以支持理由时应选更长的片段，或增加第二处引用。

证据增强版不能支持推荐时返回 `recommended=False`，引用论文实际范围并说明匹配不足；不因为摘要未提及某方法，就声称论文没有该方法。上述语义要求由提示词约束，程序不自动证明其成立。

## 公开接口与统一结果

```python
from arxiv_paper_recommend.recommendation import (
    BaseRecommendationGenerator,
    PaperRecommendation,
)
from arxiv_paper_recommend.recommendation.innovation import (
    EvidenceGroundedRecommendation,
    EvidenceGroundedRecommendationGenerator,
)
```

```text
EvidenceGroundedRecommendationGenerator(*, model=None, client=None)

generate(request: str, papers: list[SearchResult], limit=10) -> list[PaperRecommendation]
```

两种具体生成器统一返回冻结数据类 `PaperRecommendation`：

| 字段 | 含义 |
| --- | --- |
| `rank` | 从 1 开始的推荐名次 |
| `paper` | 原始 `SearchResult` 引用，保留检索分数 |
| `reason` | 非空白理由；仅证据增强版提示词要求包含 1～2 处 `[[cite]]原文片段[[/cite]]` 引用 |

`EvidenceGroundedRecommendation` 现在是 `PaperRecommendation` 的别名，保留原有导入路径。独立 `evidence` 字段已移除；旧调用方应从 `reason` 中读取带标记的引用，见[接口迁移](migration.md)。

模型配置、客户端注入及资源所有权沿用基础生成器：显式 `model` 覆盖配置，自建客户端由生成器关闭，注入客户端由调用方关闭。当前 `innovation` 包直接导出生成器，导入时会加载共享 LLM 模块。

## 引用格式与验证

第一阶段结构化输出严格只有 `recommended` 和 `reason` 两个字段，额外字段被拒绝；布尔值不接受字符串或数字替代。证据增强版提示词要求推荐和不推荐的理由均含 1～2 处引用：

```text
该文与研究需求的关系……，依据为：[[cite]]连续原文一[[/cite]]；[[cite]]连续原文二[[/cite]]。
```

证据增强版提示词要求引用标记正确配对、不嵌套，每处引用非空白且来自对应标题或摘要的连续原文，
逐字保留原文字符、空白和标点，不得改写、擅加省略号或在同一对标记内拼接不连续片段；
示例占位文本应替换为候选实际原文，缺失摘要时引用标题。
程序保留生成格式与返回的标记，但不校验标记数量、配对、嵌套、引用内容非空或原文匹配，
也不修复这些问题。不再使用旧版单个 `evidence` 字段的 20～600 字符限制。

所有逐篇响应仍检查字段、严格类型、额外字段和理由非空，包括最终未被选中和判为不推荐的论文。
每篇判断与集中排序各自最多重试 3 次，含首次最多请求 4 次；网络/API 异常、结构解析、
缺少结构化结果及理由或排序校验失败共用各自的预算，SDK 内置重试已关闭。
校验失败时附带最近错误立即重新生成，网络/API 异常保留反馈并按尝试序号等待 1、2、4 秒。
单篇重试耗尽后 `print` 论文 arXiv ID、异常类型与错误内容，按不推荐处理，继续其他候选；
集中排序最终失败仍原样传播。输入和资源初始化错误继续传播。
完整错误契约见[推荐生成](recommendation.md#thinking-与错误处理)。

## 使用示例

以下使用 BM25 检索；生成阶段仍需有效 LLM 配置。

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
        print(item.reason)  # Includes [[cite]]...[[/cite]] source spans.
```

## 能力边界与公平比较

成功返回时，两种生成器都保证候选来源正确、没有重复、数量符合成功初筛推荐数与 `limit`、理由非空。
引用标记格式、引用数量、原文匹配和具体主张的证据支持性均为证据增强版提示词要求，程序不保证这些性质成立；
也不能证明摘要陈述本身正确或所选论文一定最有用。本实现不提供论文全文、作者、URL、日期或 BibTeX。

比较基础版与证据增强版时，固定研究需求、检索候选及顺序、候选数量、推荐上限、模型、thinking 配置和并发上限。共同编排与结果结构相同，变化仅为逐篇判断提示词，包括证据增强版专属的引用要求；两种提示词可能产生不同的初筛推荐数，因此需同时报告最终返回数量。

`prepare_candidates.py` 保存固定候选，`run_comparison.py` 配对运行并记录输出、状态和耗时，两种结果均通过 `reason` 保存模型生成的理由，并保留其中可能出现的引用标记；仅证据增强版提示词要求这些标记。单篇评估失败由生成器打印并跳过，不会单独写入成功推荐结果；脚本只捕获 `RuntimeError`，集中排序等环节的客户端及解析异常仍可能中断。引用不再自动校验，不能从成功输出推导模型响应的引用格式、原文匹配或证据通过率。

正式评估仍需记录生成成功率、错误类型、返回数量、延迟与费用，并人工评价相关性、faithfulness、answer relevance；证据增强版另评价引用对理由的语义支持。标注应隐藏系统名称，并由至少两名标注者独立完成重叠样本，报告原始一致率；条件允许时报告 Cohen’s kappa。

本轮两阶段实现尚未执行真实 API 验证或正式质量评估；此前版本的功能性运行不能作为新流程的效果证据。在正式评估完成前，不能宣称证据增强版优于基础版。
