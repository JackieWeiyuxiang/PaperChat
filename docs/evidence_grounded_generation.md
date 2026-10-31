# 基于可验证证据的论文推荐生成

本文说明证据增强版的判断约束及验证边界。两阶段编排、输入校验、引用格式和错误行为与[基础生成器](recommendation.md)一致；整体评估见[实验与评估](evaluation.md)，可复现实验脚本见[生成对比实验](../experiments/generation/README.md)。

## 方法与基础版的区别

两种生成器都实现 `BaseRecommendationGenerator`，共享以下流程：

```text
研究需求 + 检索候选
→ 取前 30 条，按 arXiv ID 去重
→ 每篇独立判断（最多 5 个并发请求，thinking=none，temperature=0）
→ recommended: bool + reason: str（内嵌 1～2 处原文引用）
→ 检查全部理由中的引用精确匹配
→ 只将初筛推荐项交给 API 筛选排序（thinking=high）
→ 返回 min(limit, 初筛推荐数) 篇及原有理由
```

初筛推荐数为 0 时跳过第二阶段，直接返回空列表。初筛不推荐项不会交给第二阶段；第二阶段仅返回有序论文 ID，不重写理由或引用。

基础版允许依据完整标题和摘要解释相关性，并标注有助核查的原文。证据增强版进一步要求：理由中的每项论文具体方法、结果、能力或细节，都必须被标注的原文片段直接支持，不能依赖未引用的标题或摘要内容补充主张。短片段不足以支持理由时应选更长的片段，或增加第二处引用。

不能支持推荐时返回 `recommended=False`，引用论文实际范围并说明匹配不足；不因为摘要未提及某方法，就声称论文没有该方法。上述语义要求由提示词约束，程序不自动证明其成立。

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
| `reason` | 理由及 1～2 处 `[[cite]]原文片段[[/cite]]` 引用 |

`EvidenceGroundedRecommendation` 现在是 `PaperRecommendation` 的别名，保留原有导入路径。独立 `evidence` 字段已移除；旧调用方应从 `reason` 中读取带标记的引用，见[接口迁移](migration.md)。

模型配置、客户端注入及资源所有权沿用基础生成器：显式 `model` 覆盖配置，自建客户端由生成器关闭，注入客户端由调用方关闭。当前 `innovation` 包直接导出生成器，导入时会加载共享 LLM 模块。

## 引用格式与验证

第一阶段结构化输出严格只有 `recommended` 和 `reason` 两个字段，额外字段被拒绝；布尔值不接受字符串或数字替代。推荐和不推荐的理由均需含 1～2 处引用：

```text
该文与研究需求的关系……，依据为：[[cite]]连续原文一[[/cite]]；[[cite]]连续原文二[[/cite]]。
```

引用标记必须正确配对、不嵌套。每处引用须非空白，并逐字符匹配对应论文标题或摘要中的一个连续片段；不得改写或在同一对标记内拼接不连续片段。缺失摘要时可引用标题。不再使用旧版单个 `evidence` 字段的 20～600 字符限制。

所有逐篇响应都会校验，包括最终未被选中和判为不推荐的论文。引用校验失败时仅对该篇附带错误反馈重新判断，最多尝试三次；仍失败则整次生成抛出 `RuntimeError`，不跳过该项或返回部分结果。第二阶段还校验数量、候选归属和去重；共享 LLM 层对结构校验失败最多调用三次纠正，耗尽后传播异常，客户端异常沿用原有行为。完整错误契约见[推荐生成](recommendation.md#thinking-与错误处理)。

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

成功返回时，两种生成器都保证候选来源正确、没有重复、数量符合初筛推荐数与 `limit`、理由非空，以及 1～2 处标注引用精确存在于对应标题或摘要。精确匹配不能证明引用与需求相关、理由完全被引用语义支持、摘要陈述本身正确，或所选论文一定最有用。本实现不提供论文全文、作者、URL、日期或 BibTeX。

比较基础版与证据增强版时，固定研究需求、检索候选及顺序、候选数量、推荐上限、模型、thinking 配置和并发上限。共同编排与输出格式相同，变化仅为逐篇判断提示词；两种提示词可能产生不同的初筛推荐数，因此需同时报告最终返回数量。

`prepare_candidates.py` 保存固定候选，`run_comparison.py` 配对运行并记录输出、状态和耗时，两种结果均通过 `reason` 保存引用标记。脚本只捕获 `RuntimeError`，客户端及解析异常仍可能中断。失败响应不保存逐项内容，因此不能从成功输出中引用全部通过，推导所有模型响应的逐项证据通过率。

正式评估仍需记录生成成功率、错误类型、返回数量、延迟与费用，并人工评价相关性、faithfulness、answer relevance 及引用对理由的语义支持。标注应隐藏系统名称，并由至少两名标注者独立完成重叠样本，报告原始一致率；条件允许时报告 Cohen’s kappa。

本轮两阶段实现尚未执行真实 API 验证或正式质量评估；此前版本的功能性运行不能作为新流程的效果证据。在正式评估完成前，不能宣称证据增强版优于基础版。
