# 推荐生成

生成器消费已检索论文，先逐篇判断是否推荐，再集中筛选排序，不负责检索或补充引用元数据。基础版与[证据增强版](evidence_grounded_generation.md)共同实现 `BaseRecommendationGenerator`，返回相同的推荐模型。安装和端到端示例见 [README](../README.md)，候选接口见[检索 API](retrieval.md)，不兼容变更见[接口迁移](migration.md)。

## 公开接口

```python
from arxiv_paper_recommend.recommendation import (
    BaseRecommendationGenerator,
    PaperRecommendation,
    RecommendationGenerator,
)
```

```text
RecommendationGenerator(*, model=None, client=None)

generate(request: str, papers: list[SearchResult], limit=10) -> list[PaperRecommendation]
```

`BaseRecommendationGenerator` 是共同抽象基类，负责候选处理、并发调用、引用校验与集中排序。两种具体实现提供不同的逐篇判断提示词：基础版依据完整标题和摘要解释相关性；证据增强版进一步要求理由中的论文具体主张仅由标注的引用片段支持。

生成器复用[LLM 配置](retrieval.md#querygenerator)，显式模型名覆盖配置。也可注入客户端，由调用方统一关闭；客户端须支持并发请求。只导入 `PaperRecommendation` 不会加载 LLM 客户端模块；推荐包延迟导出共同基类与基础生成器。

## 候选与两阶段流程

`request` 必须是非空白字符串，`papers` 必须是 `SearchResult` 对象列表，`limit` 为 1～10 的整数，不接受布尔值。先取前 30 条候选，再按 `arxiv_id` 去重，保留首次出现的原始对象；不会从第 31 条之后补齐。候选为空时直接返回空列表，不调用 API。缺失摘要以空字符串提供。

第一阶段为每篇候选独立调用一次 API，输入研究需求、论文 ID、标题和摘要。使用线程池，最多 5 个请求并发；响应始终绑定提交时的论文，完成顺序不影响关联。Pydantic 结构化输出严格只有两个字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `recommended` | `bool` | 是否推荐，不接受字符串或数字代替布尔值 |
| `reason` | `str` | 判断理由，包含 1～2 处带标记的原文引用 |

即使判断为不推荐，理由也必须解释论文的实际范围与需求的关系，并包含引用。提示词不允许外部知识或虚构论文方法、结果和能力。

第一阶段全部完成后，仅将 `recommended=True` 的论文及其理由传给第二阶段。如果推荐数为 0，直接返回空列表；否则调用一次 API，从这些论文中选择并排序恰好 `min(limit, 初筛推荐数)` 篇。推荐数少于或等于 `limit` 时仍调用 API 对全部推荐项排序，不把初筛不推荐项交给第二阶段。

第二阶段结构化输出只含 `doc_ids` 有序列表。最终理由直接沿用第一阶段结果，避免排序时重新生成或修改引用。对于 N 个去重候选：无推荐项时调用 N 次 API；存在推荐项时调用 N+1 次。30 个不同候选最多调用 31 次，不含 SDK 既有重试。

## 统一结果与引用格式

| 结果字段 | 类型 | 含义 |
| --- | --- | --- |
| `rank` | `int` | 从 1 开始的推荐名次 |
| `paper` | `SearchResult` | 原始候选对象，保留检索分数 |
| `reason` | `str` | 去掉首尾空白后的理由，保留原文引用标记 |

`PaperRecommendation` 是冻结数据类，`paper` 保留原始引用；生成器不修改输入候选列表。证据直接写在 `reason` 中，不提供独立 `evidence` 字段。每处引用的固定格式为：

```text
[[cite]]标题或摘要中的连续原文[[/cite]]
```

理由示例：

```text
该文的方法适合这一研究需求，摘要指出：[[cite]]exact source text[[/cite]]。
```

示例中的占位文本必须替换为候选实际原文。引用可以来自标题或摘要，每处必须分别为连续片段；允许两处独立引用，不允许在一对标记内拼接不连续片段、改写、擅加省略号或嵌套标记。程序不修剪引用内部字符，逐字符检查原文匹配，方便后续前端按固定标记高亮展示。

## Thinking 与错误处理

所有推荐阶段请求均使用 `OpenAI.responses.parse` 和 Pydantic `text_format`，显式传入 `reasoning={"effort": "high"}`，不发送 `temperature`。服务与模型需支持 thinking 和结构化输出组合。查询生成继续使用 `reasoning={"effort": "none"}` 与 `temperature=0`。

程序检查全部逐篇响应，包括不推荐的论文：

- 理由不得为空白。
- 引用标记必须正确配对、不嵌套，数量为 1～2 处。
- 每处引用不得为空白，且必须是对应标题或摘要中的精确连续子字符串。

集中排序还会检查返回数量准确、ID 全部来自初筛推荐集合且没有重复。没有结构化结果或上述业务校验失败时抛出 `RuntimeError`，不跳过错误项，也不返回部分结果。输入、客户端及结构化解析异常直接传播；不增加业务重试，仅沿用 SDK 既有重试行为。任一第一阶段请求失败时不进入第二阶段，线程池会等待已提交任务结束。

## 使用示例与限制

以下先用 BM25 获取候选，无需安装向量化依赖；生成推荐仍需有效 LLM 配置。

```python
from arxiv_paper_recommend.recommendation import RecommendationGenerator
from arxiv_paper_recommend.retrieval.sparse import BM25Retriever

request = "How can retrieval improve language model reasoning?"
with BM25Retriever() as retriever, RecommendationGenerator() as generator:
    papers = retriever.search(
        ["retrieval augmented generation", "reasoning"], limit=30,
    )
    recommendations = generator.generate(request, papers, limit=10)
    for item in recommendations:
        print(item.rank, item.paper.arxiv_id, item.reason)
```

生成器支持上下文管理或手动 `close()`，仅关闭自建客户端，关闭后不能继续使用。单独传入 `model` 不改变客户端所有权。

精确原文校验不能证明理由在语义上被引用支持；该性质仍需人工或明确记录的模型评价。两种方法均不补充作者、URL 或日期，完整引用模块 `citations` 仍为占位。已提供[生成对比实验脚本](../experiments/generation/README.md)，生成质量、忠实性和相关性尚未正式评估，见[实验与评估](evaluation.md)。
