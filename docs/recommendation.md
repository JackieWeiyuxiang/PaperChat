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

`BaseRecommendationGenerator` 是共同抽象基类，负责候选处理、并发调用、逐篇容错、理由非空校验与集中排序。公共提示词仅规定判断与理由要求。基础版依据完整标题和摘要解释相关性，提示词不提及引用；证据增强版的专属提示词要求 1～2 处原文引用，且理由中的论文具体主张仅由标注片段直接支持。引用格式与原文要求仅由证据增强版提示词约束，程序不校验引用标记或原文匹配。

生成器复用[LLM 配置](retrieval.md#querygenerator)，显式模型名覆盖配置。也可注入客户端，由调用方统一关闭；客户端须支持并发请求。只导入 `PaperRecommendation` 不会加载 LLM 客户端模块；推荐包延迟导出共同基类与基础生成器。

## 候选与两阶段流程

`request` 必须是非空白字符串，`papers` 必须是 `SearchResult` 对象列表，`limit` 为 1～10 的整数，不接受布尔值。先取前 30 条候选，再按 `arxiv_id` 去重，保留首次出现的原始对象；不会从第 31 条之后补齐。候选为空时直接返回空列表，不调用 API。缺失摘要以空字符串提供。

第一阶段为每篇候选独立调用一次 API，输入研究需求、论文 ID、标题和摘要。使用线程池，最多 5 个请求并发；响应始终绑定提交时的论文，完成顺序不影响关联。Pydantic 结构化输出严格只有两个字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `recommended` | `bool` | 是否推荐，不接受字符串或数字代替布尔值 |
| `reason` | `str` | 非空白判断理由，解释论文与研究需求的关系 |

即使判断为不推荐，理由也必须解释论文的实际范围与需求的关系，说明不匹配或支持不足；不能因为摘要未提及某方法就声称论文没有该方法。提示词不允许外部知识或虚构论文方法、结果和能力。证据增强版另要求推荐和不推荐的理由均包含原文引用。

第一阶段全部完成后，仅保留成功评估且 `recommended=True` 的论文及其理由。推荐数为 0 时直接返回空列表；为 1 时直接返回该篇，设 `rank=1`，沿用原论文对象与初筛理由，不调用排序 API。推荐数至少为 2 时调用一次 API，从这些论文中选择并排序恰好 `min(limit, 初筛推荐数)` 篇；即使 `limit=1` 仍由 LLM 选择，不把初筛不推荐项或评估失败项交给第二阶段。

第二阶段结构化输出只含 `doc_ids` 有序列表，提示词不提及引用。最终理由直接沿用第一阶段结果，排序不重新生成或修改理由。对于 N 个去重候选：不需要重试时，初筛推荐数为 0 或 1 时调用 N 次 API，至少为 2 时调用 N+1 次。每篇判断与集中排序各自最多重试 3 次、含首次最多请求 4 次，所有失败类型共用各自的预算；SDK 内置重试已关闭，没有嵌套重试。无需排序时最多请求 4N 次，需要排序时最多请求 4(N+1) 次。30 个不同候选正常情况下最多调用 31 次，计入全部重试后最多调用 124 次；这些次数不含上游查询生成。

## 统一结果

| 结果字段 | 类型 | 含义 |
| --- | --- | --- |
| `rank` | `int` | 从 1 开始的推荐名次 |
| `paper` | `SearchResult` | 原始候选对象，保留检索分数 |
| `reason` | `str` | 去掉首尾空白后的初筛理由 |

`PaperRecommendation` 是冻结数据类，`paper` 保留原始引用；生成器不修改输入候选列表，也不提供独立 `evidence` 字段。基础版提示词不提及引用；模型自发生成的标记仍会保留。证据增强版将原文引用写入 `reason`，格式与约束见[证据生成](evidence_grounded_generation.md#引用格式与验证)。

程序只去除整个理由的首尾空白，不修复或删除引用标记，不检查引用数量、配对、嵌套、引用内容非空或原文匹配；这些问题也不会触发重试。

## Thinking 与错误处理

所有推荐阶段请求均使用 `OpenAI.responses.parse` 和 Pydantic `text_format`。第一阶段逐篇判断（包括理由与结构校验纠正）使用 `reasoning={"effort": "none"}` 与 `temperature=0`；第二阶段集中筛选排序使用 `reasoning={"effort": "high"}`，不发送 `temperature`。服务与模型需支持上述配置与结构化输出组合。统一查询生成也使用 `high`，原独立关键词／查询生成方法仍使用 `none` 与 `temperature=0`。

程序检查全部逐篇响应，包括不推荐的论文：

- 理由不得为空白。
- `recommended` 与 `reason` 的字段、类型及额外字段限制必须符合结构化模型。

集中排序还会检查返回数量准确、ID 全部来自初筛推荐集合且没有重复。共享 LLM 层以每次逻辑调用为边界统一重试：网络/API 异常、Pydantic 或 JSON 解析异常、SDK 响应校验异常、缺少结构化结果、理由非空与排序校验失败在每篇判断或集中排序中共用最多 3 次重试，含首次最多请求 4 次。每篇判断、集中排序及上游查询生成的预算相互独立；统一查询生成最多重试 5 次，旧独立生成方法最多重试 3 次。输入参数和资源状态检查不参与重试。SDK 网络/API 异常均参与重试，不按 HTTP 状态提前终止；SDK 响应校验异常归入解析／校验类。

结构与业务校验失败时，以原始提示词加上最近一次错误反馈立即重新生成，不累积历史反馈，也不放宽字段或类型约束。逐篇校验失败仅重新判断对应论文；排序失败仅重新生成有序 ID，不重新执行逐篇判断。网络/API 异常保留已有校验反馈，第 1、2、3 次失败若属于此类，分别等待 1、2、4 秒再重试。第 4 次失败不再等待，由共享层原样抛出该次异常；缺少结构化结果及业务校验失败的异常仍为 `RuntimeError`。自建客户端设置 `max_retries=0`，每次逻辑调用还通过 `with_options(max_retries=0)` 覆盖注入客户端的 SDK 重试配置，不修改原客户端配置或接管其资源。

第一阶段中，网络/API、结构解析或业务校验异常耗尽该篇预算后，收集结果时会 `print` 论文 arXiv ID、异常类型与错误内容，每篇最终失败只打印一次，按不推荐处理并继续其他候选，不增加重试。线程池仍等待全部提交任务结束，再按成功通过数决定返回或排序。输入错误、资源初始化错误及其他未处理异常继续传播；集中排序耗尽预算后也继续传播，不以初筛顺序代替排序结果。

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

两种方法的理由均需评估语义支持性；证据增强版另需评估引用格式与原文对应关系。这些性质需人工或明确记录的模型评价，程序不保证成立。两种方法均不补充作者、URL 或日期，完整引用模块 `citations` 仍为占位。生成质量、忠实性和相关性尚未正式评估，见[实验与评估](evaluation.md)。
