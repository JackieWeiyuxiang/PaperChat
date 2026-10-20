# 推荐生成

生成器消费已检索论文，输出基于标题和摘要的推荐理由，不负责检索或补充引用元数据。安装和端到端示例见 [README](../README.md)，候选接口见[检索 API](retrieval.md)，升级说明见[接口迁移](migration.md)。

## 公开接口

```python
from arxiv_paper_recommend.recommendation import (
    PaperRecommendation,
    RecommendationGenerator,
)
```

```text
RecommendationGenerator(*, model=None, client=None)
generate(request: str, papers: list[SearchResult], limit=10) -> list[PaperRecommendation]
```

生成器复用[LLM 配置和结构化调用](retrieval.md#querygenerator)，显式模型名覆盖配置。也可向查询生成器和推荐生成器注入同一客户端，由调用方统一关闭。

只导入 `PaperRecommendation` 不会加载 LLM 客户端模块。推荐模型定义与生成实现分离，推荐包延迟导出生成器。

## 候选与输出

`request` 必须是非空白字符串，`papers` 必须是检索结果对象列表，`limit` 为 1～10 的整数，不接受布尔值。

先取前 30 个候选，再按 `arxiv_id` 去重，保留首次出现的对象。候选为空时直接返回空列表，不调用 LLM。非空候选调用一次 LLM；输入包含研究请求、目标数量、论文 ID、标题和摘要，缺失摘要以空字符串提供。

模型可在证据不足时少推荐或不推荐。每项理由应为一至两句，只依据该候选标题与摘要，说明与研究需求的联系，不使用外部知识。

| 结果字段 | 类型 | 含义 |
| --- | --- | --- |
| `rank` | `int` | 从 1 开始的推荐名次 |
| `paper` | `SearchResult` | 原始候选对象，保留其检索分数 |
| `reason` | `str` | 去掉首尾空白后的推荐理由 |

`PaperRecommendation` 是冻结数据类；其中的 `paper` 保留原始引用。推荐排序由生成结果决定，不修改原始候选列表。最终最多返回 `limit` 项。

## 响应校验与错误

Pydantic 结构化输出限制字段及列表规模；生成器随后检查整个返回列表，再截取请求数量：

- 论文 ID 必须属于提供给模型的候选。
- 不得重复推荐同一论文。
- 推荐理由不得为空白。

即使错误项位于请求数量之外，也会报错。没有结构化结果、未知论文、重复论文或空白理由均报 `RuntimeError`；客户端及结构化解析异常直接传播。

## 使用示例

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

## 资源与当前限制

生成器支持上下文管理或手动 `close()`，仅关闭自建客户端，关闭后不能继续使用。单独传入 `model` 不改变客户端所有权。

当前理由不是独立引用证据；结果不补充作者、URL 或日期。`citations` 与下游创新包仍为占位。生成质量、忠实性和相关性尚未正式评估，见[实验与评估](evaluation.md)。
