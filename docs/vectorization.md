# 本地文本向量化

`Qwen3Vectorizer` 将查询或文档转换为 NumPy 向量，只负责编码，不保存向量归档或构建索引。安装与检索入口见 [README](../README.md)，向量保存及索引更新约定见[数据与构建](data.md)，完成情况见[项目目标与进度](project.md)。它是检索编码器，不生成推荐理由；现有 `vectors.npz` 为预计算输入，当前仓库没有将整份语料批量编码并保存归档的命令。

## 安装与编码

```powershell
uv sync --extra vectorization
```

```python
from arxiv_paper_recommend.vectorization import Qwen3Vectorizer

vectorizer = Qwen3Vectorizer()

# 查询自动添加检索指令；单个字符串返回 (1024,)。
query_vector = vectorizer.encode_queries(
    "How can retrieval improve language model reasoning?"
)

# 文档不添加指令；列表返回 (N, 1024)。
document_vectors = vectorizer.encode_documents([
    "Paper title A\n\nComplete abstract of paper A.",
    "Paper title B\n\nComplete abstract of paper B.",
])
```

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `encode_queries(texts)` | `str` 或 `list[str]` | 添加查询指令后的文本向量 |
| `encode_documents(texts)` | `str` 或 `list[str]` | 原始文档文本向量 |

输出为 L2 归一化、连续的 `float32` 数组。字符串返回一维向量；列表始终返回二维矩阵，即使只有一项。空列表返回 `(0, 1024)`，空白字符串及非字符串项会报错。

## 配置

```text
Qwen3Vectorizer(
    model="Qwen/Qwen3-Embedding-0.6B",
    *,
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    device=None,
    batch_size=8,
    max_length=32768,
    query_instruction=DEFAULT_QUERY_INSTRUCTION,
)
```

以上为签名说明，完整调用示例见前文。

| 参数 | 默认值与规则 |
| --- | --- |
| `model` | Hugging Face 模型 ID 或本地目录；模型必须输出 1024 维 |
| `revision` | 上述固定版本；使用本地目录时可传 `None` |
| `device` | 自动选择可用 CUDA，否则 CPU；可指定 `"cpu"` 或 `"cuda:0"` |
| `batch_size` | 8，必须为正整数 |
| `max_length` | 32768，接受 1～32768 的整数，包括查询指令长度 |
| `query_instruction` | 非空白字符串；当前默认指令见下一节 |

实例化时加载模型，首次访问远程模型会下载文件，此后复用本地缓存；仅导入向量化包不会加载模型或导入 PyTorch、Transformers。模型以 FP32 推理，CUDA 运行需要与本机兼容的 PyTorch 构建。

分词器固定使用左侧 padding 和右侧截断，池化直接取末尾有效 token，再执行 L2 归一化。单次调用内，相同文本只编码一次；不同文本按字符长度稳定分批，随后恢复原始顺序及重复项。批次变化可能产生微小浮点差异。应通过构造参数配置实例，不直接更改内部 tokenizer 的 padding 方向。

## 查询指令与归档

当前默认查询指令保持为：

```text
Given a research topic, retrieve scientific papers whose core research covers the aspects.
```

查询编码模板为：

```text
Instruct: {instruction}
Query: {query}
```

现有 `vectors.npz` 的模型和 revision 与默认配置相同，但其 metadata 记录的查询指令是：

```text
Given a research question, retrieve scientific papers whose core research addresses the question.
```

两者不相同。当前实现使用前一个默认值；如需显式使用归档记录的查询指令，可配置：

```python
vectorizer = Qwen3Vectorizer(
    query_instruction=(
        "Given a research question, retrieve scientific papers whose core "
        "research addresses the question."
    ),
)
```

改变指令会改变查询向量及可能的排名；改变模型、revision 或文档编码配置时，应检查与现有论文向量的兼容性。正式实验需保存实际使用的查询指令、模型 revision、长度与设备配置，不能仅凭归档 metadata 推断在线查询配置。

归档记录的生成环境与当前 `pyproject.toml` 安装环境并不相同；归档模型和 revision 一致不代表环境已完全复现。语料来源哈希差异见[归档模型元数据](data.md#归档模型元数据)，实验记录口径见[实验与评估](evaluation.md)。

## 注入检索器

```python
from arxiv_paper_recommend.retrieval.dense import HNSWRetriever

with HNSWRetriever(vectorizer=vectorizer) as retriever:
    results = retriever.search(
        "How can retrieved evidence support multi-step reasoning?", limit=10,
    )
```

向量化器可跨调用复用模型。检索器不会接管注入的向量化器，查询向量维度必须与索引一致。完整搜索和评分接口见[检索 API](retrieval.md)。
