# 检索数据简介

`retrieval` 分支使用从 `data-preprocess` 分支迁移至 [`data/raw/`](../data/raw/) 的论文数据和预计算向量，作为检索模块的输入。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `papers.jsonl` | 共 6,000 条论文记录，每行一个 JSON 对象，包含论文 ID、标题、摘要、作者、分类、链接及检索文本 `text`（标题 + 完整摘要）等字段。 |
| `vectors.npz` | NumPy 压缩归档，包含 `vectors`、`doc_ids`、`chunk_ids` 和 `metadata`。向量矩阵形状为 `(6000, 1024)`，类型为 `float32`；可通过 ID 与论文记录关联。 |

论文覆盖 `rag`、`agents`、`reasoning`、`multimodal`、`image_video_generation` 和 `embodied_robotics` 六个主题，每个主题 1,000 条。

## 向量使用

根据归档中的 `metadata`，向量由 `Qwen/Qwen3-Embedding-0.6B` 基于论文标题和完整摘要生成，已归一化，使用余弦相似度进行检索。生成查询向量时，应使用相同模型及以下查询模板：

```text
Instruct: Given a research question, retrieve scientific papers whose core research addresses the question.
Query: {query}
```

模型 revision、输入长度限制等详细配置保存在 `metadata` 中。
