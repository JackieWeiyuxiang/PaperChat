# 检索数据简介

`retrieval` 分支使用从 `data-preprocess` 分支迁移至 [`data/raw/`](../data/raw/) 的论文数据和预计算向量，作为检索模块的输入。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| [`documents.db`](../data/documents.db) | SQLite 数据库，`documents` 保存 6,000 条论文记录，`documents_fts` 提供 FTS5 全文检索。`id` 为整数主键；`doc_id` 为带版本号的论文标识（如 `2102.04643v1`），取自 `papers.jsonl` 的 `doc_id` 并去掉 `arxiv:` 前缀。 |
| `papers.jsonl` | 共 6,000 条论文记录，每行一个 JSON 对象，包含论文 ID、标题、摘要、作者、分类、链接及检索文本 `text`（标题 + 完整摘要）等字段。 |
| `vectors.npz` | NumPy 压缩归档，包含 `vectors`、`doc_ids`、`chunk_ids` 和 `metadata`。向量矩阵形状为 `(6000, 1024)`，类型为 `float32`；可通过 ID 与论文记录关联。 |

论文覆盖 `rag`、`agents`、`reasoning`、`multimodal`、`image_video_generation` 和 `embodied_robotics` 六个主题，每个主题 1,000 条。

## 数据库构建

在项目根目录运行：

```powershell
uv run python -m arxiv_paper_recommend.retrieval.sparse.build_sql
```

默认将 `data/raw/papers.jsonl` 导入 `data/documents.db`，可通过 `--source` 和 `--database` 指定其他路径。重复运行会更新同一 `doc_id` 的标题和摘要，保留整数 ID，不删除源文件之外的已有记录，并重建全文索引；导入失败时回滚本次修改。

## 向量使用

根据归档中的 `metadata`，向量由 `Qwen/Qwen3-Embedding-0.6B` 基于论文标题和完整摘要生成，已归一化，使用余弦相似度进行检索。生成查询向量时，应使用相同模型及以下查询模板：

```text
Instruct: Given a research question, retrieve scientific papers whose core research addresses the question.
Query: {query}
```

模型 revision、输入长度限制等详细配置保存在 `metadata` 中。
