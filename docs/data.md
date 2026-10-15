# 数据与索引构建

本文说明检索数据、存储格式和构建更新流程。使用入口见 [README](../README.md)，检索接口见[检索 API](retrieval.md)，本地模型配置见[向量化](vectorization.md)。

## 数据文件

现有语料来自 `data-preprocess` 分支迁移的数据快照，共 6,000 篇论文，覆盖 `rag`、`agents`、`reasoning`、`multimodal`、`image_video_generation` 和 `embodied_robotics` 六个主题，每个主题 1,000 篇。数量描述的是当前快照，不是构建器的固定限制。

| 文件 | 内容 |
| --- | --- |
| [papers.jsonl](../data/raw/papers.jsonl) | 每行一个论文对象，含标识、标题、摘要、作者、分类、链接和 `text`（标题 + 完整摘要）等字段 |
| [vectors.npz](../data/raw/vectors.npz) | `vectors`、`doc_ids`、`chunk_ids`、`metadata`；当前向量为 `(6000, 1024)`、`float32` |
| [documents.db](../data/documents.db) | 论文表 `documents` 和全文索引 `documents_fts` |
| [documents.faiss](../data/documents.faiss) | 以数据库整数 ID 标识向量的 FAISS HNSW 内积索引 |

默认路径由 [config.py](../src/arxiv_paper_recommend/config.py) 定义。数据文件需在本地准备，不会由安装依赖自动生成。

## 论文与数据库格式

数据库构建只读取 `doc_id`、`title`、`abstract`；其他原始字段不写入数据库。JSONL 至少需要非空字符串 `doc_id` 和 `title`，摘要可为字符串、`null` 或省略：

```json
{"doc_id": "arxiv:2102.04643v1", "title": "Paper title", "abstract": "Complete abstract."}
```

| documents 字段 | 类型与约束 | 含义 |
| --- | --- | --- |
| `id` | `INTEGER PRIMARY KEY` | 内部 ID，对应 FTS5 rowid 和 FAISS 向量 ID |
| `doc_id` | `TEXT UNIQUE NOT NULL` | 仅移除 `arxiv:` 前缀，保留版本号 |
| `title` | `TEXT NOT NULL` | 标题 |
| `abstract` | `TEXT` | 可为空的摘要 |

`documents_fts` 使用外部内容表 `documents`，通过 `porter` 分词器索引标题和摘要，`doc_id` 不参与全文索引。

## 构建数据库

在项目根目录运行：

```powershell
uv run python -m arxiv_paper_recommend.retrieval.sparse.build_database
```

| 参数 | 默认值 |
| --- | --- |
| `--source` | `data/raw/papers.jsonl` |
| `--database` | `data/documents.db` |

重复运行会更新同一 `doc_id` 的标题与摘要，保留已有整数 ID，不删除源文件之外的旧记录。导入与全文索引重建在同一事务中完成，失败时回滚；重复论文 ID 或无效字段会报告源文件行号。

Python 入口为 `retrieval.sparse.build_database.DatabaseBuilder(source_path, database_path).build()`，返回导入记录数。

## 构建向量索引

先完成数据库构建，再运行：

```powershell
uv run python -m arxiv_paper_recommend.retrieval.dense.build_index
```

| 参数 | 默认值 |
| --- | --- |
| `--source` | `data/raw/vectors.npz` |
| `--database` | `data/documents.db` |
| `--index` | `data/documents.faiss` |

例如，使用自定义文件：

```powershell
uv run python -m arxiv_paper_recommend.retrieval.dense.build_index --source data/raw/vectors.npz --database data/documents.db --index data/custom.faiss
```

Python 入口为 `retrieval.dense.build_index.IndexBuilder(source_path, database_path, index_path).build()`，返回写入的向量数。导入模块不会读取数据或构建索引。

构建器校验非空二维浮点向量、有限值、非零范数，以及一维字符串 ID 数组与向量行数的对应关系；ID 移除 `arxiv:` 前缀后必须唯一，且均存在于数据库。数据库只读打开，映射查询按 SQLite 参数上限分批；无效输入在替换索引前报错。

向量归一化并转换为连续的 `float32`，写入 `IndexIDMap2(IndexHNSWFlat)`，使用内积相似度。参数保持为 `M=128`、`efConstruction=800`、`efSearch=256`。构建成功后通过同目录临时文件替换输出；校验、构建或写入失败时保留已有目标索引。

## 更新约定

- 数据库、FTS5 和 FAISS 必须使用同一套整数 ID 映射；重新创建数据库可能改变映射，需同步重建向量索引。
- 更新数据库不会自动生成论文向量；更新论文文本后，调用方负责重新编码并保存相应向量。
- `Qwen3Vectorizer` 只返回向量。调用方负责保存向量与论文 ID 的对应关系，再显式重建索引。
- 在线检索器复用加载时的索引；更新索引文件后，应创建新的检索器实例。

## 归档模型元数据

当前归档记录的模型为 `Qwen/Qwen3-Embedding-0.6B`，revision 为 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；文档输入为标题与完整摘要，不添加文档指令。采用末尾 token 池化、L2 归一化和余弦相似度，最大输入长度为 32768。

归档记录的查询模板为：

```text
Instruct: Given a research question, retrieve scientific papers whose core research addresses the question.
Query: {query}
```

当前向量化器保留了不同的默认查询指令（`research topic ... covers the aspects.`），不能将其描述为与归档指令一致。默认值、显式使用归档指令的方法和完整编码配置见[查询指令与归档](vectorization.md#查询指令与归档)。
