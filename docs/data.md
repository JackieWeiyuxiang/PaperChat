# 数据与索引构建

本文说明实际语料、存储格式和构建更新流程。使用入口见 [README](../README.md)，检索接口见[检索 API](retrieval.md)，本地模型配置见[向量化](vectorization.md)，课程要求与规模差距见[项目目标与进度](project.md)。统计核对日期：2026-10-16。

## 数据文件

现有语料来自 `data-preprocess` 分支迁移的数据快照，共 6,000 篇论文。全部记录的 `source` 字段为 `arxiv_api`；提案中的 Kaggle 数据源属于原计划，不能作为当前快照的实际来源说明。当前 `corpus/` 为空包，未包含采集、清洗、主题筛选和批量向量归档的完整流程；构建器只负责将已准备的数据导入数据库和索引。

| 文件 | 内容 |
| --- | --- |
| [papers.jsonl](../data/raw/papers.jsonl) | 每行一个论文对象，含标识、标题、摘要、作者、分类、链接和 `text`（标题 + 完整摘要）等字段 |
| [vectors.npz](../data/raw/vectors.npz) | `vectors`、`doc_ids`、`chunk_ids`、`metadata`；当前向量为 `(6000, 1024)`、`float32` |
| [documents.db](../data/documents.db) | 论文表 `documents` 和全文索引 `documents_fts` |
| [documents.faiss](../data/documents.faiss) | 以数据库整数 ID 标识向量的 FAISS HNSW 内积索引 |

默认路径由 [config.py](../src/arxiv_paper_recommend/config.py) 定义。上表四个文件当前均在 Git 跟踪中；安装依赖不会自动生成或更新它们。根目录另有历史文件 `data.db`，其表结构不符合当前 `documents` 读取接口，也不属于默认数据流；请使用 `data/documents.db`。

## 当前快照统计

以下数量由现有 JSONL 读取统计。标题、摘要、作者、论文 ID、版本、首次提交日期及分类均无缺失；版本化 `doc_id`、不含版本的 `paper_id`、`chunk_id` 和完整 `text` 各有 6,000 个不同值。这表明当前快照中没有相同标识或完全相同检索文本，不证明采集阶段采用了哪种清洗算法，也不排除近重复内容。

| 项目 | 当前值与口径 |
| --- | --- |
| 论文／段落数 | 6,000／6,000；一篇一个标题＋完整摘要段落 |
| 检索文本 | 所有记录的 `text = title + "\n\n" + abstract`；不拼入作者关键词或正文 |
| 总词数 | 1,145,234；对每篇 `text` 使用空白分隔后计数 |
| 不同词数（types） | 74,314；同样按空白分隔、保留大小写和标点，不做词干化 |
| 不同词数（转小写口径） | 65,466；仅在上述分词后转小写，供解释统计口径差异 |
| 向量／数据库／索引条目数 | NPZ 6,000 行、`documents` 6,000 行、FAISS 6,000 个向量 |

词数不是模型 token 数；不同词数也不是 FTS5 `porter` 分词后的词表规模。作业要求至少 10,000 篇文档／段落和 100,000 词：当前满足词数门槛，但文档／段落数量仍差至少 4,000，不能用向量维数或跨方法检索结果条目充当额外段落。

| `collection_topic` | 篇数 |
| --- | --- |
| `rag` | 1,000 |
| `agents` | 1,000 |
| `reasoning` | 1,000 |
| `multimodal` | 1,000 |
| `image_video_generation` | 1,000 |
| `embodied_robotics` | 1,000 |

每篇只计入一个收录主题；配额均衡不代表整个 arXiv 的分布。按 `published` 的首次提交年份统计，2021–2025 年分别为 346、553、1,311、1,782、2,008 篇。数量描述的是当前快照，不是构建器的固定限制。

本地另有 `data/statistics.md` 统计报告，被 Git 忽略且没有在当前源码中保留生成入口；本页记录可随仓库分发的核心统计与口径，不将该报告视为可复现的数据处理流程。

## 论文与数据库格式

原始记录另有 `paper_id`、`version`、`authors`、`categories`、`primary_category`、`published`、`updated`、`url`、`pdf_url`、`collection_topic`、`chunk_id`、`text` 和作者关键词来源记录等字段。数据库构建只读取 `doc_id`、`title`、`abstract`；其他原始字段不写入数据库，也不会通过 `SearchResult` 返回。引用展示和元数据过滤仍需后续模块接入这些信息。

JSONL 至少需要非空字符串 `doc_id` 和 `title`，摘要可为字符串、`null` 或省略：

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

构建器只读取 NPZ 的 `vectors` 和 `doc_ids`，不使用 `chunk_ids` 或校验 `metadata`。当前实现要求每个规范化论文 ID 只对应一个向量；若将一篇论文切成多段，不能直接重复 `doc_id` 写入现有归档，需要先调整 ID 与存储设计。

构建器校验非空二维浮点向量、有限值、非零范数，以及一维字符串 ID 数组与向量行数的对应关系；ID 移除 `arxiv:` 前缀后必须唯一，且均存在于数据库。数据库只读打开，映射查询按 SQLite 参数上限分批；无效输入在替换索引前报错。

向量归一化并转换为连续的 `float32`，写入 `IndexIDMap2(IndexHNSWFlat)`，使用内积相似度。参数保持为 `M=128`、`efConstruction=800`、`efSearch=256`。构建成功后通过同目录临时文件替换输出；校验、构建或写入失败时保留已有目标索引。

## 更新约定

- 数据库、FTS5 和 FAISS 必须使用同一套整数 ID 映射；重新创建数据库可能改变映射，需同步重建向量索引。
- 更新数据库不会自动生成论文向量；更新论文文本后，调用方负责重新编码并保存相应向量。
- `Qwen3Vectorizer` 只返回向量。调用方负责保存向量与论文 ID 的对应关系，再显式重建索引。
- 在线检索器复用加载时的索引；更新索引文件后，应创建新的检索器实例。

## 归档模型元数据

当前归档记录的模型为 `Qwen/Qwen3-Embedding-0.6B`，revision 为 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；文档输入为标题与完整摘要，不添加文档指令。采用末尾 token 池化、L2 归一化和余弦相似度，最大输入长度为 32768。

`metadata` 是包含 JSON 文本的 NumPy 标量，还记录了生成时间、环境包版本和 `corpus_sha256`。归档中的来源哈希为 `1454e46d8c2afda47f4578a451aeecf5a078a6aefabe08646acf5ce3b270919b`；本次读取当前 `papers.jsonl` 文件字节得到的 SHA-256 为 `f5d487947be4490255a5f3d0fc2cce088283c848ce3e3ace5f0e916969ed0826`，两者不相同。文件哈希差异本身不能判断文本语义是否改变；正式复现和交付前需核实归档所指源文件、序列化格式及文本对应关系。现有构建流程不会自动检查这项来源一致性。

归档记录的查询模板为：

```text
Instruct: Given a research question, retrieve scientific papers whose core research addresses the question.
Query: {query}
```

当前向量化器保留了不同的默认查询指令（`research topic ... covers the aspects.`），不能将其描述为与归档指令一致。默认值、显式使用归档指令的方法和完整编码配置见[查询指令与归档](vectorization.md#查询指令与归档)。
