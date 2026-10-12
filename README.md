# arXiv Paper Recommendation

步骤一将原始文献清洗、去重，生成可供步骤二检索的文本和语料统计报告。当前共有 6,000 篇文献，每篇完整标题＋摘要作为一个检索单元。

## 数据文件

`data/` 只保留以下文件：

| 文件 | 内容 |
| --- | --- |
| [sources.json](data/sources.json) | 数据来源、检索范围、采集时间与抽样记录。 |
| [raw_papers.jsonl](data/raw_papers.jsonl) | 原始文献，含标题、完整摘要、作者、关键词与来源链接。 |
| [papers.jsonl](data/papers.jsonl) | 清洗、去重后的文献，每行含 `doc_id`、`chunk_id`、检索用 `text` 和论文信息。 |
| [vectors.npz](data/vectors.npz) | BGE-M3 稠密向量库，包含向量、论文 ID 对应关系与模型版本。 |
| [statistics.md](data/statistics.md) | 主题、年份、分类、文本长度、关键词覆盖及数量统计。 |
| [example_queries.json](data/example_queries.json) | 20 个研究问题示例，覆盖六个主题，含中英文检索句。 |

临时分析结论保存在 [conclusions.md](data/conclusions.md)。

## 运行

在仓库根目录的 CMD 中执行，使用 Python 标准库：

```cmd
set PYTHONPATH=%CD%\src
python -m arxiv_paper_recommend.corpus.processing.prepare
python -m arxiv_paper_recommend.retrieval.bm25.baseline "multi-hop retrieval augmented generation" --top-k 5
python -m unittest discover -s tests -q
```

处理入口只读取 `raw_papers.jsonl`，生成 `papers.jsonl` 和统计报告，不依赖已有处理结果。可用 `--input`、`--output`（输出目录）和 `--report` 指定其他路径。

BM25 示例直接读取 `papers.jsonl`，用 `text` 建内存索引，返回对应论文；后续检索模块可直接使用同一文件。

[数据来源说明](docs/data_sources.md) · [处理与交接说明](docs/pipeline.md)

向量化使用 Python 3.12 和额外模型依赖；[向量库生成与使用方法](docs/pipeline.md#向量库)包含安装、重建和查询示例。
