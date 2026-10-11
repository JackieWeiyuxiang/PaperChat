# 步骤一交接说明

## 交付内容

`data/` 包含数据来源 `sources.json`、原始文献 `raw_papers.jsonl`、清洗去重后的检索文本 `papers.jsonl`、统计报告 `statistics.md`，以及保留的示例查询 `example_queries.json`。

在仓库根目录的 CMD 中先执行 `set PYTHONPATH=%CD%\src`，使 Python 能找到源码模块。

## 处理流程

运行 `python -m arxiv_paper_recommend.corpus.processing.prepare`：

1. 读取原始文献，检查必需字段与论文 ID／版本。
2. 规范化 Unicode 和空白，清除 BOM、零宽空格，将控制字符替换为空格；保留大小写、标点、数字、公式和 LaTeX。
3. 移除完全重复的记录和清洗后标题＋摘要完全相同的文献。同 ID 的冲突内容或多个版本报错，不擅自选择版本。
4. 每篇标题＋完整摘要组成一个检索单元，零重叠、不截断。
5. 保存处理文本并生成语料统计报告。

原始文件不被修改。关键词、作者、分类和链接保留为元数据，检索文本只包含标题与摘要。

## 步骤二如何读取

```python
import json
from pathlib import Path

papers = [json.loads(line) for line in Path("data/papers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
# 用每条记录的 text 建索引；doc_id / chunk_id 用于定位结果。
# 同一条记录内的 title、abstract、url 用于显示论文。
```

`doc_id` 格式为 `arxiv:{paper_id}v{version}`，`chunk_id` 追加 `:abstract:0000`。BM25 示例：

```cmd
python -m arxiv_paper_recommend.retrieval.bm25.baseline "multi-hop retrieval augmented generation" --top-k 5
```

统计见 [statistics.md](../data/statistics.md)。当前 6,000 篇尚需扩充至最终至少 10,000 篇文档或段落。
