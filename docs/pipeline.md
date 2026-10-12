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

## 向量库

使用 `BAAI/bge-m3` 将每篇论文的 `text`（标题＋完整摘要）编码成一个 1,024 维稠密向量。采用模型自带的 CLS 池化，L2 归一化，float32 保存。关键词等元数据不参与编码，查询无需增加前缀。超过 8,192 tokens 的输入会报错，不会静默截断。

`data/vectors.npz` 同时保存 `vectors`、`doc_ids`、`chunk_ids` 和 `metadata`，下标一一对应。`metadata` 包含模型的固定 commit、依赖版本和输入文本指纹。论文正文信息仍从 `papers.jsonl` 读取。

### 环境与生成

使用 Python 3.12。在仓库根目录的 CMD 中执行：

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[embedding]"
set PYTHONPATH=%CD%\src
python -m arxiv_paper_recommend.corpus.processing.embed
```

如需 NVIDIA GPU，先安装 CUDA 版 PyTorch，再安装项目依赖：

```cmd
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[embedding]"
python -m arxiv_paper_recommend.corpus.processing.embed --device cuda --batch-size 8
```

可用 `--device cpu` 在 CPU 上运行。显存不足时调小 `--batch-size`。首次运行会下载模型到 Hugging Face 本地缓存；模型权重不放入仓库。生成完成后整体替换 `vectors.npz`，原论文数据保持不变。

pipeline 默认固定模型 commit 为 `5617a9f61b028005a4858fdac845db406aefb181`，并将其写入库内。也可读取下例中的 `meta["revision"]`，通过 `--revision <commit>` 指定版本。更新论文的 ID、顺序或检索文本后需要重新生成向量库。

### 加载与查询

在上述环境下运行以下 Python 代码。加载时核对语料指纹和 ID，查询使用向量库记录的同一模型版本；归一化向量的点积就是余弦相似度。

```python
import numpy as np
from arxiv_paper_recommend.corpus.processing.embed import (
    load_library, load_encoder, encode_texts, read_papers,
)

vectors, doc_ids, chunk_ids, meta = load_library(
    "data/vectors.npz", "data/papers.jsonl"
)
papers = {p["doc_id"]: p for p in read_papers("data/papers.jsonl")}
model = load_encoder(meta["revision"])  # 模型加载一次，多次查询复用
query = "How can retrieval reduce hallucinations in language models?"
q = encode_texts(model, [query])[0]     # 也可以输入中文
scores = vectors @ q
for i in np.argsort(-scores)[:5]:
    paper = papers[str(doc_ids[i])]
    print(float(scores[i]), paper["title"], paper["url"])
```

这是对全部论文的精确向量检索。后续可按 `doc_id` 与 BM25 结果融合；返回的相似度分数不代表相关概率。

当前库包含 6,000 个向量，文件约 22.9 MB。2026-10-12 已完成加载、ID 对应、数值与归一化检查；12 篇抽样论文重新编码均找回自身，20 个示例的中英文版本共 40 次查询均成功返回结果。这些检查确认向量库可用，检索相关性仍需后续标注评估。验证代码未保存。
