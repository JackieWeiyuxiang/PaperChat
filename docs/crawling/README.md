# 步骤一：arXiv 文献摘要抓取

本次贡献仅负责步骤一中的文献抓取：取得真实标题、摘要和元数据，并保存采集代码、100 条样本和验证记录。字段整理与统计用于采集验收。

## 目录分工

以下路径均相对于仓库根目录，各层以 `crawling` 区分这部分工作。

| 路径 | 内容 |
| --- | --- |
| `src/arxiv_paper_recommend/corpus/crawling/arxiv.py` | HTTP 请求、限速、解析、去重、缓存与采集统计 |
| `scripts/crawling/fetch_arxiv.py` | 命令行入口 |
| `data/crawling/raw/arxiv/` | 原始 XML 与请求缓存，本地保存、Git 忽略 |
| `data/crawling/samples/` | 100 条 JSONL 样本及数据 manifest |
| `reports/crawling/` | 连通性、证书排查、OAI 探测、验收与原始实测记录 |
| `tests/crawling/` | 采集模块的离线测试 |
| `docs/crawling/` | 本说明与可交给其他 GPT 的详细策略 |
| `artifacts/crawling/` | 生成的交接 ZIP，Git 忽略 |

## 运行

在仓库根目录执行：

```powershell
python scripts/crawling/fetch_arxiv.py --query "cat:cs.CL AND submittedDate:[202501010000 TO 202610022359]" --limit 100 --page-size 25
python -m unittest discover -s tests/crawling -v
```

也可以从其他目录通过完整路径运行入口。默认输出锚定仓库的 `data/crawling/samples/arxiv_cs_cl_100.jsonl`，默认缓存锚定 `data/crawling/raw/arxiv/`，不会因启动目录不同而散落文件。显式传入的相对 `--output` / `--cache` 路径按当前工作目录解释。

采集脚本已在 Python 3.12.14 下验证。如果 Windows 的 `python` 命令打开商店，请先定位本机真实的 Python 解释器，再在 PowerShell 中使用 `& 'Python解释器的完整路径' scripts/crawling/fetch_arxiv.py ...`。脚本只依赖 Python 标准库；安装整个项目仍须满足 `pyproject.toml` 中既有的 Python >=3.14 要求。

## 已验证结果

2026-10-03 01:00（Asia/Shanghai）完成网络采集，4 页各 25 篇均返回 HTTP 200，包含限速等待耗时 14.828 秒。100 条记录对应 100 个唯一 ID，标题、摘要、作者和日期均完整。摘要按空白分词为 96～280 词，平均 195.6 词。

该样本按提交时间倒序截取，实际发表于 2026-10-01（UTC），用于连通性验证。所有论文带 `cs.CL` 分类，其中 64 篇以它为主分类；不是随机样本、最终评价集或最终知识库范围。

11 项离线测试通过，覆盖限流、Retry-After、403、无效响应、版本去重和断点续跑。禁止网络后的缓存回放可生成 SHA-256 完全相同的 100 条数据。网络耗时不代表 RAG 系统的查询速度。

## 访问与缓存策略

单连接，每次请求结束后至少等待 3.2 秒。每个请求最多重试 2 次，429、500、502、503、504 和连接故障采用退避；尊重 `Retry-After`，要求等待超过 60 秒时退出并提示稍后重试。401/403 或 HTML 验证页立即停止。一次运行限定为 1,000 篇以内、每页最多 200 篇；不要并行启动采集进程。

本机 Python 初次请求出现 `CERTIFICATE_VERIFY_FAILED`，Windows 系统 curl 验证证书后成功。脚本在 Windows 默认采用 curl/Schannel；其他系统默认采用 urllib，可通过 `--transport` 指定。TLS 验证始终开启。

同参数重跑会复用缓存，并更新本次运行的 manifest。要取得更新数据，请使用新的缓存目录和输出文件，区分缓存回放与实时采集。

`reports/crawling/first_success_manifest.json` 保留了重组前的原始实测证据，其中旧路径 `data/raw/arxiv/` 对应现在的 `data/crawling/raw/arxiv/`。当前样本 manifest 已更新为项目根目录相对路径，论文数据字节保持不变。

## 备用路线与交接

官方 OAI-PMH 单篇 GetRecord 已成功获取 `1706.03762`《Attention Is All You Need》的摘要；批量 ListRecords 和续页令牌尚未验证。OAI 日期过滤针对元数据修改时间，不等同于论文提交日期。

官方指向的 [Kaggle arXiv 数据集](https://www.kaggle.com/datasets/Cornell-University/arxiv) 本次未下载，未验证下载权限和速度。

交接文件为 `artifacts/crawling/arxiv_handoff_100.zip`。解压后进入 `arxiv_crawling/`，保留分层结构，按上述命令运行。详细策略见 [arxiv_collection_strategy.txt](arxiv_collection_strategy.txt)。

参考：[API 手册](https://info.arxiv.org/help/api/user-manual.html)、[访问频率要求](https://info.arxiv.org/help/api/tou.html)、[批量数据说明](https://info.arxiv.org/help/bulk_data.html)、[OAI-PMH 文档](https://info.arxiv.org/help/oa/index.html)。
