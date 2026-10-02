# arXiv Paper Recommendation

本次贡献仅涉及步骤一中的文献摘要抓取。

- [采集模块说明与运行方式](docs/crawling/README.md)
- [可交给其他 GPT 的详细采集策略](docs/crawling/arxiv_collection_strategy.txt)
- [100 条摘要样本](data/crawling/samples/arxiv_cs_cl_100.jsonl)

源码位于 `src/arxiv_paper_recommend/corpus/crawling/`，运行入口位于 `scripts/crawling/`。数据、测试、说明和采集记录分别放在 `data/crawling/`、`tests/crawling/`、`docs/crawling/` 和 `reports/crawling/`。
