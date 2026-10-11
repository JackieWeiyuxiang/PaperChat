"""Render a corpus composition report without intermediate report files."""

from collections import Counter

TOPICS = {"rag": "RAG", "agents": "智能体", "reasoning": "大模型推理",
          "multimodal": "多模态", "image_video_generation": "图像／视频生成",
          "embodied_robotics": "具身智能"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def render_report(documents, stats, input_count, removed):
    count = len(documents)
    share = lambda n: f"{n / count:.2%}" if count else "0.00%"
    topics = Counter(row.get("collection_topic", "未标注") for row in documents)
    years = stats["distributions"]["published_year"]
    categories = sorted(stats["distributions"]["primary_category"].items(), key=lambda x: (-x[1], x[0]))
    category_rows = [[name, n, share(n)] for name, n in categories[:10]]
    if len(categories) > 10:
        other = sum(n for _, n in categories[10:])
        category_rows.append(["其他", other, share(other)])
    statuses = {"found": "已获得", "not_found_in_source": "检查过的来源中未发现",
                "needs_review": "待复核", "fetch_failed": "获取失败", "pending": "尚未处理"}
    parts = ["# 语料统计报告",
             f"原始文献 {input_count:,} 条，精确去重移除 {removed:,} 条，处理后 {count:,} 篇。每篇标题＋完整摘要作为一个检索段落。",
             f"标题＋摘要共 {stats['counts']['corpus_words']:,} 个空白分隔词，按论文计算一次。",
             "## 主题分布", table(["主题", "篇数", "占比"],
                 [[TOPICS.get(name, name), n, share(n)] for name, n in sorted(topics.items())]),
             "主题表示收录分组，每篇只计入一个主题；配额分布不代表整个 arXiv 的论文分布。",
             "## 首次提交年份", table(["年份", "篇数", "占比"],
                 [[year, n, share(n)] for year, n in sorted(years.items())]),
             "## 主题与年份", table(["主题", *sorted(years), "合计"],
                 [[TOPICS.get(topic, topic), *[sum(row.get('collection_topic', '未标注') == topic and
                     row['published'].startswith(year) for row in documents) for year in sorted(years)], n]
                  for topic, n in sorted(topics.items())]),
             "## arXiv 主分类", table(["主分类", "篇数", "占比"], category_rows),
             "按每篇论文的主分类统计，前十类单列，其余合并。主分类与收录主题是不同维度。",
             "## 文本长度", table(["内容", "平均词数", "中位数", "最短", "最长"],
                 [[label, f"{stats['word_statistics'][key]['mean']:.1f}",
                   stats['word_statistics'][key]['median'], stats['word_statistics'][key]['min'],
                   stats['word_statistics'][key]['max']]
                  for key, label in [("title", "标题"), ("abstract", "摘要"), ("combined", "标题＋摘要")]]),
             "词数按空白分词计算，不是模型 token 数。",
             "## 作者关键词", table(["状态", "篇数", "占比"],
                 [[statuses.get(status, status), n, share(n)]
                  for status, n in sorted(stats['author_keywords']['status_counts'].items())]),
             f"关键词覆盖率为 {stats['author_keywords']['coverage_fraction']:.2%}；关键词只作元数据，不拼入检索文本。",
             "## 核心字段完整性", table(["字段", "缺失篇数"],
                 [[name, stats['missing_fields']['documents'][name]]
                  for name in ['title', 'abstract', 'authors', 'paper_id', 'version', 'published', 'categories']]),
             "本报告由步骤一 pipeline 根据本次处理结果生成。"]
    return "\n\n".join(parts) + "\n"
