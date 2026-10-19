# P3：检索创新与 Generation Baseline

本文记录 P3 在课程作业中的实现方案、代码位置、验证方式和当前证据边界。P3 的重点包括两部分：一是针对现有检索基线提出并实现一项 retrieval innovation；二是实现用于论文推荐的 generation baseline。本文区分“代码已实现”“做过探索性观察”和“尚无数据可验证”三种状态，避免把排名变化误写成相关性提升。

## 1. 项目现有检索基线与问题分析

项目已有三类检索能力：

- **稀疏检索 BM25**：基于 SQLite FTS5，对论文标题和摘要做关键词匹配。
- **稠密检索 HNSW**：使用 Qwen3 向量表示检索语义相近的论文。
- **混合检索**：包括 HRRF、Semantic RRF 和 SD Cascade，将稀疏与稠密检索结果进行融合或级联。

基线能覆盖词面匹配、语义相似和多路融合，但仍有以下问题：

1. **BM25 受词汇不匹配影响。** 用户问题和论文摘要使用不同表达、同义词或缩写时，相关论文可能无法获得足够高的词面得分。查询扩展可以增加召回机会，但效果也会依赖扩展词是否准确。
2. **标题与摘要被同等对待。** 当前基线对标题和摘要使用相同字段权重。标题通常概括论文主题，长摘要则可能因为偶然出现某个查询词而获得较高分数。
3. **稠密相似度对精确术语不够敏感。** 数据集名称、方法名、缩写和罕见技术术语通常需要精确词面匹配；只使用语义向量可能把相近但不满足具体约束的论文排到前面。
4. **融合方法的名称不能证明其更好。** 三种混合策略的候选数量、融合方式不同。没有统一查询、人工相关性标注和相同评估条件，就不能据此判断哪种方法效果更优。

基于以上问题，P3 实现了一个范围清晰、可以和现有 BM25 直接比较的创新：提高标题匹配在 BM25 中的权重。

## 2. Retrieval Innovation：标题加权 BM25

### 2.1 设计思路

创新方法保留现有 FTS5 索引、分词结果和 BM25 检索流程，只改变字段权重：

| 字段 | 基线权重 | 创新方法权重 |
| --- | ---: | ---: |
| 标题 `title` | 1.0 | 3.0 |
| 摘要 `abstract` | 1.0 | 1.0 |
| 文档编号 `doc_id` | 0.0 | 0.0 |

因此，方法暂命名为 **Title-Boosted BM25（标题加权 BM25）**。3:1 是一个待评估的初始假设，不是通过标注数据调优得到的最优参数。预期是：当标题直接点明研究主题时，相关论文更容易靠前；可能的副作用是，标题中只出现宽泛查询词、却没有满足用户具体意图的论文也可能被提升。

### 2.2 新增代码模块与职责

本次新增和修改的主要文件如下：

| 路径 | 作用 |
| --- | --- |
| `src/arxiv_paper_recommend/retrieval/innovation/title_aware_bm25.py` | 新增 `TitleAwareBM25Retriever`，通过 FTS5 BM25 字段权重实现标题 3、摘要 1 的检索。 |
| `src/arxiv_paper_recommend/retrieval/innovation/__init__.py` | 导出检索创新模块的公开类。 |
| `src/arxiv_paper_recommend/retrieval/sparse/bm25_retriever.py` | 将等权 BM25 查询整理为内部可复用方法，并支持指定字段权重；原有 BM25 基线保持等权行为。 |
| `src/arxiv_paper_recommend/retrieval/search_engine.py` | 把 `TITLE_BM25` 接入统一检索入口，允许通过 SearchEngine 使用创新方法。 |
| `src/arxiv_paper_recommend/evaluation/retrieval.py` | 新增固定检索输入的多方法运行、候选池导出和 qrels 排名指标计算。 |
| `structure.md` | 记录新增模块的职责和数据流。 |

统一检索接口的调用形式为：

```python
from arxiv_paper_recommend.retrieval.search_engine import SearchEngine

with SearchEngine() as engine:
    results = engine.search(request, method="TITLE_BM25")
```

底层检索器也可以独立使用。其核心行为是调用现有 BM25 检索逻辑，并传入 `column_weights=(3.0, 1.0, 0.0)`。这使创新方法和基线复用同一数据、同一查询词和同一 FTS5 索引，便于开展公平对照。

### 2.3 初步探索性对比

在当前 10,000 篇论文数据库上，用两个手工固定的词表分别运行等权 BM25 与标题加权 BM25，并观察 Top 8 排名。以下结果只说明排名如何变化，不代表人工判断的相关性提升：

| 用户需求示例 | 观察结果 |
| --- | --- |
| 寻找评估 RAG 回答中引用是否真正支持具体论断的论文；固定词包括 `citation reliability`、`citation support`、`claim attribution`、`retrieval augmented generation`。 | 两篇直接讨论引用评估的论文仍排在第 1、2 名。一篇标题含“Retrieval-Augmented Generation”的文化知识评测论文从第 8 名升到第 3 名；一篇 RAG 基准论文从第 3 名降到第 4 名。新 Top 8 中有 6 篇发生了名次变化。该例提示宽泛标题词可能造成误提升。 |
| 寻找利用其他文件检索代码上下文的仓库级代码补全论文；固定词包括 `repository-level code completion`、`cross-file code`、`code retrieval`、`repository code context`、`code completion`。 | 基线和创新方法的 Top 8 论文及顺序相同。这个查询中的核心术语已经出现在候选论文标题里，增加标题权重没有改变排序。 |

这些案例体现了方法的适用条件和风险：标题直接表达用户主题时，加权可能有帮助；查询包含宽泛词时，加权也可能放大不够相关的结果。下一步应在开发集上比较多个标题权重，并检查被提升的误检结果；参数确定后，再在独立测试集上报告结果。不能使用测试集标签来选择权重。

### 2.4 正式评估设计与当前限制

评估脚本可对以下方法使用同一组冻结输入进行对照：`BM25`、`TITLE_BM25`、`HNSW`、`HRRF`、`SEMANTIC_RRF` 和 `SD_CASCADE`。冻结输入记录包含查询文本、稀疏词项、稠密查询和评分查询，避免比较时同时改变查询生成方式。脚本会运行多次、记录耗时、导出各方法排名以及去重后的候选池；获得人工相关性标注后，可计算 Precision@K、Recall@K、MRR@K 和 NDCG@K。

**当前尚不能报告正式指标或宣称创新优于基线。** 工作区没有 P2 的基线运行结果，没有冻结的正式评估输入，也没有 P5 的 qrels 人工标注。评估脚本会在某个查询的候选中存在未标注论文时，将该查询从所有被比较方法中一并排除，以免未判断项目被错误当作不相关项。

## 3. Generation Baseline：检索结果上的论文推荐生成

### 3.1 任务定义和方案选择

Generation baseline 接收用户的研究需求和检索阶段返回的论文候选，完成两件事：

1. 从检索候选中选择最多指定数量的论文，并按对需求的有用程度排序。
2. 为每篇入选论文生成一至两句简短理由，说明论文标题或摘要如何对应用户需求。

本项目选择“**检索后生成**”作为基础方案：检索负责提供证据候选，语言模型负责筛选和组织自然语言理由。这个方案适合论文推荐任务，因为用户请求的研究意图表达多样，而论文主题和方法信息主要以非结构化标题、摘要呈现。当前没有足够的任务专用标注数据支持训练分类器或微调生成模型，因此先建立一个可调用、输出可追溯的零样本生成基线。

### 3.2 新增代码模块与职责

| 路径 | 作用 |
| --- | --- |
| `src/arxiv_paper_recommend/recommendation/generation/baseline_generator.py` | 实现结构化推荐输出、候选筛选、理由生成和结果校验。 |
| `src/arxiv_paper_recommend/recommendation/generation/__init__.py` | 导出 `RecommendationGenerator` 和 `PaperRecommendation`，作为包级调用入口。 |
| `src/arxiv_paper_recommend/config.py` | 提供 LLM API Key、API 地址和模型名配置，供生成器创建客户端和选择模型。 |
| `.env.example` | 展示所需环境变量的配置格式；实际密钥应保存在本地 `.env`，不提交到代码仓库。 |
| `structure.md` | 记录 generation 模块在项目结构中的位置、职责和数据流。 |

`RecommendationGenerator` 使用 OpenAI Responses API 的结构化解析能力和 Pydantic 输出模型。输入候选来自检索结果 `SearchResult`；候选提供 `doc_id`、标题和摘要，不把候选之外的论文交给模型。模型最多接收前 30 个去重候选，最终推荐数量由 `limit` 指定，默认最多 10 篇。

调用示例：

```python
from arxiv_paper_recommend.recommendation.generation import RecommendationGenerator

with RecommendationGenerator() as generator:
    recommendations = generator.generate(
        request="寻找评估 RAG 回答中引用可靠性的论文",
        papers=search_results,
        limit=10,
    )
```

### 3.3 生成步骤与输出结构

生成流程如下：

```text
研究需求 + SearchResult 候选
          ↓
检查输入并截取最多 30 篇去重候选
          ↓
仅向模型提供 doc_id、title、abstract
          ↓
Responses API 按 Pydantic 结构返回推荐 ID 和理由
          ↓
校验 ID、去重、检查理由非空
          ↓
将 ID 映射回原 SearchResult 并分配 rank
```

模型返回的中间结构为：

```json
{
  "recommendations": [
    {
      "doc_id": "论文编号",
      "reason": "根据标题或摘要说明它与用户研究需求的联系。"
    }
  ]
}
```

程序最终返回 `list[PaperRecommendation]`。每条结果包括：

- `rank`：生成结果中的顺序，从 1 开始；
- `result`：原始 `SearchResult`，保留论文编号、标题和摘要等检索证据；
- `reason`：模型生成的一至两句推荐理由。

保留原始 `SearchResult` 的好处是推荐理由仍能关联到对应的检索论文，后续界面或评估流程可以读取同一候选的证据内容，而不需要根据模型重写的标题猜测来源。

### 3.4 提示约束与程序校验

生成指令要求模型：

- 只使用传入的检索候选，不得自行补充论文；
- 按对研究需求的有用程度排序；
- 理由必须由对应论文的标题或摘要支持，不使用外部知识；
- 当给定证据不足以支持推荐时，省略该论文；
- 每条理由保持简短。

程序还会执行结构和来源校验：

- 输出的 `doc_id` 必须属于本次提供的候选；
- 同一论文不能重复出现；
- 理由不能为空；
- 超出请求数量的结果不会返回；
- 没有结构化结果时会报错，而不是把无法解析的文本当作有效推荐。

模型通过配置的 Responses API 模型运行，采样温度设为 0，以减少随机变化。由生成器自行创建的客户端会随生成器关闭；通过参数注入的客户端仍由调用方负责生命周期管理。

### 3.5 能力边界和后续评估

此版本是 **generation baseline**，不是经过证据核验的最终推荐系统。结构化输出和 ID 校验只能确保格式正确、推荐对象来自候选，不能证明自然语言理由中的每个事实都被摘要蕴含。当前版本也不生成出版商链接或逐句引用标记。理由的支持性核验、来源引用和更强的下游创新留给后续模块处理。

生成质量应按作业要求使用独立评估协议衡量，例如检查：

- **Faithfulness（忠实度）**：理由是否被所对应论文的标题或摘要支持；
- **Answer relevance（回答相关性）**：推荐列表及理由是否回应了用户的研究需求；
- **Context relevance / precision（上下文相关性／精确率）**：检索到并用于生成的论文候选是否与需求相关。

当前工作区没有 LLM 环境配置，也没有执行真实模型生成，因此本文不展示模型生成样例，不声称生成质量达到任何数值。运行前需在本地 `.env` 中设置 `LLM_API_KEY`、`LLM_API_URL` 和 `LLM_MODEL`。模型、API 兼容性及输出质量均需在小规模调用和人工评估中确认。

## 4. P3 交付状态与依赖

| 工作项 | 当前状态 | 后续依赖 |
| --- | --- | --- |
| 分析 BM25、稠密检索和混合检索的局限 | 已完成分析并写入本文 | 结合小组最终方案检查表述 |
| Retrieval innovation 实现 | 已实现标题加权 BM25，并接入统一检索入口 | 小组确认创新方案；在开发集确定权重 |
| 创新方法的探索性对比 | 已对两个冻结词表示例观察 Top 8 排名 | 不能替代有标注评估 |
| 多方法评估工具 | 已实现固定输入运行、候选池导出及排名指标计算 | P2 提供基线输出；P5 提供覆盖候选池的人工 qrels |
| Generation baseline 实现 | 已实现结构化选择与理由生成、候选 ID 校验和结果映射 | 配置可用的 LLM API，并与小组确认模型和输出字段 |
| Generation 质量评估 | 尚无结果 | 需要真实生成输出、P4 的评估方案及人工／自动评分 |

P3 已完成可独立实现的代码和评估框架工作。正式的检索效果结论依赖 P2 的基线运行、P5 的相关性判断以及小组对创新设定的确认；生成质量结论还依赖可用模型配置和后续评估。拿到这些材料后，应在同一冻结查询集和同一候选池上补充定量结果，并如实报告正向、无变化或负向结果。
