# FC-HRRF 四方法代理实验

> 本实验没有人工相关性标签。质量指标把论文的 `collection_topic` 与查询 `topic` 相同视为相关，因此只反映主题级检索能力，不能替代人工0/1/2相关性评估。

- 查询：40条；开发集10条，测试集30条。
- 内部候选深度：50；保存深度：20；指标：@10。
- 开发集选择的覆盖权重：0.5。
- 查询向量预计算耗时：44.998秒；检索延迟不含LLM和查询编码。

## 测试集结果（K=10）

| 方法 | P@5 | P@10 | Recall@10 | MRR@10 | NDCG@10 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.7067 | 0.6967 | 0.00697 | 0.7833 | 0.6958 | 8.420 | 22.804 |
| HNSW | 0.8200 | 0.8233 | 0.00823 | 0.9000 | 0.8221 | 2.108 | 2.785 |
| HRRF | 0.7867 | 0.7700 | 0.00770 | 0.8722 | 0.7756 | 117.520 | 239.124 |
| FC_HRRF | 0.8800 | 0.8133 | 0.00813 | 0.9778 | 0.8416 | 122.742 | 243.565 |

## FC-HRRF 的 NDCG 差值

| 对照方法 | 平均差值 | 95% bootstrap CI |
|---|---:|---:|
| BM25 | +0.1459 | [+0.0934, +0.1998] |
| HNSW | +0.0195 | [-0.0137, +0.0543] |
| HRRF | +0.0661 | [+0.0322, +0.1048] |

## 与 HRRF 的逐查询比较

FC-HRRF 在 16 条测试查询上更高、10 条持平、4 条更低。

| 案例 | 查询 | HRRF NDCG | FC-HRRF NDCG | 差值 |
|---|---|---:|---:|---:|
| 最大提升（q10） | Please recommend papers on verifying LLM reasoning processes, particularly methods for locating errors in intermediate steps. | 0.1610 | 0.4695 | +0.3084 |
| 最大下降（q2） | Please find papers on context compression for RAG, particularly methods that preserve the key evidence needed to answer a question. | 1.0000 | 0.9364 | -0.0636 |

## 开发集权重选择

| coverage weight | NDCG |
|---:|---:|
| 0.1 | 0.7440 |
| 0.2 | 0.7629 |
| 0.3 | 0.7647 |
| 0.4 | 0.7664 |
| 0.5 | 0.7740 |

## 按查询类型的 NDCG@10

| 查询类型 | BM25 | HNSW | HRRF | FC_HRRF |
|---|---:|---:|---:|---:|
| exploration | 0.5310 | 0.5806 | 0.6114 | 0.6667 |
| research_problem | 0.7514 | 0.8916 | 0.8358 | 0.9008 |
| specific_direction | 0.6379 | 0.7856 | 0.6990 | 0.7799 |

## 限制

- `collection_topic` 是构建语料时的收录类别，不代表论文满足查询的具体方法、场景或约束。
- Recall 的分母是该主题的全部论文数；它不是人工判断的完整相关论文集合。
- 正式课程结论仍需使用盲法双人标注的 query–paper 标签重新运行同一评估流程。
