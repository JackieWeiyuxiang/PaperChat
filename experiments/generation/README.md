# Generation Innovation Benchmark

This experiment compares two paper recommendation generators:

- `CitationFreeRecommendationGenerator`: generates recommendation reasons without requiring citations.
- `EvidenceGroundedRecommendationGenerator`: requires evidence copied from the paper title or abstract inside `[[cite]]...[[/cite]]` markers.

## Experimental Setup

Both methods use:

- the same 10 research queries covering 10 topics;
- the same BM25 retrieval candidates;
- up to 30 candidate papers per query;
- up to 10 final recommendations;
- the same language model and generation workflow.

BM25 is used to fix the candidate sets so that this experiment focuses only on the generation module.

## Results

| Metric | Citation-Free | Evidence-Grounded |
| --- | ---: | ---: |
| Average returned papers | 9.1 | 9.0 |
| Average time (seconds) | 23.42 | 22.36 |
| Citation coverage | N/A | 93.33% |
| Exact citation rate | N/A | 99.41% |
| Evidence verifiability | 1.1 / 2 | 2.0 / 2 |
| LLM judge preference | 0 / 10 | 9 / 10 |

One query was judged as a tie.

The evidence-grounded method improved evidence verifiability while maintaining recommendation relevance, faithfulness, helpfulness, and overall quality.

## Limitations

- The experiment contains only 10 queries.
- It uses BM25 candidates instead of the final end-to-end retrieval pipeline.
- The evaluation is model-assisted rather than a human evaluation.
- Citation formatting was not perfect in every generated recommendation.

## Files

- `benchmark_queries_10.json`: selected benchmark queries.
- `prepare_candidates.py`: prepares fixed retrieval candidates.
- `run_comparison.py`: runs both generators.
- `evaluate_results.py`: calculates automatic metrics.
- `judge_results.py`: performs blind model-assisted evaluation.
- `results/`: stores the evaluation results.
