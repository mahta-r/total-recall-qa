# Task evaluation (c5)

## `run_evalution.py`

Runs evaluation on QA datasets with two pipelines:

- **Retrieval:** entity recall@k only (no LLM). Uses retriever + index; `retrieval_eval_ks` defines k values.
- **Generation:** full RAG or LLM-only. Uses `generation_method` and optionally `deep_research_model`.

**Corpus and indices:** You can download the corpus and retrieval indices from [Hugging Face (HeydarS)](https://huggingface.co/HeydarS):

| Resource | Link |
|----------|------|
| Corpus | [HeydarS/enwiki_20251001_infoboxconv_rewritten](https://huggingface.co/datasets/HeydarS/enwiki_20251001_infoboxconv_rewritten) |
| BM25 index | [HeydarS/enwiki_20251001_bm25_index](https://huggingface.co/datasets/HeydarS/enwiki_20251001_bm25_index) |
| SPLADEPP index | [HeydarS/enwiki_20251001_spladepp_index](https://huggingface.co/datasets/HeydarS/enwiki_20251001_spladepp_index) |
| BGE index | [HeydarS/enwiki_20251001_bge_index](https://huggingface.co/datasets/HeydarS/enwiki_20251001_bge_index) |
| Contriever index | [HeydarS/enwiki_20251001_contriever_index](https://huggingface.co/datasets/HeydarS/enwiki_20251001_contriever_index) |
| E5 index | [HeydarS/enwiki_20251001_e5_index](https://huggingface.co/datasets/HeydarS/enwiki_20251001_e5_index) |

Point `--corpus_path` and `--index_dir` to the downloaded paths.

**Input variables**

| Variable | Description |
|----------|-------------|
| `--pipeline` | `retrieval` or `generation`. |
| `--dataset` | `qald10_quest`, `wikidata`, or `synthetic_ecommerce`. |
| `--subset` | `train`, `val`, or `test`. |
| `--dataset_file` | Override dataset JSONL path. |
| `--qrel_file` | Override qrel path (retrieval). |
| `--retriever` | `bm25`, `spladepp`, `contriever`, `dpr`, `e5`, `bge`, `rerank_l6`, `rerank_l12`, `oracle`. |
| `--index_dir` | Directory containing retrieval indices. |
| `--corpus_path` | Corpus JSONL (for BM25/oracle or lookups). |
| `--retrieval_eval_ks` | K values for recall@k (e.g. `3 10 100 1000`). |
| `--retrieval_topk` | Passages passed to LLM (generation; single_retrieval / deep_research). |
| `--retrieval_results_file` | Optional TREC retrieval results; if set and exists, skip retrieval (single_retrieval). |
| `--model` | Generation model (e.g. `Qwen/Qwen2.5-7B-Instruct`). Used when `pipeline=generation`. |
| `--generation_method` | `no_retrieval`, `single_retrieval`, or `deep_research`. |
| `--deep_research_model` | When deep_research: `self_ask`, `react`, `search_o1`, `research`, `search_r1`, `step_search`. |
| `--run` | Run ID for output paths. |
| `--output_dir` | Override output directory. |
| `--limit` | Max samples (for testing). |
| `--faiss_gpu` | Use GPU for FAISS. |
| `--devices` | Comma-separated GPU IDs (e.g. `0,1,2,3`). |

**Oracle retriever:** `--retriever oracle` is an upper-bound retriever: for each query it returns only the gold (relevant) passage IDs from the qrels, then looks up their contents from the corpus. No index is used. Requires `--qrel_file` and `--corpus_path`. Use it to measure generation quality when retrieval is perfect (e.g. to compare with real retrievers like E5).

**Examples**

All examples use one flag per line. Run from project root.

**1. Retrieval only** (entity recall@3,10,100,1000):

```bash
python c5_task_evaluation/run_evalution.py \
  --pipeline retrieval \
  --dataset wikidata \
  --subset test \
  --retriever e5 \
  --retrieval_eval_ks 3 10 100 1000 \
  --run run_1
```

**2. Generation, no retrieval** (LLM only, GPT-5.2):

```bash
python c5_task_evaluation/run_evalution.py \
  --pipeline generation \
  --dataset wikidata \
  --subset test \
  --model openai/gpt-5.2 \
  --generation_method no_retrieval \
  --run run_1
```

**3. Single-step retrieval + generation** (E5 + GPT-5.2):

```bash
python c5_task_evaluation/run_evalution.py \
  --pipeline generation \
  --dataset wikidata \
  --subset test \
  --model openai/gpt-5.2 \
  --generation_method single_retrieval \
  --retriever e5 \
  --retrieval_topk 3 \
  --run run_1
```

**4. Deep research (search_r1):** Interleaved retrieval + generation over multiple steps. You get **retrieval eval** and **generation eval**. Retrieval eval is computed by merging, per query, all step-level ranked lists into one list (interleave with dedup by doc ID), then computing entity recall@k on that merged list. Generation eval is the usual exact/soft match on the final answer.

```bash
python c5_task_evaluation/run_evalution.py \
  --pipeline generation \
  --dataset wikidata \
  --subset test \
  --model openai/gpt-5.2 \
  --generation_method deep_research \
  --deep_research_model search_r1 \
  --retriever e5 \
  --retrieval_topk 3 \
  --retrieval_eval_ks 3 10 100 1000 \
  --run run_1
```

**Script (run from project root):** `scripts/evaluation/run_evaluation.sh` — set `pipeline`, `dataset`, `subset`, `retriever`, `model`, `generation_method`, `deep_research_model`, `run` in the script then run (e.g. via Slurm).

---

## `significance_test.py`

Compares two systems using per-query metrics: paired Wilcoxon signed-rank test and optional bootstrap 95% CI for the difference. Use after runs that write `evaluation_results_per_query_metrics.jsonl`.

**Input variables**

| Variable | Description |
|----------|-------------|
| `metrics_a` | Path to first system’s `evaluation_results_per_query_metrics.jsonl`. |
| `metrics_b` | Path to second system’s `evaluation_results_per_query_metrics.jsonl`. |
| `--metric` | Metric to compare (e.g. `entity_recall@10`, `entity_recall@100`, `exact_match`, `soft_match_5.0`). Default: `entity_recall@10`. |
| `--names` | Two display names for the systems (e.g. `bm25 spladepp`). |
| `--alpha` | Significance level (default: 0.05). |
| `--bootstrap` | Bootstrap sample count for CI (0 to disable). Default: 10000. |

**Examples**

Retrieval (bm25 vs spladepp on entity_recall@10):

```bash
python c5_task_evaluation/significance_test.py \
  run_output/run_1/qald10_quest_test/retrieval_bm25/evaluation_results_per_query_metrics.jsonl \
  run_output/run_1/qald10_quest_test/retrieval_spladepp/evaluation_results_per_query_metrics.jsonl \
  --metric entity_recall@10 --names bm25 spladepp
```

Generation (exact_match: e5 vs oracle):

```bash
python c5_task_evaluation/significance_test.py \
  run_output/run_1/qald10_quest_test/generation_gpt-5.2_single_retrieval_e5/evaluation_results_per_query_metrics.jsonl \
  run_output/run_1/qald10_quest_test/generation_gpt-5.2_single_retrieval_oracle/evaluation_results_per_query_metrics.jsonl \
  --metric exact_match --names e5 oracle
```
