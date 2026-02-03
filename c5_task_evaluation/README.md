# Task Evaluation (c5)

Runs evaluation on QA datasets with two pipelines:

- **Retrieval**: entity recall@k only (no LLM). Uses retriever args; `retrieval_eval_ks` defines k values.
- **Generation**: full RAG (or LLM-only). Uses `generation_method` and optionally `deep_research_model`.

---

## Main input variables

| Argument | Description | Default |
|----------|-------------|---------|
| `--pipeline` | `retrieval` or `generation` | `retrieval` |
| `--dataset` | `heydar`, `mahta`, or `zahra` | `heydar` |
| `--subset` | `val` or `test` | `test` |
| `--run` | Run ID for output paths | `run_1` |
| `--retriever` | `bm25`, `spladepp`, `contriever`, `dpr`, `e5`, `bge`, `rerank_l6`, `rerank_l12` | `bm25` |
| `--retrieval_eval_ks` | K values for recall@k (retrieval pipeline) | `1 3 10 100` |
| `--retrieval_topk` | Passages to LLM (generation only) | `3` |
| `--model` | Generation model (e.g. `openai/gpt-4o`) | `openai/gpt-4o` |
| `--generation_method` | `no_retrieval`, `single_retrieval`, `deep_research` | `no_retrieval` |
| `--deep_research_model` | When `generation_method=deep_research`: `self_ask`, `react`, `search_o1`, `research`, `search_r1`, `step_search` | `react` |
| `--index_dir` | Directory with retrieval indices | (project default) |
| `--dataset_file` | Override dataset JSONL path | auto from dataset/subset |
| `--qrel_file` | Override qrel path (retrieval) | auto from dataset/subset |
| `--output_dir` | Override output directory | auto from run/subset/retriever or model |
| `--limit` | Max samples (for testing) | None |

---

## Examples

**Retrieval (entity recall@1,3,10,100):**
```bash
python c5_task_evaluation/run_evalution.py --pipeline retrieval --dataset heydar --subset test --retriever bm25 --retrieval_eval_ks 1 3 10 100
python c5_task_evaluation/run_evalution.py --pipeline retrieval --dataset heydar --subset val --retriever contriever
```

**Generation — LLM only:**
```bash
python c5_task_evaluation/run_evalution.py --pipeline generation --dataset heydar --subset val --model openai/gpt-4o --generation_method no_retrieval
```

**Generation — single retrieval + LLM:**
```bash
python c5_task_evaluation/run_evalution.py --pipeline generation --dataset heydar --subset test --model openai/gpt-4o --generation_method single_retrieval --retriever contriever --retrieval_topk 5
```

**Generation — deep research (e.g. ReAct):**
```bash
python c5_task_evaluation/run_evalution.py --pipeline generation --dataset heydar --subset val --model openai/gpt-4o --generation_method deep_research --deep_research_model react --retriever contriever
```

**Quick test (few samples):**
```bash
python c5_task_evaluation/run_evalution.py --pipeline retrieval --dataset heydar --subset test --retriever bm25 --limit 5
```
