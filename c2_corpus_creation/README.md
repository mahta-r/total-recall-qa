# Corpus creation

## `index_builder.py`

Builds retrieval indices from a JSONL corpus. Supports:

- **Dense**: contriever, dpr, e5, bge, reasonir (FAISS index + optional saved embeddings)
- **Sparse**: spladepp, bm25

**Input variables**

| Variable | Description |
|----------|-------------|
| `--retrieval_method` | Retriever: `contriever`, `dpr`, `e5`, `bge`, `reasonir`, `spladepp`, `bm25`. |
| `--corpus_path` | Input JSONL corpus (one doc per line; each doc has `id` and `contents`/`content`). |
| `--save_dir` | Directory where the index (and optionally embeddings) are written. |
| `--max_length` | Max token length per passage (dense; default 512). |
| `--batch_size` | Encoding batch size (dense; default 512). |
| `--faiss_type` | FAISS index type, e.g. `Flat` (dense). |
| `--use_fp16` | Use half precision for encoding (dense). |
| `--save_embedding` | Save passage embeddings to disk (dense/spladepp). |
| `--embedding_path` | If set, skip encoding and build index from this JSONL of vectors (e.g. spladepp index-only). |
| `--faiss_gpu` | Build FAISS index on GPU (dense). |

**Example** (dense index with BGE, from project root):

```bash
python c2_corpus_creation/index_builder.py \
  --retrieval_method bge \
  --corpus_path corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl \
  --save_dir /path/to/indices \
  --use_fp16 \
  --max_length 512 \
  --batch_size 512 \
  --faiss_type Flat \
  --save_embedding
```

**Corpus JSONL:** one JSON object per line with `id` and `contents` (or `content`) per document. This setup follows the [Pyserini](https://github.com/castorini/pyserini) format.

---

## Test indexing

**Why:** Validates the full indexing pipeline before running on the full corpus. If gold passages from qrels are retrievable on a small sub-corpus, the index and retrieval setup are correct.

**Steps:** (1) Build a small sub-corpus from query qrels (gold passage IDs + optional distractors). (2) Build an index on that sub-corpus. (3) Run retrieval with the same retriever. (4) Evaluate entity recall; high recall = pipeline OK.

The script uses the same variables as `index_builder.py` plus:

| Variable | Description |
|----------|-------------|
| `--query_file` | Query JSONL (qid + query text). |
| `--qrel_file` | TREC-format qrels (query id → relevant doc ids). |
| `--query_limit` | Max number of queries to use (script default: 20). |
| `--sub_corpus_max_size` | Sub-corpus size: gold passages + distractors up to this many; unset = gold only. |
| `--dataset` / `--subset` | Used when `query_file`/`qrel_file` are not set (defaults: `qald10_quest`, `test`). |

Results go to `/tmp/index_builder_test_<retriever>/` (or `--save_dir` if set).

**Example** (from project root):

```bash
python c2_corpus_creation/index_builder_test.py \
  --retrieval_method bge \
  --corpus_path corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl \
  --save_dir /tmp/index_builder_test_bge \
  --query_file corpus_datasets/generated_datasets/queries/queries_wikidata_test.jsonl \
  --qrel_file corpus_datasets/generated_datasets/qrels/trec_qrels_wikidata_test.txt \
  --query_limit 20 \
  --use_fp16 --max_length 512 --batch_size 64
```

---

## Scripts (run from project root)

| Script | Purpose |
|--------|---------|
| `scripts/corpus/build_index.sh` | Dense index (e.g. bge) with GPU |
| `scripts/corpus/build_index_bm25.sh` | BM25 index |
| `scripts/corpus/build_index_spladepp.sh` | SPLADEPP index (optionally from precomputed `embedding_path`) |
| `scripts/corpus/run_index_builder_test.sh` | Test indexing (small sub-corpus → index → retrieval → recall) |
