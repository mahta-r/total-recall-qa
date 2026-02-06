#!/usr/bin/env python3
"""
Index Builder Test Script

Tests that the indexing pipeline works correctly by:
1. Creating a small corpus from a subset of queries, their qrels, and gold passages
2. Building an index on that small corpus
3. Running retrieval using c5_task_evaluation retrieval models
4. Evaluating entity recall

High recall indicates the indexing pipeline is working correctly (gold passages are in the corpus
and should be retrievable when the index is built properly).

Inputs: Same as index_builder.py plus:
  - --query_file: Path to query JSONL (or use --dataset/--subset for defaults)
  - --qrel_file: Path to TREC-format qrels
  - --query_limit: Optional limit on number of queries to use (default: 10)
  - --sub_corpus_max_size: Total sub corpus size (gold passages + distractors). If set, e.g. 1000,
    the sub corpus has 1000 passages total: all gold passages for the queries plus non-gold
    distractors to reach 1000. If not set, sub corpus = gold passages only.

Usage:
  python c2_corpus_creation/index_builder_test.py \\
    --retrieval_method bge \\
    --corpus_path corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl \\
    --save_dir /tmp/index_test \\
    --query_file corpus_datasets/generated_datasets/queries/queries_wikidata_test.jsonl \\
    --qrel_file corpus_datasets/generated_datasets/qrels/trec_qrels_wikidata_test.txt \\
    --query_limit 20
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

# Add project root for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from tqdm import tqdm

from c2_corpus_creation.index_builder import Index_Builder, MODEL2PATH, MODEL2POOLING
from c5_task_evaluation.metrics.retrieval_eval_metrics import load_qrels, evaluate_entity_retrieval


def load_queries(query_file: str, limit: int = None):
    """Load queries from JSONL. Returns list of (qid, query_text) tuples."""
    queries = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get('id', data.get('qid'))
            query_text = data.get('total_recall_query', data.get('query', data.get('question', '')))
            if qid and query_text:
                queries.append((qid, query_text))
    return queries


def get_gold_passage_ids_for_queries(qrels: dict, query_ids: set) -> set:
    """Get all unique gold passage IDs for the given query IDs."""
    gold_ids = set()
    for qid in tqdm(query_ids, desc="Collecting gold passage IDs", unit="query"):
        gold_ids.update(qrels.get(qid, []))
    return gold_ids


def create_small_corpus(
    full_corpus_path: str,
    gold_passage_ids: set,
    output_path: str,
    max_size: int = None,
):
    """
    Create sub corpus: gold passages plus distractors up to max_size.
    - Gold passages: all relevant passages for the query subset (from qrels).
    - If max_size is set: add non-gold passages (distractors) until total length = max_size.
    - If max_size is None: sub corpus = only gold passages (no limit).
    """
    gold_ids = frozenset(gold_passage_ids)
    found_gold = set()
    written = 0

    with open(full_corpus_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        # Phase 1: write all gold passages
        for line in tqdm(infile, desc="Phase 1: extracting gold passages", unit="line"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = doc.get('id', doc.get('doc_id', doc.get('passage_id')))
            if doc_id is not None and str(doc_id) in gold_ids:
                outfile.write(json.dumps(doc, ensure_ascii=False) + '\n')
                written += 1
                found_gold.add(str(doc_id))

        # Phase 2: if max_size set, add distractors (non-gold) until we reach max_size
        if max_size is not None and written < max_size:
            infile.seek(0)
            for line in tqdm(infile, desc="Phase 2: adding distractors", unit="line"):
                if written >= max_size:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = doc.get('id', doc.get('doc_id', doc.get('passage_id')))
                if doc_id is not None and str(doc_id) not in gold_ids:
                    outfile.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    written += 1

    missing = gold_ids - found_gold
    if missing:
        print(f"Warning: {len(missing)} gold passage IDs not found in corpus (e.g. {list(missing)[:5]})")
    n_gold = len(found_gold)
    n_distractors = written - n_gold
    if max_size is not None:
        print(f"Small corpus: {written} passages total ({n_gold} gold, {n_distractors} distractors) -> {output_path}")
    else:
        print(f"Small corpus: {written} passages (gold only) -> {output_path}")
    return written


def create_qrels_subset(qrels: dict, query_ids: set, output_path: str, qrel_file_format: str = "trec"):
    """Write qrels filtered to the query subset (TREC format)."""
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for qid in sorted(query_ids):
            for passage_id in qrels.get(qid, []):
                # TREC format: query_id 0 passage_id relevance
                f.write(f"{qid} 0 {passage_id} 1\n")
                count += 1
    print(f"Qrels subset: {count} rows for {len(query_ids)} queries -> {output_path}")
    return count


def _get_passage_id(doc):
    """Extract passage ID from retrieved doc."""
    for key in ('id', 'doc_id', 'passage_id', 'docid'):
        val = doc.get(key)
        if val is not None:
            return str(val)
    if 'title' in doc:
        return str(doc['title']).strip()
    raise KeyError("Retrieved doc has no id field. Keys: %s" % list(doc.keys()))


def run_retrieval_and_evaluate(
    retriever,
    queries: list,
    qrels: dict,
    k_values: list,
):
    """Run retrieval for each query and compute entity recall metrics."""
    results = []
    for qid, query_text in tqdm(queries, desc="Retrieval", unit="query"):
        gold_ids = qrels.get(qid, [])
        try:
            docs, scores = retriever.search(query_text, return_score=True)
        except Exception as e:
            print(f"Retrieval error for qid={qid}: {e}")
            docs, scores = [], []
        retrieved_ids = [_get_passage_id(d) for d in docs]
        metrics = evaluate_entity_retrieval(retrieved_ids, gold_ids, k_values)
        metrics['qid'] = qid
        results.append(metrics)
    return results


def main():
    parser = argparse.ArgumentParser(description="Index builder test: small corpus -> index -> retrieval -> recall")

    # Index builder args (same as index_builder.py)
    parser.add_argument('--retrieval_method', type=str, default='spladepp', choices=['bm25', 'spladepp', 'contriever', 'dpr', 'e5', 'bge'])
    parser.add_argument('--corpus_path', type=str, default="corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl", help='Path to FULL corpus JSONL (to extract gold passages from)')
    parser.add_argument('--save_dir', type=str, default="corpus_datasets/corpus/index_builder_test", help='Directory for test index and outputs (default: temp dir)')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--faiss_type', type=str, default='Flat')
    parser.add_argument('--embedding_path', type=str, default=None)
    parser.add_argument('--save_embedding', action='store_true', default=True)
    parser.add_argument('--use_fp16', action='store_true', default=True)
    parser.add_argument('--faiss_gpu', action='store_true', default=False)

    # Query / qrel args
    parser.add_argument('--query_file', type=str, default=None, help='Path to query JSONL')
    parser.add_argument('--qrel_file', type=str, default=None, help='Path to TREC-format qrels')
    parser.add_argument('--dataset', type=str, default='qald10_quest', help='Dataset name (used if query_file/qrel_file not set)')
    parser.add_argument('--subset', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--query_limit', type=int, default=10, help='Max number of queries to use')
    parser.add_argument('--sub_corpus_max_size', type=int, default=3000, help='Total sub corpus size: gold passages + distractors up to this many (default: gold only, no limit)')

    args = parser.parse_args()

    # Default paths
    if args.query_file is None:
        args.query_file = f"corpus_datasets/generated_datasets/queries/queries_{args.dataset}_{args.subset}.jsonl"
    if args.qrel_file is None:
        args.qrel_file = f"corpus_datasets/generated_datasets/qrels/trec_qrels_{args.dataset}_{args.subset}.txt"
    if args.save_dir is None:
        args.save_dir = tempfile.mkdtemp(prefix='index_builder_test_')
        print(f"Using temp save_dir: {args.save_dir}")

    # Resolve model path
    args.model_path = MODEL2PATH.get(args.retrieval_method, '')
    pooling_method = MODEL2POOLING.get(args.retrieval_method)

    # Check files exist
    for p, name in [(args.query_file, 'query_file'), (args.qrel_file, 'qrel_file'), (args.corpus_path, 'corpus_path')]:
        if not Path(p).exists():
            print(f"Error: {name} not found: {p}")
            return 1

    print("=" * 70)
    print("INDEX BUILDER TEST")
    print("=" * 70)

    # 1. Load queries and qrels
    print("\n=== 1. Loading queries and qrels ===")
    queries = load_queries(args.query_file, limit=args.query_limit)
    if not queries:
        print("Error: No queries loaded")
        return 1
    query_ids = {qid for qid, _ in queries}
    qrels = load_qrels(args.qrel_file)
    queries_with_qrels = [(qid, qt) for qid, qt in queries if qid in qrels]
    if not queries_with_qrels:
        print("Error: No queries have qrels. Check query IDs match qrel file.")
        return 1
    print(f"Queries: {len(queries_with_qrels)} (with qrels)")

    # 2. Get gold passage IDs and create small corpus (or use existing)
    os.makedirs(args.save_dir, exist_ok=True)
    small_corpus_path = os.path.join(args.save_dir, "small_corpus.jsonl")
    if os.path.exists(small_corpus_path):
        n_lines = sum(1 for _ in open(small_corpus_path, 'r', encoding='utf-8'))
        print(f"\n=== 2. Small corpus (using existing) ===")
        print(f"Small corpus exists: {small_corpus_path} ({n_lines} passages)")
    else:
        print("\n=== 2. Creating small corpus (gold passages + distractors) ===")
        gold_ids = get_gold_passage_ids_for_queries(qrels, {qid for qid, _ in queries_with_qrels})
        if not gold_ids:
            print("Error: No gold passage IDs found")
            return 1
        create_small_corpus(args.corpus_path, gold_ids, small_corpus_path, max_size=args.sub_corpus_max_size)

    # 3. Build index on small corpus
    print("\n=== 3. Building index ===")
    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=small_corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=pooling_method,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
    )
    index_builder.build_index()

    # 4. Initialize retriever and run retrieval
    print("\n=== 4. Running retrieval ===")
    from c5_task_evaluation.src.retrieval_models_local import BM25Retriever, DenseRetriever, SPLADERetriever

    # Build minimal config object for retriever
    class RetrieverConfig:
        pass
    config = RetrieverConfig()
    config.retriever_name = args.retrieval_method
    config.corpus_path = small_corpus_path
    config.index_dir = args.save_dir
    config.retrieval_topk = 1000  # retrieve enough for recall@k
    config.retrieval_batch_size = 32
    config.retrieval_query_max_length = 64
    config.retrieval_use_fp16 = args.use_fp16
    config.bm25_k1 = 0.9
    config.bm25_b = 0.4
    config.faiss_gpu = args.faiss_gpu
    config.device = __import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    config.splade_max_length = args.max_length  # Match index build for SPLADE

    if args.retrieval_method == 'bm25':
        retriever = BM25Retriever(config)
    elif args.retrieval_method == 'spladepp':
        retriever = SPLADERetriever(config)
    elif args.retrieval_method in ['contriever', 'dpr', 'e5', 'bge']:
        retriever = DenseRetriever(config)
    else:
        print(f"Error: Unsupported retriever for test: {args.retrieval_method}")
        return 1

    k_values = [1, 3, 10, 100]
    results = run_retrieval_and_evaluate(retriever, queries_with_qrels, qrels, k_values)

    # 5. Aggregate and report recall
    print("\n=== 5. Recall evaluation ===")
    n = len(results)
    if n == 0:
        print("No results to aggregate")
        return 1

    recall_all = sum(r['entity_recall'] for r in results) / n
    recall_by_k = {k: sum(r[f'entity_recall@{k}'] for r in results) / n for k in k_values}

    print(f"\nEntity Recall (index builder test) over {n} queries:")
    for k in k_values:
        print(f"  Entity Recall@{k:3d}: {recall_by_k[k]:.4f} ({recall_by_k[k]*100:.2f}%)")
    print(f"  Entity Recall (all): {recall_all:.4f} ({recall_all*100:.2f}%)")

    # Save metrics
    metrics_path = os.path.join(args.save_dir, "index_test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'n': n,
            'retrieval_method': args.retrieval_method,
            'entity_recall': recall_all,
            'entity_recall_by_k': recall_by_k,
        }, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Interpretation
    print("\n" + "=" * 70)
    if recall_all >= 0.9:
        print("PASS: High recall indicates the indexing pipeline is working correctly.")
    elif recall_all >= 0.5:
        print("PARTIAL: Moderate recall. Check index build parameters and model compatibility.")
    else:
        print("LOW RECALL: Index may have issues. Verify corpus format, model, and index/query encoder alignment.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())


# python c2_corpus_creation/index_builder_test.py
