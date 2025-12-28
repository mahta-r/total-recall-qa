import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

from utils.general_utils import set_seed
from c3_task_evaluation.models.retrieval_rankers import create_retriever
from c3_task_evaluation.metrics.retrieval_eval_metrics import evaluate_retrieval_ranking


def run_retrieval_evaluation(args):
    """
    Main function to run passage retrieval evaluation.
    This evaluates retrievers that output ranked lists of passages.
    """
    print("\n== Running Passage Retrieval Evaluation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset_name}
        Retriever:   {args.retriever_name}
        Retriever Type: {args.retriever_type}
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))

    # === Step 0: Read input dataset ========================
    print("\n=== Loading Dataset ===")
    query_ids, test_dataset = [], {}
    if os.path.exists(args.dataset_file):
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    query_ids.append(data['qid'])
                    test_dataset[data['qid']] = data
    print(f"Test dataset size: {len(test_dataset)}")

    # === Read existing results (for resumption) ============
    generated_qids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.add(data['qid'])

    # Filter dataset to only process new queries
    filtered_dataset = {qid: data for qid, data in test_dataset.items() if qid not in generated_qids}
    print(f"Remaining queries to process: {len(filtered_dataset)}")

    # === Initialize Retriever ===============================
    print("\n=== Initializing Retriever ===")
    retriever = create_retriever(args.retriever_type, args.device, args)
    print(f"Retriever initialized: {args.retriever_type} using {args.retriever_name}")

    # === Main Evaluation Loop ===============================
    print("\n=== Starting Retrieval Evaluation ===")
    all_results = []

    with open(args.output_file, 'a') as res_f:
        for i, (qid, sample) in enumerate(tqdm(filtered_dataset.items(), desc="Processing queries")):
            # Extract query
            query = sample.get('total_recall_query', sample.get('query', sample.get('question', '')))

            # Extract gold document IDs
            # TODO: Adjust the field name based on your dataset structure
            # This could be 'intermidate_list', 'gold_doc_ids', 'relevant_docs', etc.
            gold_doc_ids = sample.get('intermidate_list', [])
            if isinstance(gold_doc_ids, str):
                gold_doc_ids = [gold_doc_ids]

            # === Step 1: Retrieve documents =====================
            retrieved_docs, retrieved_doc_ids = retriever.retrieve(query)

            # === Step 2: Evaluate retrieval quality =============
            retrieval_metrics = evaluate_retrieval_ranking(
                retrieved_doc_ids,
                gold_doc_ids,
                k_values=args.k_values
            )

            # === Step 3: Write results to file ===================
            result_item = {
                "qid": qid,
                "query": query,
                "num_gold_docs": len(gold_doc_ids),
                "gold_doc_ids": gold_doc_ids,
                "num_retrieved_docs": len(retrieved_doc_ids),
                "retrieved_doc_ids": retrieved_doc_ids,
                # Retrieval metrics
                "recall": retrieval_metrics['recall'],
                "precision": retrieval_metrics['precision'],
                "mrr": retrieval_metrics['mrr'],
                "map": retrieval_metrics['map'],
                "ndcg": retrieval_metrics['ndcg'],
            }

            # Add k-specific metrics
            for k in args.k_values:
                result_item[f'recall@{k}'] = retrieval_metrics[f'recall@{k}']
                result_item[f'precision@{k}'] = retrieval_metrics[f'precision@{k}']
                result_item[f'ndcg@{k}'] = retrieval_metrics[f'ndcg@{k}']
                result_item[f'hit@{k}'] = retrieval_metrics[f'hit@{k}']

            # Add optional fields if they exist
            if 'file_id' in sample:
                result_item['file_id'] = sample['file_id']
            if 'original_query' in sample:
                result_item['original_query'] = sample['original_query']
            if 'total_recall_answer' in sample:
                result_item['gt_answer'] = sample['total_recall_answer']
            elif 'answer' in sample:
                if isinstance(sample['answer'], dict):
                    result_item['gt_answer'] = sample['answer'].get('value', '')
                else:
                    result_item['gt_answer'] = sample['answer']

            # Store full retrieved documents if requested
            if args.store_retrieved_docs:
                result_item['retrieved_docs'] = retrieved_docs

            all_results.append(result_item)

            # Write to file line by line
            res_f.write(json.dumps(result_item) + '\n')
            res_f.flush()  # Ensure immediate write

    # === Print Summary Statistics ===========================
    print("\n=== Retrieval Evaluation Summary ===")
    if all_results:
        print(f"Number of queries evaluated: {len(all_results)}")
        print(f"\nOverall Metrics:")
        print(f"  Recall:    {np.mean([r['recall'] for r in all_results]):.4f}")
        print(f"  Precision: {np.mean([r['precision'] for r in all_results]):.4f}")
        print(f"  MRR:       {np.mean([r['mrr'] for r in all_results]):.4f}")
        print(f"  MAP:       {np.mean([r['map'] for r in all_results]):.4f}")
        print(f"  NDCG:      {np.mean([r['ndcg'] for r in all_results]):.4f}")

        print(f"\nMetrics by k:")
        for k in args.k_values:
            print(f"  k={k}:")
            print(f"    Recall@{k}:    {np.mean([r[f'recall@{k}'] for r in all_results]):.4f}")
            print(f"    Precision@{k}: {np.mean([r[f'precision@{k}'] for r in all_results]):.4f}")
            print(f"    NDCG@{k}:      {np.mean([r[f'ndcg@{k}'] for r in all_results]):.4f}")
            print(f"    Hit@{k}:       {np.mean([r[f'hit@{k}'] for r in all_results]):.4f}")

        print(f"\nAverage retrieved docs per query: {np.mean([r['num_retrieved_docs'] for r in all_results]):.2f}")
        print(f"Average gold docs per query: {np.mean([r['num_gold_docs'] for r in all_results]):.2f}")
    else:
        print("No new queries were processed.")

    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run passage retrieval evaluation on QA datasets')

    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='qald10',
                        choices=['qald10', 'quest'],
                        help='Dataset to evaluate on')
    parser.add_argument('--dataset_file', type=str, default=None,
                        help='Path to dataset file (overrides dataset_name)')

    # Model arguments
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        choices=[
                            "openai/gpt-4o", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
                            "deepseek/deepseek-chat-v3-0324", "qwen/qwen3-235b-a22b-2507",
                            "Qwen/Qwen2.5-7B-Instruct"
                        ],
                        help='Model for agentic retrieval (ignored for single_step)')

    # Retriever type arguments
    parser.add_argument('--retriever_type', type=str, default='single_step',
                        choices=['single_step', 'agentic', 'reasoning'],
                        help='Type of retrieval: single_step (no reasoning), agentic (with LLM), or reasoning (multi-hop)')

    # Retriever arguments
    parser.add_argument('--retriever_name', type=str, default='contriever',
                        choices=['bm25', 'rerank_l6', 'rerank_l12', 'contriever', 'dpr', 'e5', 'bge'])
    parser.add_argument('--index_dir', type=str, default='/projects/0/prjs0834/heydars/INDICES')
    parser.add_argument('--corpus_path', type=str, default='corpus_datasets/enwiki_20251001.jsonl')
    parser.add_argument('--retrieval_topk', type=int, default=10,
                        help='Number of documents to retrieve')
    parser.add_argument('--faiss_gpu', action='store_false', help='Use GPU for FAISS computation')
    parser.add_argument('--retrieval_query_max_length', type=int, default=64)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='Use FP16 for retrieval')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)

    # Evaluation arguments
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='k values for computing metrics (e.g., Recall@k, NDCG@k)')
    parser.add_argument('--store_retrieved_docs', action='store_true',
                        help='Store full retrieved documents in output (can be large)')

    # Other arguments
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum iterations for agentic retrievers')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--run', type=str, default='run_1', help='Run identifier')
    parser.add_argument("--seed", type=int, default=10, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (overrides default)')

    args = parser.parse_args()

    # === Define CUDA device =======
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")

    # === Set model source for agentic retrieval ====================
    if args.retriever_type in ['agentic', 'reasoning']:
        if args.model_name_or_path in [
            "openai/gpt-4o", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
            "deepseek/deepseek-chat-v3-0324", "qwen/qwen3-235b-a22b-2507"
        ]:
            args.model_source = 'api'
        else:
            args.model_source = 'hf_local'
    else:
        # single_step doesn't need a model source
        args.model_source = None

    # === Set dataset file path if not provided ====================
    if args.dataset_file is None:
        if args.dataset_name == 'qald10':
            args.dataset_file = "corpus_datasets/dataset_creation_heydar/qald10/qald10_queries.jsonl"
        elif args.dataset_name == 'quest':
            args.dataset_file = "corpus_datasets/dataset_creation_heydar/quest/test_quest_queries.jsonl"
        else:
            raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    # === Set output directory and file ====================
    if args.output_dir is None:
        if args.retriever_type == 'single_step':
            args.output_dir = f"run_output/{args.run}/passage_retrieval/{args.dataset_name}/{args.retriever_name}"
        else:
            model_short = args.model_name_or_path.split('/')[-1]
            args.output_dir = f"run_output/{args.run}/passage_retrieval/{args.dataset_name}/{args.retriever_type}_{model_short}_{args.retriever_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = f"{args.output_dir}/retrieval_evaluation_results.jsonl"

    # === Run evaluation ========================
    set_seed(args.seed)
    run_retrieval_evaluation(args)

    # Example usage:
    # python c3_task_evaluation/passage_retrieval_eval.py --dataset_name qald10 --retriever_type single_step --retriever_name contriever
    # python c3_task_evaluation/passage_retrieval_eval.py --dataset_name qald10 --retriever_type agentic --retriever_name bm25 --model_name_or_path "Qwen/Qwen2.5-7B-Instruct"
