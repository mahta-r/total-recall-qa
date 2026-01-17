#!/usr/bin/env python3
"""
Evaluation Pipeline Runner

This script runs evaluation on QA datasets using different method types:
1. No retrieval: Direct LLM generation
2. Sequential: Retrieve first, then generate (e.g., single_retrieval, generate_then_retrieve)
3. Interleaved: Retrieval and generation interleaved (e.g., ReAct, SelfAsk, SearchO1)

Usage:
    # No retrieval (LLM only)
    python c4_task_evaluation/run_evalution.py --dataset qald10 --model openai/gpt-5.2 --method_type no_retrieval
    python c4_task_evaluation/run_evalution.py --dataset qald10 --model openai/gpt-4o --method_type no_retrieval

    # Test with limited samples (e.g., first 10)
    python c4_task_evaluation/run_evalution.py --dataset qald10 --model openai/gpt-4o --method_type no_retrieval --limit 10
    python c4_task_evaluation/run_evalution.py --dataset quest --model openai/gpt-4o --method_type no_retrieval --limit 10

    # Sequential retrieval + generation
    python c4_task_evaluation/run_evalution.py --dataset qald10 --model openai/gpt-4o --method_type sequential --method single_retrieval

    # Interleaved retrieval + generation
    python c4_task_evaluation/run_evalution.py --dataset qald10 --model openai/gpt-4o --method_type interleaved --method react
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add current directory to path for imports
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir.parent))

from utils.general_utils import set_seed
from c4_task_evaluation.methods.retrieval_augmented_models import NoRetrieval, SingleRetrieval, ReAct_Model, SelfAsk_Model, SearchO1_Model, ReSearch_Model, SearchR1_Model, StepSearch_Model
from c4_task_evaluation.metrics.generation_eval_metrics import soft_exact_match


def load_dataset(dataset_file):
    """Load dataset from JSONL file."""
    query_ids, test_dataset = [], {}
    if os.path.exists(dataset_file):
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                qid = data.get('qid', data.get('id'))
                if qid:
                    data['qid'] = qid
                    query_ids.append(qid)
                    test_dataset[qid] = data
    return query_ids, test_dataset


def get_existing_results(output_file):
    """Get already processed query IDs for resumption."""
    generated_qids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.add(data['qid'])
    return generated_qids


def initialize_model(args):
    """Initialize the model based on method type and method name."""
    print(f"\n=== Initializing Model: {args.method} ===")

    if args.method_type == 'no_retrieval':
        model = NoRetrieval(args.device, args)
    elif args.method_type == 'sequential':
        if args.method == 'single_retrieval':
            model = SingleRetrieval(args.device, args)
        else:
            raise NotImplementedError(f"Sequential method {args.method} not implemented")
    elif args.method_type == 'interleaved':
        if args.method == 'self_ask':
            model = SelfAsk_Model(args.device, args)
        elif args.method == 'react':
            model = ReAct_Model(args.device, args)
        elif args.method == 'search_o1':
            model = SearchO1_Model(args.device, args)
        elif args.method == 'research':
            model = ReSearch_Model(args.device, args)
        elif args.method == 'search_r1':
            model = SearchR1_Model(args.device, args)
        elif args.method == 'step_search':
            model = StepSearch_Model(args.device, args)
        else:
            raise NotImplementedError(f"Interleaved method {args.method} not implemented")
    else:
        raise ValueError(f"Unknown method_type: {args.method_type}")

    return model


def run_evaluation(args):
    """
    Main evaluation loop.

    For each query:
    1. Get prediction and ranked list (method-dependent)
    2. Evaluate against ground truth
    3. Save results
    """
    print("=" * 70)
    print("EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Dataset:     {args.dataset_name}")
    print(f"Model:       {args.model_name_or_path}")
    print(f"Method Type: {args.method_type}")
    print(f"Method:      {args.method}")
    print(f"Seed:        {args.seed}")
    print(f"Device:      {args.device}")
    print()

    # Load dataset
    print("=== Loading Dataset ===")
    query_ids, test_dataset = load_dataset(args.dataset_file)
    print(f"Total queries: {len(test_dataset)}")

    # Get existing results for resumption
    generated_qids = get_existing_results(args.output_file)
    filtered_dataset = {qid: data for qid, data in test_dataset.items() if qid not in generated_qids}
    print(f"Already processed: {len(generated_qids)}")
    print(f"Remaining: {len(filtered_dataset)}")

    # Initialize model
    model = initialize_model(args)

    # Main evaluation loop
    print("\n=== Starting Evaluation ===")
    all_results = []

    # Tolerance percentages for soft matching
    tolerance_percentages = [1.0, 5.0, 10.0, 20.0, 50.0, 90.0]

    # Counters for metrics
    exact_match_count = 0
    soft_match_counts = {pct: 0 for pct in tolerance_percentages}
    total_count = 0

    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        filtered_dataset_items = list(filtered_dataset.items())[:args.limit]
        print(f"Limiting to first {args.limit} samples")
    else:
        filtered_dataset_items = list(filtered_dataset.items())

    with open(args.output_file, 'a') as res_f:
        for i, (qid, sample) in enumerate(tqdm(filtered_dataset_items, desc="Processing queries")):

            # Extract query and ground truth
            query = sample.get('total_recall_query', sample.get('query', sample.get('question', '')))

            # Handle different answer formats
            if 'total_recall_answer' in sample:
                gt_answer = str(sample['total_recall_answer'])
            elif 'answer' in sample:
                answer = sample['answer']
                if isinstance(answer, dict):
                    gt_answer = str(answer.get('value', ''))
                else:
                    gt_answer = str(answer)
            else:
                gt_answer = ''

            # Get prediction based on method type
            if args.method_type == 'no_retrieval':
                # No retrieval - just generation
                reasoning_path, prediction = model.inference(query)
                ranked_list = []

            elif args.method_type == 'sequential':
                # Sequential: retrieve first, then generate
                # Model returns (reasoning_path, prediction)
                # We need to also track retrieved docs
                reasoning_path, prediction = model.inference(query)
                # TODO: Extract ranked_list from model (will be added when we refactor methods)
                ranked_list = []

            elif args.method_type == 'interleaved':
                # Interleaved: retrieval and generation happen together
                # Model returns (reasoning_path, prediction)
                # reasoning_path contains retrieval steps
                reasoning_path, prediction = model.inference(query)
                # TODO: Extract ranked_list from reasoning_path (will be added when we refactor methods)
                ranked_list = []

            # Compute soft exact match for each tolerance
            soft_matches = {}
            for pct in tolerance_percentages:
                match_result = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=pct)
                soft_matches[pct] = match_result['soft_match']
                if match_result['soft_match']:
                    soft_match_counts[pct] += 1

            # Exact match (no tolerance)
            exact_result = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=None)
            exact_match = exact_result['exact_match']
            if exact_match:
                exact_match_count += 1

            total_count += 1

            # Prepare result
            result_item = {
                "qid": qid,
                "query": query,
                "gt_answer": gt_answer,
                "prediction": prediction,
                "exact_match": exact_match,
                "soft_matches": soft_matches,
                "reasoning_path": reasoning_path,
                "ranked_list": ranked_list
            }

            # Add optional fields
            if 'file_id' in sample:
                result_item['file_id'] = sample['file_id']
            if 'original_query' in sample:
                result_item['original_query'] = sample['original_query']
            if 'aggregation_function' in sample:
                result_item['aggregation_function'] = sample['aggregation_function']

            all_results.append(result_item)

            # Write to file
            res_f.write(json.dumps(result_item) + '\n')
            res_f.flush()

    # Compute metrics summary
    print("\n=== Evaluation Summary ===")
    if all_results:
        print(f"Queries evaluated: {total_count}")

        exact_match_pct = (exact_match_count / total_count) * 100
        print(f"\nExact Match:       {exact_match_pct:.2f}%")

        print("\nSoft Exact Match (with tolerance):")
        for pct in sorted(tolerance_percentages):
            soft_pct = (soft_match_counts[pct] / total_count) * 100
            print(f"  {pct:5.1f}% tolerance: {soft_pct:.2f}%")

        # Save metrics summary
        metrics_summary = {
            "n": total_count,
            "exact_match": exact_match_count / total_count,
            "soft_exact_match": {pct: soft_match_counts[pct] / total_count for pct in tolerance_percentages}
        }

        metrics_file = args.output_file.replace('.jsonl', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
    else:
        print("No new queries processed.")

    print(f"\nResults saved to: {args.output_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Run evaluation pipeline on QA datasets')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['qald10', 'quest'], help='Dataset to evaluate on')
    parser.add_argument('--dataset_file', type=str, default=None, help='Path to dataset file (overrides default)')

    # Model arguments
    parser.add_argument('--model', type=str, default='openai/gpt-4o', help='Model to use for generation (default: openai/gpt-4o)')

    # Method type and method
    parser.add_argument('--method_type', type=str, default='no_retrieval', choices=['no_retrieval', 'sequential', 'interleaved'], help='Method type: no_retrieval, sequential, or interleaved (default: no_retrieval)')
    parser.add_argument('--method', type=str, default='no_retrieval', choices=['no_retrieval', 'single_retrieval', 'self_ask', 'react', 'search_o1', 'research', 'search_r1', 'step_search'], help='Specific method to use (default: no_retrieval)')

    # Retriever arguments
    parser.add_argument('--retriever', type=str, default='contriever', choices=['bm25', 'rerank_l6', 'rerank_l12', 'contriever', 'dpr', 'e5', 'bge'], help='Retriever to use (default: contriever)')
    parser.add_argument('--index_dir', type=str, default='/projects/0/prjs0834/heydars/INDICES', help='Directory containing retrieval indices')
    parser.add_argument('--corpus_path', type=str, default='corpus_datasets/enwiki_20251001.jsonl', help='Path to corpus file')
    parser.add_argument('--retrieval_topk', type=int, default=3, help='Number of documents to retrieve (default: 3)')
    parser.add_argument('--faiss_gpu', action='store_true', default=False, help='Use GPU for FAISS computation')

    # Other arguments
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum iterations for multi-step models (default: 10)')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--run', type=str, default='run_3', help='Run identifier (default: run_1)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples to process (for testing, default: None = process all)')

    args = parser.parse_args()

    # Set dataset_name for compatibility
    args.dataset_name = args.dataset

    # Set dataset file path if not provided
    if args.dataset_file is None:
        if args.dataset_name == 'qald10':
            args.dataset_file = "corpus_datasets/dataset_creation_heydar/qald10/qald10_queries.jsonl"
        elif args.dataset_name == 'quest':
            args.dataset_file = "corpus_datasets/dataset_creation_heydar/quest/test_quest_queries.jsonl"

    # Set model_name_or_path for compatibility
    args.model_name_or_path = args.model

    # Set generation_model for compatibility with existing model classes
    args.generation_model = args.method

    # Set retriever_name for compatibility
    args.retriever_name = args.retriever

    # Set model source
    if args.model_name_or_path in ["openai/gpt-4o", "openai/gpt-5.2", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash", "deepseek/deepseek-chat-v3-0324", "qwen/qwen3-235b-a22b-2507"]:
        args.model_source = 'api'
    else:
        args.model_source = 'hf_local'

    # Set device
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(args.device)}")

    # Set output directory and file
    if args.output_dir is None:
        model_short = args.model_name_or_path.split('/')[-1]
        if args.method_type == 'no_retrieval':
            args.output_dir = f"run_output/{args.run}/{model_short}/{args.dataset_name}/{args.method}"
        else:
            args.output_dir = f"run_output/{args.run}/{model_short}/{args.dataset_name}/{args.method}_{args.retriever_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = f"{args.output_dir}/evaluation_results.jsonl"

    # Additional retriever args for compatibility
    args.retrieval_query_max_length = 64
    args.retrieval_use_fp16 = True
    args.retrieval_batch_size = 512
    args.bm25_k1 = 0.9
    args.bm25_b = 0.4

    # Set seed
    set_seed(args.seed)

    # Check if dataset file exists
    if not Path(args.dataset_file).exists():
        print(f"Error: Dataset file not found: {args.dataset_file}")
        return 1

    # Run evaluation
    try:
        return run_evaluation(args)
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
