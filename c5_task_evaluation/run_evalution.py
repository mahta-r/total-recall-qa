#!/usr/bin/env python3
"""
Evaluation Pipeline Runner

This script runs evaluation on QA datasets using two pipelines:
1. Retrieval Pipeline: Evaluate retrieval only using entity recall (only Retriever arguments are used)
2. Generation Pipeline: Evaluate retrieval + generation (or generation only)

Pipeline Types:
- retrieval: Only retrieve, evaluate entity recall (no LLM). Use only Retriever arguments.
- generation: Full RAG evaluation with LLM. Use generation_method (and deep_research_model when applicable).

Generation methods (valid only when pipeline=generation):
- no_retrieval: Direct LLM generation, no retrieval
- single_retrieval: Retrieve once, then generate
- deep_research: Interleaved retrieval + generation; requires --deep_research_model

Deep research models (when generation_method=deep_research):
  self_ask, react, search_o1, research, search_r1, step_search

Usage:
    # RETRIEVAL PIPELINE - retrieval_eval_ks defines k values for entity recall@k; we retrieve max(eval_ks) (retrieval_topk is not used)
    python c5_task_evaluation/run_evalution.py --pipeline retrieval --dataset heydar --subset test --retriever bm25 --retrieval_eval_ks 1 3 10 100
    python c5_task_evaluation/run_evalution.py --pipeline retrieval --dataset heydar --subset val --retriever contriever

    # GENERATION PIPELINE
    # No retrieval (LLM only)
    python c5_task_evaluation/run_evalution.py --pipeline generation --dataset heydar --subset val --model openai/gpt-4o --generation_method no_retrieval

    # Single retrieval + generation
    python c5_task_evaluation/run_evalution.py --pipeline generation --dataset heydar --subset test --model openai/gpt-4o --generation_method single_retrieval --retriever contriever --retrieval_topk 5

    # Deep research (interleaved); specify which model
    python c5_task_evaluation/run_evalution.py --pipeline generation --dataset heydar --subset val --model openai/gpt-4o --generation_method deep_research --deep_research_model react --retriever contriever
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root and this package to path so imports work from any cwd
_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent
sys.path.insert(0, str(_project_root))

from utils.general_utils import set_seed
from c5_task_evaluation.methods.retrieval_augmented_models import NoRetrieval, SingleRetrieval, ReAct_Model, SelfAsk_Model, SearchO1_Model, ReSearch_Model, SearchR1_Model, StepSearch_Model
from c5_task_evaluation.metrics.generation_eval_metrics import soft_exact_match
from c5_task_evaluation.metrics.retrieval_eval_metrics import (
    load_qrels,
    evaluate_entity_retrieval,
    entity_recall_at_k,
)


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


def _get_soft_match_value(record, pct):
    """Fetch stored soft match value for a tolerance pct from a result record."""
    soft_matches = record.get('soft_matches')
    if not isinstance(soft_matches, dict):
        return None

    key_variants = [pct, str(pct)]
    if isinstance(pct, float) and pct.is_integer():
        key_variants.extend([int(pct), str(int(pct))])

    for key in key_variants:
        if key in soft_matches:
            return soft_matches[key]
    return None


def aggregate_generation_metrics_from_file(output_file, tolerance_percentages):
    """
    Aggregate generation metrics across every result stored in output_file.
    This allows resumed runs to report metrics over both past and newly processed queries.
    """
    if not os.path.exists(output_file):
        return None

    exact_match_count = 0
    soft_match_counts = {pct: 0 for pct in tolerance_percentages}
    total_count = 0

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            total_count += 1

            prediction = str(data.get('prediction', ''))
            gt_answer = str(data.get('gt_answer', ''))

            exact_match = data.get('exact_match')
            if exact_match is None:
                exact_match = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=None)['exact_match']
            if exact_match:
                exact_match_count += 1

            for pct in tolerance_percentages:
                soft_match = _get_soft_match_value(data, pct)
                if soft_match is None:
                    soft_match = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=pct)['soft_match']
                if soft_match:
                    soft_match_counts[pct] += 1

    if total_count == 0:
        return None

    return {
        "n": total_count,
        "exact_match_count": exact_match_count,
        "soft_match_counts": soft_match_counts,
        "exact_match": exact_match_count / total_count,
        "soft_exact_match": {pct: soft_match_counts[pct] / total_count for pct in tolerance_percentages},
    }


def aggregate_retrieval_metrics_from_file(output_file, k_values):
    """
    Aggregate retrieval metrics across every stored result, ensuring resumed runs
    report metrics over the union of processed queries.
    """
    if not os.path.exists(output_file):
        return None

    entity_recall_sums = {k: 0.0 for k in k_values}
    entity_recall_sum_all = 0.0
    total_count = 0

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            total_count += 1
            entity_recall_sum_all += float(data.get('entity_recall', 0.0))
            for k in k_values:
                entity_recall_sums[k] += float(data.get(f'entity_recall@{k}', 0.0))

    if total_count == 0:
        return None

    return {
        "n": total_count,
        "entity_recall": entity_recall_sum_all / total_count,
        "entity_recall_by_k": {k: entity_recall_sums[k] / total_count for k in k_values},
    }


def initialize_model(args):
    """Initialize the model based on generation_method and (for deep_research) deep_research_model."""
    print(f"\n=== Initializing Model: {args.generation_method}" + (f" ({args.deep_research_model})" if args.generation_method == 'deep_research' else "") + " ===")

    if args.generation_method == 'no_retrieval':
        model = NoRetrieval(args.device, args)
    elif args.generation_method == 'single_retrieval':
        model = SingleRetrieval(args.device, args)
    elif args.generation_method == 'deep_research':
        dm = args.deep_research_model
        if dm == 'self_ask':
            model = SelfAsk_Model(args.device, args)
        elif dm == 'react':
            model = ReAct_Model(args.device, args)
        elif dm == 'search_o1':
            model = SearchO1_Model(args.device, args)
        elif dm == 'research':
            model = ReSearch_Model(args.device, args)
        elif dm == 'search_r1':
            model = SearchR1_Model(args.device, args)
        elif dm == 'step_search':
            model = StepSearch_Model(args.device, args)
        else:
            raise NotImplementedError(f"Deep research model {dm} not implemented")
    else:
        raise ValueError(f"Unknown generation_method: {args.generation_method}")

    return model


def run_generation_evaluation(args):
    """
    Generation evaluation pipeline.

    For each query:
    1. Get prediction and ranked list (method-dependent)
    2. Evaluate against ground truth using exact/soft match
    3. Save results
    """
    print("=" * 70)
    print("GENERATION EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Dataset:     {args.dataset_name}")
    print(f"Subset:      {args.subset_name}")
    print(f"Model:       {args.model_name_or_path}")
    print(f"Generation method: {args.generation_method}" + (f" ({args.deep_research_model})" if args.generation_method == 'deep_research' else ""))
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
    # Tolerance percentages for soft matching
    tolerance_percentages = [1.0, 5.0, 10.0, 20.0, 50.0, 90.0]

    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        filtered_dataset_items = list(filtered_dataset.items())[:args.limit]
        print(f"Limiting to first {args.limit} samples")
    else:
        filtered_dataset_items = list(filtered_dataset.items())

    new_queries_processed = 0

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

            # Get prediction based on generation method
            if args.generation_method == 'no_retrieval':
                reasoning_path, prediction = model.inference(query)
                ranked_list = []
            elif args.generation_method == 'single_retrieval':
                reasoning_path, prediction = model.inference(query)
                ranked_list = []
            elif args.generation_method == 'deep_research':
                reasoning_path, prediction = model.inference(query)
                ranked_list = []

            # Compute soft exact match for each tolerance
            soft_matches = {}
            for pct in tolerance_percentages:
                match_result = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=pct)
                soft_matches[pct] = match_result['soft_match']

            # Exact match (no tolerance)
            exact_result = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=None)
            exact_match = exact_result['exact_match']
            new_queries_processed += 1

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

            # Write to file
            res_f.write(json.dumps(result_item) + '\n')
            res_f.flush()

    if new_queries_processed == 0:
        print("\nNo new queries processed in this run.")
    else:
        print(f"\nNew queries processed in this run: {new_queries_processed}")

    # Compute metrics summary aggregated over all stored results
    aggregated_metrics = aggregate_generation_metrics_from_file(args.output_file, tolerance_percentages)
    print("\n=== Evaluation Summary (aggregated) ===")
    if aggregated_metrics:
        print(f"Total queries evaluated so far: {aggregated_metrics['n']}")

        exact_match_pct = aggregated_metrics['exact_match'] * 100
        print(f"\nExact Match:       {exact_match_pct:.2f}%")

        print("\nSoft Exact Match (with tolerance):")
        for pct in sorted(tolerance_percentages):
            soft_pct = aggregated_metrics['soft_exact_match'][pct] * 100
            print(f"  {pct:5.1f}% tolerance: {soft_pct:.2f}%")

        metrics_summary = {
            "n": aggregated_metrics['n'],
            "exact_match": aggregated_metrics['exact_match'],
            "soft_exact_match": aggregated_metrics['soft_exact_match']
        }

        metrics_file = args.output_file.replace('.jsonl', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
    else:
        print("No evaluation results available yet.")

    print(f"\nResults saved to: {args.output_file}")
    return 0


def _get_passage_id(doc):
    """
    Get passage/document id from a retrieved doc dict.
    Retrievers may use different keys: DenseRetriever returns corpus rows with 'id';
    BM25 with stored index may return only 'title', 'text', 'contents'.
    """
    for key in ('id', 'doc_id', 'passage_id', 'docid'):
        val = doc.get(key)
        if val is not None:
            return str(val)
    # BM25 with contain_doc=True builds only title/text/contents; title may be passage id in some indices
    if 'title' in doc:
        return str(doc['title']).strip()
    raise KeyError("Retrieved doc has no id field. Keys: %s" % list(doc.keys()))


def initialize_retriever(args):
    """Initialize retriever for retrieval-only evaluation."""
    from c5_task_evaluation.src.retrieval_models_local import BM25Retriever, RerankRetriever, DenseRetriever, SPLADERetriever
    
    print(f"\n=== Initializing Retriever: {args.retriever_name} ===")
    
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)
    elif args.retriever_name == 'spladepp':
        retriever = SPLADERetriever(args)
    elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
        retriever = DenseRetriever(args)
    else:
        raise ValueError(f"Unknown retriever: {args.retriever_name}")
    
    return retriever


def run_retrieval_evaluation(args):
    """
    Retrieval evaluation pipeline.
    
    Evaluates retrieval performance using entity recall:
    - For each query, retrieve top-k passages
    - Compare retrieved passage entities against gold entities from qrels
    - Entity recall = |retrieved_entities ∩ gold_entities| / |gold_entities|
    
    Multiple passages from the same entity only count as covering that entity once.
    """
    print("=" * 70)
    print("RETRIEVAL EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Dataset:     {args.dataset_name}")
    print(f"Subset:      {args.subset_name}")
    print(f"Retriever:   {args.retriever_name}")
    print(f"Eval k's:    {args.retrieval_eval_ks} (retrieving top {args.retrieval_topk} for evaluation)")
    print(f"Qrel File:   {args.qrel_file}")
    print(f"Seed:        {args.seed}")
    print()
    
    # Load qrels
    print("=== Loading QRels ===")
    if not Path(args.qrel_file).exists():
        print(f"Error: Qrel file not found: {args.qrel_file}")
        return 1
    
    qrels = load_qrels(args.qrel_file)
    print(f"Loaded qrels for {len(qrels)} queries")
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    query_ids, test_dataset = load_dataset(args.dataset_file)
    print(f"Total queries in dataset: {len(test_dataset)}")
    
    # Filter to queries that have qrels
    queries_with_qrels = {qid: data for qid, data in test_dataset.items() if qid in qrels}
    print(f"Queries with qrels: {len(queries_with_qrels)}")
    
    # Get existing results for resumption
    generated_qids = get_existing_results(args.output_file)
    filtered_dataset = {qid: data for qid, data in queries_with_qrels.items() if qid not in generated_qids}
    print(f"Already processed: {len(generated_qids)}")
    print(f"Remaining: {len(filtered_dataset)}")
    
    # Initialize retriever
    retriever = initialize_retriever(args)
    
    # Main evaluation loop
    print("\n=== Starting Retrieval Evaluation ===")
    # K values for entity recall (retrieval pipeline always retrieves max(eval_ks), so all requested k's are computed)
    k_values = sorted(set(args.retrieval_eval_ks))
    
    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        filtered_dataset_items = list(filtered_dataset.items())[:args.limit]
        print(f"Limiting to first {args.limit} samples")
    else:
        filtered_dataset_items = list(filtered_dataset.items())
    
    new_queries_processed = 0

    with open(args.output_file, 'a') as res_f:
        for i, (qid, sample) in enumerate(tqdm(filtered_dataset_items, desc="Processing queries")):
            
            # Extract query
            query = sample.get('total_recall_query', sample.get('query', sample.get('question', '')))
            
            # Get gold passage IDs from qrels
            gold_passage_ids = qrels.get(qid, [])
            
            # Retrieve passages
            retrieved_docs = retriever.search(query)
            retrieved_ids = [_get_passage_id(doc) for doc in retrieved_docs]
            
            # Compute entity recall metrics
            metrics = evaluate_entity_retrieval(retrieved_ids, gold_passage_ids, k_values)
            
            new_queries_processed += 1
            
            # Prepare result
            result_item = {
                "qid": qid,
                "query": query,
                "num_gold_passages": len(gold_passage_ids),
                "num_gold_entities": metrics['num_gold_entities'],
                "num_retrieved_passages": len(retrieved_ids),
                "num_retrieved_entities": metrics['num_retrieved_entities'],
                "retrieved_ids": retrieved_ids,
                "entity_recall": metrics['entity_recall'],
            }
            
            # Add entity recall at different k
            for k in k_values:
                result_item[f'entity_recall@{k}'] = metrics[f'entity_recall@{k}']
            
            # Add optional fields
            if 'file_id' in sample:
                result_item['file_id'] = sample['file_id']
            if 'src' in sample:
                result_item['src'] = sample['src']
            
            # Write to file
            res_f.write(json.dumps(result_item) + '\n')
            res_f.flush()
    
    if new_queries_processed == 0:
        print("\nNo new queries processed in this run.")
    else:
        print(f"\nNew queries processed in this run: {new_queries_processed}")

    # Compute metrics summary aggregated over all stored results
    aggregated_metrics = aggregate_retrieval_metrics_from_file(args.output_file, k_values)
    print("\n=== Retrieval Evaluation Summary (aggregated) ===")
    if aggregated_metrics:
        print(f"Total queries evaluated so far: {aggregated_metrics['n']}")

        print("\nEntity Recall@k:")
        metrics_summary = {
            "n": aggregated_metrics['n'],
            "retriever": args.retriever_name,
            "retrieval_eval_ks": args.retrieval_eval_ks,
            "retrieval_eval_max_k": args.retrieval_topk,
        }

        for k in k_values:
            mean_recall = aggregated_metrics['entity_recall_by_k'][k]
            metrics_summary[f'entity_recall@{k}'] = mean_recall
            print(f"  Entity Recall@{k:3d}: {mean_recall:.4f} ({mean_recall*100:.2f}%)")

        mean_recall_all = aggregated_metrics['entity_recall']
        metrics_summary['entity_recall'] = mean_recall_all
        print(f"\n  Entity Recall (all): {mean_recall_all:.4f} ({mean_recall_all*100:.2f}%)")

        metrics_file = args.output_file.replace('.jsonl', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
    else:
        print("No evaluation results available yet.")
    
    print(f"\nResults saved to: {args.output_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Run evaluation pipeline on QA datasets')

    # Pipeline selection
    parser.add_argument('--pipeline', type=str, default='retrieval', choices=['retrieval', 'generation'], help='Pipeline type: retrieval (entity recall only) or generation (full RAG with LLM)')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default="heydar", choices=['heydar', 'mahta', 'zahra'], help='Dataset to evaluate on (heydar, mahta, or zahra)')
    parser.add_argument('--subset', type=str, default='test', choices=['val', 'test'], help='Subset: val or test (for heydar/mahta/zahra)')
    parser.add_argument('--dataset_file', type=str, default=None, help='Path to dataset file (overrides default)')
    parser.add_argument('--qrel_file', type=str, default=None, help='Path to qrel file (for retrieval pipeline, overrides default)')

    # Generation arguments (only used when pipeline=generation)
    parser.add_argument('--model', type=str, default='openai/gpt-4o', help='Model for generation (default: openai/gpt-4o). Used only when pipeline=generation.')
    parser.add_argument('--generation_method', type=str, default='no_retrieval', choices=['no_retrieval', 'single_retrieval', 'deep_research'], help='Generation method: no_retrieval, single_retrieval, or deep_research (default: no_retrieval). Used only when pipeline=generation.')
    parser.add_argument('--deep_research_model', type=str, default='react', choices=['self_ask', 'react', 'search_o1', 'research', 'search_r1', 'step_search'], help='Deep research model when generation_method=deep_research (default: react). Used only when pipeline=generation and generation_method=deep_research.')

    # Retriever arguments (used for pipeline=retrieval; also for pipeline=generation when generation_method is single_retrieval or deep_research)
    parser.add_argument('--retriever', type=str, default='e5', choices=['bm25', 'spladepp', 'rerank_l6', 'rerank_l12', 'contriever', 'dpr', 'e5', 'bge'], help='Retriever to use (default: contriever)')
    parser.add_argument('--index_dir', type=str, default='/projects/0/prjs0834/heydars/CORPUS_Mahta/indices', help='Directory containing retrieval indices')
    parser.add_argument('--corpus_path', type=str, default='corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl', help='Path to corpus file')
    parser.add_argument('--retrieval_topk', type=int, default=3, help='Number of passages passed to the LLM when using single_retrieval or deep_research (generation pipeline only). Not used for retrieval pipeline (default: 3)')
    parser.add_argument('--retrieval_eval_ks', type=int, nargs='+', default=[1, 3, 10, 100], help='K values for retrieval evaluation (entity recall@k). Retrieval pipeline retrieves max(eval_ks) and reports these k’s (default: 1 3 10 100)')
    parser.add_argument('--faiss_gpu', action='store_true', default=False, help='Use GPU for FAISS computation')

    # Other arguments
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum iterations for multi-step models (default: 10)')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--run', type=str, default='run_1', help='Run identifier (default: run_1)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples to process (for testing, default: None = process all)')

    args = parser.parse_args()

    # Set dataset_name for compatibility
    args.dataset_name = args.dataset

    # Subset name for display and output paths (e.g., val_heydar, test_heydar)
    args.subset_name = f"{args.subset}_{args.dataset}"

    # Set dataset file path if not provided
    # heydar/mahta/zahra: val -> queries_validation_final.jsonl, test -> queries_test_final.jsonl
    if args.dataset_file is None:
        if args.subset == "val":
            query_filename = "queries_validation_final.jsonl"
        else:
            query_filename = "queries_test_final.jsonl"
        args.dataset_file = f"corpus_datasets/dataset_creation_{args.dataset}/{query_filename}"

    # Set qrel file path if not provided (for retrieval pipeline)
    if args.qrel_file is None:
        if args.subset == "val":
            qrel_filename = "qrels_validation_final.txt"
        else:
            qrel_filename = "qrels_test_final.txt"
        args.qrel_file = f"corpus_datasets/dataset_creation_{args.dataset}/{qrel_filename}"

    # Set model_name_or_path for compatibility
    args.model_name_or_path = args.model

    # Set generation_model for compatibility with model classes (no_retrieval, single_retrieval, or deep_research model name)
    if args.pipeline == 'generation':
        if args.generation_method == 'deep_research':
            args.generation_model = args.deep_research_model
        else:
            args.generation_model = args.generation_method
    else:
        args.generation_model = None  # not used for retrieval pipeline

    # Set retriever_name for compatibility
    args.retriever_name = args.retriever

    # Retrieval pipeline: retrieval_topk is only for generation (LLM). For retrieval eval we retrieve max(eval_ks).
    if args.pipeline == 'retrieval':
        args.retrieval_topk = max(args.retrieval_eval_ks)

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
        if args.pipeline == 'retrieval':
            # Retrieval pipeline: run_output/{run}/retrieval/{subset_name}/{retriever}_topk{k}
            args.output_dir = f"run_output/{args.run}/{args.subset_name}/retrieval_{args.retriever_name}"
        else:
            # Generation pipeline: run_output/{run}/{model}/{subset_name}/{generation_method}[_{retriever}]
            model_short = args.model_name_or_path.split('/')[-1]
            if args.generation_method == 'no_retrieval':
                args.output_dir = f"run_output/{args.run}/{args.subset_name}/{model_short}_{args.generation_method}"
            elif args.generation_method == 'deep_research':
                args.output_dir = f"run_output/{args.run}/{args.subset_name}/{model_short}_{args.generation_method}_{args.deep_research_model}_{args.retriever_name}"
            else:
                args.output_dir = f"run_output/{args.run}/{args.subset_name}/{model_short}_{args.generation_method}_{args.retriever_name}"

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

    # Check if qrel file exists (for retrieval pipeline)
    if args.pipeline == 'retrieval':
        if not Path(args.qrel_file).exists():
            print(f"Error: Qrel file not found: {args.qrel_file}")
            return 1

    # For generation pipeline, generation_method is required (and deep_research_model when generation_method=deep_research)
    if args.pipeline == 'generation':
        if args.generation_method not in ('no_retrieval', 'single_retrieval', 'deep_research'):
            print(f"Error: generation_method must be one of no_retrieval, single_retrieval, deep_research (got {args.generation_method})")
            return 1
        if args.generation_method == 'deep_research' and args.deep_research_model not in ('self_ask', 'react', 'search_o1', 'research', 'search_r1', 'step_search'):
            print(f"Error: deep_research_model must be one of self_ask, react, search_o1, research, search_r1, step_search (got {args.deep_research_model})")
            return 1

    # Run evaluation based on pipeline type
    try:
        if args.pipeline == 'retrieval':
            return run_retrieval_evaluation(args)
        else:
            return run_generation_evaluation(args)
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# --- RETRIEVAL PIPELINE (only Retriever arguments) ---
# python c5_task_evaluation/run_evalution.py --pipeline retrieval --dataset heydar --subset val --retriever bm25 --retrieval_topk 100 --limit 5
# python c5_task_evaluation/run_evalution.py --pipeline retrieval --dataset heydar --subset val --retriever contriever --retrieval_topk 50

# --- GENERATION PIPELINE ---
# python c5_task_evaluation/run_evalution.py --pipeline generation --dataset heydar --subset test --model openai/gpt-4o --generation_method no_retrieval --limit 10
# python c5_task_evaluation/run_evalution.py --pipeline generation --dataset heydar --subset test --model openai/gpt-4o --generation_method single_retrieval --retriever contriever --retrieval_topk 5
# python c5_task_evaluation/run_evalution.py --pipeline generation --dataset heydar --subset val --model openai/gpt-4o --generation_method deep_research --deep_research_model react --retriever contriever