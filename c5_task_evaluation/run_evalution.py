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

Multi-GPU: use --devices "0,1,2,3" to split queries across GPUs (data parallelism) for both pipelines. On CPU, use --num_workers N for parallel retrieval (BM25) or parallel generation (e.g. API-based models).

Usage:
    # RETRIEVAL PIPELINE - retrieval_eval_ks defines k values for entity recall@k; we retrieve max(eval_ks) (retrieval_topk is not used)
    python c5_task_evaluation/run_evalution.py --pipeline retrieval --dataset wikidata --subset test --retriever bm25 --retrieval_eval_ks 1 3 10 100
    python c5_task_evaluation/run_evalution.py --pipeline retrieval --dataset qald10_quest --subset val --retriever contriever --devices "0,1"

    # GENERATION PIPELINE
    # No retrieval (LLM only)
    python c5_task_evaluation/run_evalution.py --pipeline generation --dataset qald10_quest --subset val --model openai/gpt-4o --generation_method no_retrieval

    # Single retrieval + generation
    python c5_task_evaluation/run_evalution.py --pipeline generation --dataset wikidata --subset test --model openai/gpt-4o --generation_method single_retrieval --retriever contriever --retrieval_topk 5

    # Deep research (interleaved); specify which model
    python c5_task_evaluation/run_evalution.py --pipeline generation --dataset qald10_quest --subset val --model openai/gpt-4o --generation_method deep_research --deep_research_model react --retriever contriever

Generation output layout (run_output/{run}/{dataset_subset}/):
  - no_retrieval:       generation_{model}_no_retrieval/
  - single_retrieval:   generation_{model}_single_retrieval_{retriever}/
  - deep_research:      generation_{model}_deep_research_{deep_research_model}/

Generation writes: evaluation_results_per_query_metrics.jsonl (per-query + metrics for significance_test)
and evaluation_results_metrics.json (aggregated). Deep research runs also store reasoning_trajectory per query.
"""

import os
import sys
import json
import time
import torch
import argparse
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Add project root and this package to path so imports work from any cwd
_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent
sys.path.insert(0, str(_project_root))

from utils.general_utils import set_seed
from c5_task_evaluation.methods.retrieval_augmented_models import NoRetrieval, SingleRetrieval, ReAct_Model, SelfAsk_Model, SearchO1_Model, ReSearch_Model, SearchR1_Model, StepSearch_Model
from c5_task_evaluation.metrics.generation_eval_metrics import (
    soft_exact_match,
    clean_prediction_for_numeric,
    clean_gold_for_numeric,
)
from c5_task_evaluation.metrics.retrieval_eval_metrics import (
    load_qrels,
    evaluate_entity_retrieval,
    entity_recall_at_k,
)


def _format_duration(seconds):
    """Format seconds as human-readable duration (e.g. 2m 30s, 1h 5m)."""
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _poll_worker_progress(progress_dir, progress_prefix, worker_label, worker_ids, started_ranks):
    """
    Read progress files from workers. Files contain: current total start_ts.
    Returns (total_done, total_items, elapsed_sec, eta_sec, per_worker_parts).
    """
    total_done = 0
    total_items = 0
    start_ts_min = None
    parts = []
    for rank in started_ranks:
        pf = progress_dir / f"{progress_prefix}_{rank}"
        if not pf.exists():
            continue
        try:
            line = pf.read_text().strip().split()
            if len(line) >= 2:
                cur = int(line[0])
                tot = int(line[1])
                total_done += cur
                total_items += tot
                parts.append(f"{worker_label} {worker_ids[rank]}: {cur}/{tot}")
                if len(line) >= 3:
                    try:
                        st = int(line[2])
                        if start_ts_min is None or st < start_ts_min:
                            start_ts_min = st
                    except ValueError:
                        pass
        except Exception:
            pass
    elapsed_sec = (time.time() - start_ts_min) if start_ts_min else 0
    eta_sec = None
    if total_items > 0 and total_done > 0 and elapsed_sec > 0 and total_done < total_items:
        rate = total_done / elapsed_sec
        eta_sec = (total_items - total_done) / rate
    return total_done, total_items, elapsed_sec, eta_sec, parts


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

    # Load qrels when using oracle retriever (needed to look up gold passages)
    if args.retriever_name == 'oracle':
        if not Path(args.qrel_file).exists():
            print(f"Error: Qrel file required for oracle retriever. Not found: {args.qrel_file}")
            return 1
        print("=== Loading QRels (for oracle retriever) ===")
        args.qrels = load_qrels(args.qrel_file)
        print(f"Loaded qrels for {len(args.qrels)} queries")
    else:
        args.qrels = getattr(args, 'qrels', None)

    # Load dataset
    print("\n=== Loading Dataset ===")
    query_ids, test_dataset = load_dataset(args.dataset_file)
    print(f"Total queries: {len(test_dataset)}")

    # Get existing results for resumption
    generated_qids = get_existing_results(args.output_file)
    filtered_dataset = {qid: data for qid, data in test_dataset.items() if qid not in generated_qids}
    print(f"Already processed: {len(generated_qids)}")
    print(f"Remaining: {len(filtered_dataset)}")

    # Tolerance percentages for soft matching
    tolerance_percentages = [1.0, 5.0, 10.0, 20.0, 50.0, 90.0]

    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        filtered_dataset_items = list(filtered_dataset.items())[:args.limit]
        print(f"Limiting to first {args.limit} samples")
    else:
        filtered_dataset_items = list(filtered_dataset.items())

    device_ids = getattr(args, 'device_ids', [0])
    num_workers = getattr(args, 'num_workers', None) or 0

    # Multi-GPU: split work across GPUs and spawn workers
    if len(device_ids) > 1:
        print(f"\n=== Multi-GPU Generation: {len(device_ids)} devices ===")
        worker_ids = device_ids
        worker_count = len(device_ids)
        kind = "GPU"
    # Multi-CPU: split work across CPU workers (e.g. for API-based models)
    elif num_workers > 1:
        print(f"\n=== Multi-CPU Generation: {num_workers} workers ===")
        worker_ids = list(range(num_workers))
        worker_count = num_workers
        kind = "Worker"
    else:
        worker_count = 0

    if worker_count > 1:
        chunks = [[] for _ in worker_ids]
        for i, item in enumerate(filtered_dataset_items):
            chunks[i % len(worker_ids)].append(item)
        procs = []
        started_ranks = []
        for rank, (wid, chunk) in enumerate(zip(worker_ids, chunks)):
            if not chunk:
                continue
            # For CPU workers, pass rank as device_id; _generation_worker uses cpu when cuda not available
            device_id = wid if kind == "GPU" else 0
            p = multiprocessing.Process(
                target=_generation_worker,
                args=(rank, device_id, args, chunk),
            )
            p.start()
            procs.append(p)
            started_ranks.append(rank)
        progress_dir = Path(args.output_file).parent
        last_print_time = 0
        last_total_done = -1
        use_cr = sys.stdout.isatty()
        while any(p.is_alive() for p in procs):
            time.sleep(5)
            total_done, total_items, elapsed_sec, eta_sec, parts = _poll_worker_progress(
                progress_dir, ".progress_gen", kind, worker_ids, started_ranks
            )
            if not parts:
                continue
            now = time.time()
            if total_done <= last_total_done and (now - last_print_time) < 10:
                continue
            last_total_done = total_done
            last_print_time = now
            pct = 100.0 * total_done / total_items if total_items else 0
            elapsed_str = _format_duration(elapsed_sec)
            eta_str = _format_duration(eta_sec) if eta_sec is not None else "?"
            line = f"Generation: {total_done}/{total_items} ({pct:.1f}%) | elapsed {elapsed_str} | ETA ~{eta_str} | " + " | ".join(parts)
            if use_cr:
                print("\r" + line, end="", flush=True)
            else:
                print(line, flush=True)
        if use_cr:
            print(flush=True)
        for p in procs:
            p.join()
        for rank in started_ranks:
            try:
                (progress_dir / f".progress_gen_{rank}").unlink(missing_ok=True)
            except Exception:
                pass
        _merge_rank_files(args.output_file, worker_count, append=True)
        new_queries_processed = len(filtered_dataset_items)
    else:
        # Single-GPU path
        model = initialize_model(args)
        print("\n=== Starting Evaluation ===")
        new_queries_processed = 0
        n_single = len(filtered_dataset_items)
        use_tqdm = sys.stdout.isatty()
        start_time_single = time.time()
        with open(args.output_file, 'a') as res_f:
            for i, (qid, sample) in enumerate(tqdm(filtered_dataset_items, desc="Generation", position=0, leave=True, disable=not use_tqdm, mininterval=5, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")):
                if not use_tqdm:
                    current = i + 1
                    if current % 50 == 0 or current == 1 or current == n_single:
                        elapsed = time.time() - start_time_single
                        eta = (n_single - current) * (elapsed / current) if current > 0 else 0
                        print(f"Generation: {current}/{n_single} ({100*current/n_single:.1f}%) | elapsed {_format_duration(elapsed)} | ETA ~{_format_duration(eta)}", flush=True)
                query = sample.get('total_recall_query', sample.get('query', sample.get('question', '')))
                if 'total_recall_answer' in sample:
                    gt_answer = str(sample['total_recall_answer'])
                elif 'answer' in sample:
                    answer = sample['answer']
                    gt_answer = str(answer.get('value', '')) if isinstance(answer, dict) else str(answer)
                else:
                    gt_answer = ''
                if args.generation_method == 'no_retrieval':
                    reasoning_path, prediction = model.inference(query)
                elif args.generation_method == 'single_retrieval':
                    reasoning_path, prediction = model.inference(query, qid=qid)
                else:
                    reasoning_path, prediction = model.inference(query, qid=qid)
                soft_matches = {}
                for pct in tolerance_percentages:
                    match_result = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=pct)
                    soft_matches[pct] = match_result['soft_match']
                exact_result = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=None)
                exact_match = exact_result['exact_match']
                new_queries_processed += 1
                gt_answer_cleaned = clean_gold_for_numeric(gt_answer, decimals=2)
                prediction_cleaned = clean_prediction_for_numeric(prediction)
                path_for_output = _reasoning_path_doc_ids_only(reasoning_path, _get_passage_id) if args.generation_method in ('single_retrieval', 'deep_research') else reasoning_path
                result_item = {
                    "qid": qid,
                    "query": query,
                    "gt_answer": gt_answer,
                    "gt_answer_cleaned": gt_answer_cleaned,
                    "prediction": prediction,
                    "prediction_cleaned": prediction_cleaned,
                    "exact_match": float(1 if exact_match else 0),
                    "soft_matches": soft_matches,
                    "reasoning_path": path_for_output,
                }
                if args.generation_method == 'deep_research':
                    result_item["reasoning_trajectory"] = path_for_output
                if 'file_id' in sample:
                    result_item['file_id'] = sample['file_id']
                if 'original_query' in sample:
                    result_item['original_query'] = sample['original_query']
                if 'aggregation_function' in sample:
                    result_item['aggregation_function'] = sample['aggregation_function']
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

        with open(args.metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nMetrics saved to: {args.metrics_file}")
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


def _reasoning_path_doc_ids_only(reasoning_path, get_passage_id_fn):
    """
    Return a copy of reasoning_path where each step's 'docs' list is replaced by
    a list of passage ids only (no document content). Used for single_retrieval
    and deep_research per-query output to avoid storing full doc content.
    """
    out = []
    for step in reasoning_path:
        step_copy = dict(step)
        if 'docs' in step_copy and step_copy['docs']:
            step_copy['doc_ids'] = [get_passage_id_fn(d) for d in step_copy['docs']]
            del step_copy['docs']
        out.append(step_copy)
    return out


def _merge_rank_files(base_path, num_ranks, delete_rank_files=True, append=True):
    """Merge .rank0, .rank1, ... into base_path (append to existing content) and optionally delete rank files."""
    mode = 'a' if append else 'w'
    with open(base_path, mode) as out_f:
        for rank in range(num_ranks):
            rpath = f"{base_path}.rank{rank}"
            if os.path.exists(rpath):
                with open(rpath, 'r') as f:
                    for line in f:
                        out_f.write(line)
                if delete_rank_files:
                    os.remove(rpath)


def merge_rank_files_in_dir(output_dir):
    """
    Merge leftover .rank0, .rank1, ... files in output_dir into their base files,
    then update the metrics file (retrieval or generation) from the merged results.
    Call this manually (e.g. after an interrupted multi-GPU run) with the run output directory path.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        print(f"Not a directory: {output_dir}")
        return
    bases = {}  # base_name -> set of rank indices
    for f in output_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if ".rank" in name:
            parts = name.rsplit(".rank", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name = parts[0]
                n = int(parts[1])
                bases.setdefault(base_name, set()).add(n)
    for base_name, indices in bases.items():
        base_path = output_dir / base_name
        num_ranks = max(indices) + 1
        print(f"Merging {base_name}.rank[0..{num_ranks - 1}] into {base_path}")
        _merge_rank_files(str(base_path), num_ranks, delete_rank_files=True, append=True)
    if not bases:
        print(f"No .rank* files found in {output_dir}")
        return

    # Update metrics file from merged results
    per_query_file = output_dir / "evaluation_results_per_query_metrics.jsonl"
    metrics_file_retrieval = output_dir / "evaluation_results_metrics.json"
    metrics_file_gen = output_dir / "evaluation_results_metrics.json"

    if per_query_file.exists():
        # Detect retrieval vs generation from first line
        is_retrieval = False
        is_generation = False
        k_values_set = set()
        with open(per_query_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if 'entity_recall' in data or any(k.startswith('entity_recall@') for k in data):
                    is_retrieval = True
                    for key in data:
                        if key.startswith('entity_recall@'):
                            try:
                                k_values_set.add(int(key.split('@')[1]))
                            except ValueError:
                                pass
                if 'exact_match' in data or 'prediction' in data:
                    is_generation = True
                break
        k_values = sorted(k_values_set) if is_retrieval else []

        if is_retrieval and k_values:
            aggregated = aggregate_retrieval_metrics_from_file(str(per_query_file), k_values)
            if aggregated:
                retriever_name = output_dir.name.replace("retrieval_", "", 1) if output_dir.name.startswith("retrieval_") else "retriever"
                metrics_summary = {
                    "n": aggregated["n"],
                    "retriever": retriever_name,
                    "retrieval_eval_ks": k_values,
                    "retrieval_eval_max_k": max(k_values) if k_values else 0,
                    "entity_recall": aggregated["entity_recall"],
                }
                for k in k_values:
                    metrics_summary[f"entity_recall@{k}"] = aggregated["entity_recall_by_k"][k]
                with open(metrics_file_retrieval, 'w') as f:
                    json.dump(metrics_summary, f, indent=2)
                print(f"Updated {metrics_file_retrieval}")

        if is_generation:
            tolerance_percentages = [1.0, 5.0, 10.0, 20.0, 50.0, 90.0]
            aggregated = aggregate_generation_metrics_from_file(str(per_query_file), tolerance_percentages)
            if aggregated:
                metrics_summary = {
                    "n": aggregated["n"],
                    "exact_match": aggregated["exact_match"],
                    "soft_exact_match": aggregated["soft_exact_match"],
                }
                with open(metrics_file_gen, 'w') as f:
                    json.dump(metrics_summary, f, indent=2)
                print(f"Updated {metrics_file_gen}")


# CPU-only retriever: use multi-CPU workers (no GPU)
CPU_RETRIEVERS = ('bm25', 'oracle')


def _retrieval_worker(rank, device_id, args, items_subset, qrels, is_cpu_retriever=False):
    """Worker for multi-GPU or multi-CPU retrieval evaluation. Writes to args.trec_output_file.rank{N} and args.per_query_metrics_file.rank{N}."""
    args.qrels = qrels
    if is_cpu_retriever:
        args.device = torch.device("cpu")
        args._faiss_single_gpu = False
        worker_label = "Worker"
    else:
        args.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        args._faiss_single_gpu = True
        worker_label = "GPU"
    args.trec_output_file = f"{args.trec_output_file}.rank{rank}"
    args.per_query_metrics_file = f"{args.per_query_metrics_file}.rank{rank}"
    retriever = initialize_retriever(args)
    run_id = args.retriever_name
    k_values = sorted(set(args.retrieval_eval_ks))
    n_items = len(items_subset)
    use_tqdm = sys.stdout.isatty()
    progress_dir = Path(args.trec_output_file).parent
    progress_file = progress_dir / f".progress_{rank}"
    start_time = time.time()
    start_ts = int(start_time)
    with open(args.trec_output_file, 'w') as trec_f, open(args.per_query_metrics_file, 'w') as metrics_f:
        try:
            for i, (qid, sample) in enumerate(tqdm(items_subset, desc=f"Retrieval {worker_label} {device_id}", position=0, leave=True, disable=not use_tqdm, mininterval=10, smoothing=0.1)):
                current = i + 1
                if current % 5 == 0 or current == 1 or current == n_items:
                    try:
                        progress_file.write_text(f"{current} {n_items} {start_ts}\n")
                    except Exception:
                        pass
                if not use_tqdm and (current % 50 == 0 or current == 1 or current == n_items):
                    elapsed = time.time() - start_time
                    eta = (n_items - current) * (elapsed / current) if current > 0 else 0
                    print(f"Retrieval {worker_label} {device_id}: {current}/{n_items} | elapsed {_format_duration(elapsed)} | ETA ~{_format_duration(eta)}", flush=True)
                query = sample.get('total_recall_query', sample.get('query', sample.get('question', '')))
                gold_passage_ids = qrels.get(qid, [])
                retrieved_docs, retrieved_scores = retriever.search(query, return_score=True, qid=qid)
                retrieved_ids = [_get_passage_id(doc) for doc in retrieved_docs]
                metrics = evaluate_entity_retrieval(retrieved_ids, gold_passage_ids, k_values)
                for r, (doc, score) in enumerate(zip(retrieved_docs, retrieved_scores), start=1):
                    pid = _get_passage_id(doc)
                    trec_f.write(f"{qid} Q0 {pid} {r} {score:.4f} {run_id}\n")
                trec_f.flush()
                result_item = {
                    "qid": qid,
                    "num_gold_passages": len(gold_passage_ids),
                    "num_gold_entities": metrics['num_gold_entities'],
                    "num_retrieved_passages": len(retrieved_ids),
                    "num_retrieved_entities": metrics['num_retrieved_entities'],
                    "entity_recall": metrics['entity_recall'],
                }
                for k in k_values:
                    result_item[f'entity_recall@{k}'] = metrics[f'entity_recall@{k}']
                if 'file_id' in sample:
                    result_item['file_id'] = sample['file_id']
                if 'src' in sample:
                    result_item['src'] = sample['src']
                metrics_f.write(json.dumps(result_item) + '\n')
                metrics_f.flush()
        finally:
            try:
                progress_file.unlink(missing_ok=True)
            except Exception:
                pass


def _generation_worker(rank, device_id, args, items_subset):
    """Worker for multi-GPU generation evaluation. Writes to args.output_file.rank{N}."""
    args.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    args.output_file = f"{args.output_file}.rank{rank}"
    model = initialize_model(args)
    tolerance_percentages = [1.0, 5.0, 10.0, 20.0, 50.0, 90.0]
    n_items = len(items_subset)
    use_tqdm = sys.stdout.isatty()
    progress_dir = Path(args.output_file).parent
    progress_file = progress_dir / f".progress_gen_{rank}"
    start_time = time.time()
    start_ts = int(start_time)
    with open(args.output_file, 'w') as res_f:
        try:
            for i, (qid, sample) in enumerate(tqdm(items_subset, desc=f"Generation GPU {device_id}", position=0, leave=True, disable=not use_tqdm, mininterval=10, smoothing=0.1)):
                current = i + 1
                if current % 5 == 0 or current == 1 or current == n_items:
                    try:
                        progress_file.write_text(f"{current} {n_items} {start_ts}\n")
                    except Exception:
                        pass
                if not use_tqdm and (current % 50 == 0 or current == 1 or current == n_items):
                    elapsed = time.time() - start_time
                    eta = (n_items - current) * (elapsed / current) if current > 0 else 0
                    print(f"Generation GPU {device_id}: {current}/{n_items} | elapsed {_format_duration(elapsed)} | ETA ~{_format_duration(eta)}", flush=True)
                query = sample.get('total_recall_query', sample.get('query', sample.get('question', '')))
                if 'total_recall_answer' in sample:
                    gt_answer = str(sample['total_recall_answer'])
                elif 'answer' in sample:
                    answer = sample['answer']
                    gt_answer = str(answer.get('value', '')) if isinstance(answer, dict) else str(answer)
                else:
                    gt_answer = ''
                if args.generation_method == 'no_retrieval':
                    reasoning_path, prediction = model.inference(query)
                elif args.generation_method == 'single_retrieval':
                    reasoning_path, prediction = model.inference(query, qid=qid)
                else:
                    reasoning_path, prediction = model.inference(query, qid=qid)
                soft_matches = {}
                for pct in tolerance_percentages:
                    match_result = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=pct)
                    soft_matches[pct] = match_result['soft_match']
                exact_result = soft_exact_match(prediction, gt_answer, decimals=3, tolerance_pct=None)
                exact_match = exact_result['exact_match']
                gt_answer_cleaned = clean_gold_for_numeric(gt_answer, decimals=2)
                prediction_cleaned = clean_prediction_for_numeric(prediction)
                path_for_output = _reasoning_path_doc_ids_only(reasoning_path, _get_passage_id) if args.generation_method in ('single_retrieval', 'deep_research') else reasoning_path
                result_item = {
                    "qid": qid,
                    "query": query,
                    "gt_answer": gt_answer,
                    "gt_answer_cleaned": gt_answer_cleaned,
                    "prediction": prediction,
                    "prediction_cleaned": prediction_cleaned,
                    "exact_match": float(1 if exact_match else 0),
                    "soft_matches": soft_matches,
                    "reasoning_path": path_for_output,
                }
                if args.generation_method == 'deep_research':
                    result_item["reasoning_trajectory"] = path_for_output
                for key in ('file_id', 'original_query', 'aggregation_function'):
                    if key in sample:
                        result_item[key] = sample[key]
                res_f.write(json.dumps(result_item) + '\n')
                res_f.flush()
        finally:
            try:
                progress_file.unlink(missing_ok=True)
            except Exception:
                pass


def initialize_retriever(args):
    """Initialize retriever for retrieval-only evaluation."""
    from c5_task_evaluation.src.retrieval_models_local import BM25Retriever, RerankRetriever, DenseRetriever, SPLADERetriever, OracleRetriever

    print(f"\n=== Initializing Retriever: {args.retriever_name} ===")

    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)
    elif args.retriever_name == 'spladepp':
        retriever = SPLADERetriever(args)
    elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
        retriever = DenseRetriever(args)
    elif args.retriever_name == 'oracle':
        qrels = getattr(args, 'qrels', None)
        if not qrels:
            raise ValueError("Oracle retriever requires args.qrels (set by retrieval/generation pipeline)")
        retriever = OracleRetriever(args, qrels)
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
    args.qrels = qrels

    # Load dataset
    print("\n=== Loading Dataset ===")
    query_ids, test_dataset = load_dataset(args.dataset_file)
    print(f"Total queries in dataset: {len(test_dataset)}")
    
    # Filter to queries that have qrels
    queries_with_qrels = {qid: data for qid, data in test_dataset.items() if qid in qrels}
    print(f"Queries with qrels: {len(queries_with_qrels)}")
    
    # Get existing results for resumption (from per-query metrics file)
    generated_qids = get_existing_results(args.per_query_metrics_file)
    filtered_dataset = {qid: data for qid, data in queries_with_qrels.items() if qid not in generated_qids}
    print(f"Already processed: {len(generated_qids)}")
    print(f"Remaining: {len(filtered_dataset)}")
    
    # K values and apply limit
    k_values = sorted(set(args.retrieval_eval_ks))
    if args.limit is not None and args.limit > 0:
        filtered_dataset_items = list(filtered_dataset.items())[:args.limit]
        print(f"Limiting to first {args.limit} samples")
    else:
        filtered_dataset_items = list(filtered_dataset.items())
    
    run_id = args.retriever_name
    device_ids = getattr(args, 'device_ids', [0])
    is_cpu_retriever = args.retriever_name in CPU_RETRIEVERS
    if is_cpu_retriever:
        worker_count = getattr(args, 'num_workers', None) or len(device_ids)
        worker_ids = list(range(worker_count))
        worker_label = "Worker"
    else:
        worker_count = len(device_ids)
        worker_ids = device_ids
        worker_label = "GPU"

    # Multi-worker: split work across GPUs or CPU workers
    if worker_count > 1:
        kind = "CPU" if is_cpu_retriever else "GPU"
        print(f"\n=== Multi-{kind} Retrieval: {worker_count} workers ===")
        chunks = [[] for _ in worker_ids]
        for i, item in enumerate(filtered_dataset_items):
            chunks[i % len(worker_ids)].append(item)
        procs = []
        started_ranks = []
        for rank, (wid, chunk) in enumerate(zip(worker_ids, chunks)):
            if not chunk:
                continue
            p = multiprocessing.Process(
                target=_retrieval_worker,
                args=(rank, wid, args, chunk, qrels, is_cpu_retriever),
            )
            p.start()
            procs.append(p)
            started_ranks.append(rank)
        n_workers = max(started_ranks) + 1 if started_ranks else 0
        progress_dir = Path(args.trec_output_file).parent
        last_print_time = 0
        last_total_done = -1
        use_cr = sys.stdout.isatty()
        while any(p.is_alive() for p in procs):
            time.sleep(5)
            total_done, total_items, elapsed_sec, eta_sec, parts = _poll_worker_progress(
                progress_dir, ".progress", worker_label, worker_ids, started_ranks
            )
            if not parts:
                continue
            now = time.time()
            if total_done <= last_total_done and (now - last_print_time) < 10:
                continue
            last_total_done = total_done
            last_print_time = now
            pct = 100.0 * total_done / total_items if total_items else 0
            elapsed_str = _format_duration(elapsed_sec)
            eta_str = _format_duration(eta_sec) if eta_sec is not None else "?"
            line = f"Retrieval: {total_done}/{total_items} ({pct:.1f}%) | elapsed {elapsed_str} | ETA ~{eta_str} | " + " | ".join(parts)
            if use_cr:
                print("\r" + line, end="", flush=True)
            else:
                print(line, flush=True)
        if use_cr:
            print(flush=True)
        for p in procs:
            p.join()
        for rank in started_ranks:
            try:
                (progress_dir / f".progress_{rank}").unlink(missing_ok=True)
            except Exception:
                pass
        # Merge rank files into final outputs
        _merge_rank_files(args.trec_output_file, n_workers)
        _merge_rank_files(args.per_query_metrics_file, n_workers)
        new_queries_processed = len(filtered_dataset_items)
    else:
        # Single-GPU path
        retriever = initialize_retriever(args)
        print("\n=== Starting Retrieval Evaluation ===")
        new_queries_processed = 0
        n_single = len(filtered_dataset_items)
        use_tqdm = sys.stdout.isatty()
        start_time_single = time.time()
        with open(args.trec_output_file, 'a') as trec_f, open(args.per_query_metrics_file, 'a') as metrics_f:
            for i, (qid, sample) in enumerate(tqdm(filtered_dataset_items, desc="Retrieval", position=0, leave=True, disable=not use_tqdm, mininterval=5, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")):
                if not use_tqdm:
                    current = i + 1
                    if current % 50 == 0 or current == 1 or current == n_single:
                        elapsed = time.time() - start_time_single
                        eta = (n_single - current) * (elapsed / current) if current > 0 else 0
                        print(f"Retrieval: {current}/{n_single} ({100*current/n_single:.1f}%) | elapsed {_format_duration(elapsed)} | ETA ~{_format_duration(eta)}", flush=True)
                query = sample.get('total_recall_query', sample.get('query', sample.get('question', '')))
                gold_passage_ids = qrels.get(qid, [])
                retrieved_docs, retrieved_scores = retriever.search(query, return_score=True, qid=qid)
                retrieved_ids = [_get_passage_id(doc) for doc in retrieved_docs]
                metrics = evaluate_entity_retrieval(retrieved_ids, gold_passage_ids, k_values)
                new_queries_processed += 1
                for r, (doc, score) in enumerate(zip(retrieved_docs, retrieved_scores), start=1):
                    pid = _get_passage_id(doc)
                    trec_f.write(f"{qid} Q0 {pid} {r} {score:.4f} {run_id}\n")
                trec_f.flush()
                result_item = {
                    "qid": qid,
                    "num_gold_passages": len(gold_passage_ids),
                    "num_gold_entities": metrics['num_gold_entities'],
                    "num_retrieved_passages": len(retrieved_ids),
                    "num_retrieved_entities": metrics['num_retrieved_entities'],
                    "entity_recall": metrics['entity_recall'],
                }
                for k in k_values:
                    result_item[f'entity_recall@{k}'] = metrics[f'entity_recall@{k}']
                if 'file_id' in sample:
                    result_item['file_id'] = sample['file_id']
                if 'src' in sample:
                    result_item['src'] = sample['src']
                metrics_f.write(json.dumps(result_item) + '\n')
                metrics_f.flush()

    if new_queries_processed == 0:
        print("\nNo new queries processed in this run.")
    else:
        print(f"\nNew queries processed in this run: {new_queries_processed}")

    # Compute metrics summary aggregated over all stored results
    aggregated_metrics = aggregate_retrieval_metrics_from_file(args.per_query_metrics_file, k_values)
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

        with open(args.metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nMetrics saved to: {args.metrics_file}")
    else:
        print("No evaluation results available yet.")
    
    print(f"\nTREC run file saved to: {args.trec_output_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Run evaluation pipeline on QA datasets')

    # Pipeline selection
    parser.add_argument('--pipeline', type=str, default='retrieval', choices=['retrieval', 'generation'], help='Pipeline type: retrieval (entity recall only) or generation (full RAG with LLM)')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default="qald10_quest", choices=['qald10_quest', 'wikidata', 'synthetic_ecommerce'], help='Dataset to evaluate on (qald10_quest, wikidata, or synthetic_ecommerce)')
    parser.add_argument('--subset', type=str, default='test', choices=['train', 'val', 'test'], help='Subset: train, val, or test')
    parser.add_argument('--dataset_file', type=str, default=None, help='Path to dataset file (overrides default)')
    parser.add_argument('--qrel_file', type=str, default=None, help='Path to qrel file (for retrieval pipeline, overrides default)')

    # Generation arguments (only used when pipeline=generation)
    parser.add_argument('--model', type=str, default='openai/gpt-5.2', choices=['openai/gpt-4o', 'openai/gpt-5.2', 'anthropic/claude-sonnet-4.5', 'deepseek/deepseek-v3.2', 'qwen/qwen3-235b-a22b', 'qwen/qwen-2.5-7b-instruct'], help='Model for generation (default: openai/gpt-4o). Used only when pipeline=generation.')
    parser.add_argument('--generation_method', type=str, default='single_retrieval', choices=['no_retrieval', 'single_retrieval', 'deep_research'], help='Generation method: no_retrieval, single_retrieval, or deep_research (default: no_retrieval). Used only when pipeline=generation.')
    parser.add_argument('--deep_research_model', type=str, default='react', choices=['self_ask', 'react', 'search_o1', 'research', 'search_r1', 'step_search'], help='Deep research model when generation_method=deep_research (default: react). Used only when pipeline=generation and generation_method=deep_research.')

    # Retriever arguments (used for pipeline=retrieval; also for pipeline=generation when generation_method is single_retrieval or deep_research)
    parser.add_argument('--retriever', type=str, default='spladepp', choices=['bm25', 'spladepp', 'rerank_l6', 'rerank_l12', 'contriever', 'dpr', 'bge', 'e5', 'oracle'], help='Retriever to use (default: contriever)')
    parser.add_argument('--index_dir', type=str, default='/projects/0/prjs0834/heydars/CORPUS_Mahta/indices', help='Directory containing retrieval indices', choices=['corpus_datasets/corpus', '/projects/0/prjs0834/heydars/CORPUS_Mahta/indices'])
    parser.add_argument('--corpus_path', type=str, default='corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl', help='Path to corpus file')
    parser.add_argument('--retrieval_topk', type=int, default=50, help='Number of passages passed to the LLM when using single_retrieval or deep_research (generation pipeline only). Not used for retrieval pipeline (default: 3)')
    parser.add_argument('--retrieval_results_file', type=str, default=None, help='Optional path to TREC-format retrieval results (e.g. evaluation_results.txt). If provided and file exists, use it instead of running retrieval. If not provided, will check run_output/{run}/{subset_name}/retrieval_{retriever}/evaluation_results.txt. If no file is found, retrieval is run as usual.')
    parser.add_argument('--retrieval_eval_ks', type=int, nargs='+', default=[3, 10, 100, 1000], help='K values for retrieval evaluation (entity recall@k). Retrieval pipeline retrieves max(eval_ks) and reports these k’s (default: 1 3 10 100)')
    parser.add_argument('--splade_max_length', type=int, default=256, help='Max token length for SPLADE query encoding; must match index build (default: 256)')
    parser.add_argument('--faiss_gpu', action='store_true', default=False, help='Use GPU for FAISS computation')

    # Other arguments
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum iterations for multi-step models (default: 10)')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID when using single GPU (default: 0)')
    parser.add_argument('--devices', type=str, default=None, help='Comma-separated GPU IDs (e.g. "0,1,2,3"). If not set, all available GPUs are used automatically. Set to "0" to force single-GPU.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of CPU workers for parallel runs: retrieval with BM25 (CPU-only), or generation on CPU (e.g. API models). Enables multi-process parallelism. Default: 1 for generation; for retrieval BM25 same as devices or 1.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--run', type=str, default='run_1', help='Run identifier (default: run_1)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples to process (for testing, default: None = process all)')

    args = parser.parse_args()

    # Set dataset_name for compatibility
    args.dataset_name = args.dataset

    # Subset name for display and output paths: Dataset_subset (e.g., qald10_quest_test, wikidata_val)
    args.subset_name = f"{args.dataset}_{args.subset}"

    # Set dataset file path if not provided
    # Generated datasets (qald10_quest, wikidata, synthetic_ecommerce): corpus_datasets/generated_datasets/
    if args.dataset_file is None:
        args.dataset_file = f"corpus_datasets/generated_datasets/queries/queries_{args.dataset}_{args.subset}.jsonl"

    # Set qrel file path if not provided (for retrieval pipeline)
    if args.qrel_file is None:
        args.qrel_file = f"corpus_datasets/generated_datasets/qrels/trec_qrels_{args.dataset}_{args.subset}.txt"

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
    if args.model_name_or_path in ["openai/gpt-4o", "openai/gpt-5.2", "anthropic/claude-sonnet-4.5", "deepseek/deepseek-v3.2", "qwen/qwen3-235b-a22b", "qwen/qwen-2.5-7b-instruct"]:
        args.model_source = 'api'
    else:
        args.model_source = 'hf_local'

    # Set device and device_ids for multi-GPU (auto-detect all GPUs when --devices not given)
    if getattr(args, 'devices', None):
        args.device_ids = [int(x.strip()) for x in args.devices.split(',') if x.strip()]
    else:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        args.device_ids = list(range(n_gpus)) if n_gpus > 0 else [0]
    args.device = torch.device("cuda:" + str(args.device_ids[0]) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if len(args.device_ids) > 1:
            print(f"Multi-GPU: using {len(args.device_ids)} devices (auto-detected: {args.device_ids})")
        else:
            print(f"GPU available: {torch.cuda.get_device_name(args.device)}")

    # Set output directory and file
    if args.output_dir is None:
        if args.pipeline == 'retrieval':
            # Retrieval pipeline: run_output/{run}/retrieval/{subset_name}/{retriever}_topk{k}
            args.output_dir = f"run_output/{args.run}/{args.subset_name}/retrieval_{args.retriever_name}"
        else:
            # Generation pipeline: run_output/{run}/{dataset_subset}/generation_{model}_{generation_method}[_{retriever}|_{deep_research_model}]
            model_short = args.model_name_or_path.split('/')[-1]
            if args.generation_method == 'no_retrieval':
                args.output_dir = f"run_output/{args.run}/{args.subset_name}/generation_{model_short}_{args.generation_method}"
            elif args.generation_method == 'single_retrieval':
                args.output_dir = f"run_output/{args.run}/{args.subset_name}/generation_{model_short}_{args.generation_method}_{args.retriever_name}"
            elif args.generation_method == 'deep_research':
                args.output_dir = f"run_output/{args.run}/{args.subset_name}/generation_{model_short}_{args.generation_method}_{args.deep_research_model}"
            else:
                args.output_dir = f"run_output/{args.run}/{args.subset_name}/generation_{model_short}_{args.generation_method}"

    os.makedirs(args.output_dir, exist_ok=True)
    if args.pipeline == 'retrieval':
        args.trec_output_file = f"{args.output_dir}/evaluation_results.txt"
        args.per_query_metrics_file = f"{args.output_dir}/evaluation_results_per_query_metrics.jsonl"
        args.metrics_file = f"{args.output_dir}/evaluation_results_metrics.json"
        args.output_file = args.trec_output_file  # for any code that still references output_file
    else:
        # Per-query results (used for aggregation and significance testing)
        args.output_file = f"{args.output_dir}/evaluation_results_per_query_metrics.jsonl"
        args.metrics_file = f"{args.output_dir}/evaluation_results_metrics.json"

    # Additional retriever args for compatibility
    args.retrieval_query_max_length = 64
    args.retrieval_use_fp16 = True
    # SPLADE: splade_max_length set via --splade_max_length (must match index build)
    args.retrieval_batch_size = 512
    args.bm25_k1 = 0.9
    args.bm25_b = 0.4

    # Precomputed retrieval (generation with single_retrieval or deep_research): use file if available to avoid running retrieval
    if args.pipeline == 'generation' and args.generation_method in ('single_retrieval', 'deep_research'):
        candidate = None
        if getattr(args, 'retrieval_results_file', None) and Path(args.retrieval_results_file).exists():
            candidate = Path(args.retrieval_results_file).resolve()
        if candidate is None:
            default_path = f"run_output/{args.run}/{args.subset_name}/retrieval_{args.retriever_name}/evaluation_results.txt"
            if Path(default_path).exists():
                candidate = Path(default_path).resolve()
        if candidate is not None:
            args.precomputed_retrieval_path = str(candidate)
            print(f"Using precomputed retrieval from: {args.precomputed_retrieval_path}")
        else:
            args.precomputed_retrieval_path = None
            print("No precomputed retrieval file found; will run retrieval.")

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

    # Uncomment to merge leftover .rank* files in this run's output dir and exit (e.g. after interrupted multi-GPU run):
    # merge_rank_files_in_dir(args.output_dir)
    # return 0

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
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # already set (e.g. in Jupyter)
    sys.exit(main())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# python c5_task_evaluation/run_evalution.py --limit 400 --num_workers 10
