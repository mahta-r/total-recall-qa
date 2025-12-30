import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

from utils.general_utils import set_seed
from c4_task_evaluation.models.retrieval_augmented_models import (
    NoRetrieval,
    SingleRetrieval,
    ReSearch_Model,
    SearchR1_Model,
    StepSearch_Model,
    ReAct_Model,
    SearchO1_Model,
    SelfAsk_Model
)
from c4_task_evaluation.metrics.generation_eval_metrics import em_score, f1_qald_score, f1_score



def evaluate_retrieval(retrieved_docs, gold_entities):
    """
    Evaluate the quality of retrieved documents.

    Args:
        retrieved_docs: List of retrieved documents
        gold_entities: List of gold entity IDs (if available)

    Returns:
        dict: Dictionary containing retrieval metrics
    """
    # TODO: Implement retrieval evaluation metrics
    # For now, return basic statistics
    metrics = {
        'num_retrieved': len(retrieved_docs),
        'retrieved_doc_ids': [doc['id'] for doc in retrieved_docs]
    }
    return metrics


def run_evaluation(args):
    print("\n== Running Evaluation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset_name}
        Gen. Model:  {args.generation_model}
        Retriever:   {args.retriever_name}
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
                # Handle both 'qid' and 'id' fields
                qid = data.get('qid', data.get('id'))
                if qid:
                    data['qid'] = qid  # Normalize to 'qid'
                    query_ids.append(qid)
                    test_dataset[qid] = data
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

    # === Initialize Models ==================================
    print("\n=== Initializing Models ===")
    if args.generation_model == 'no_retrieval':
        generation_model = NoRetrieval(args.device, args)
    elif args.generation_model == 'single_retrieval':
        generation_model = SingleRetrieval(args.device, args)
    elif args.generation_model == 'self_ask':
        generation_model = SelfAsk_Model(args.device, args)
    elif args.generation_model == 'react':
        generation_model = ReAct_Model(args.device, args)
    elif args.generation_model == 'search_o1':
        generation_model = SearchO1_Model(args.device, args)
    elif args.generation_model == 'research':
        generation_model = ReSearch_Model(args.device, args)
    elif args.generation_model == 'search_r1':
        generation_model = SearchR1_Model(args.device, args)
    elif args.generation_model == 'step_search':
        generation_model = StepSearch_Model(args.device, args)
    else:
        raise NotImplementedError(f"Generation model {args.generation_model} not implemented")

    print(f"Model initialized: {args.generation_model}")

    # === Main Evaluation Loop ===============================
    print("\n=== Starting Evaluation ===")
    all_results = []

    with open(args.output_file, 'a') as res_f:
        for i, (qid, sample) in enumerate(tqdm(filtered_dataset.items(), desc="Processing queries")):
            
            if i == 10:
                break
            
            # Extract query and ground truth answer
            query = sample.get('total_recall_query', sample.get('query', sample.get('question', '')))
            gt_answer = sample.get('total_recall_answer', sample.get('answer', {}).get('value', ''))

            # Convert gt_answer to string if needed
            gt_answer = str(gt_answer)

            # === Answer generation with retrieval ========================
            reasoning_path, prediction = generation_model.inference(query)

            # === Evaluate final answer ====================
            if prediction:
                em_eval = em_score(prediction, gt_answer)
                f1_qald_eval = f1_qald_score(prediction, gt_answer)
                f1_eval = f1_score(prediction, gt_answer)
            else:
                em_eval = f1_qald_eval = 0.0
                f1_eval = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

            # === Write results to file ===================
            result_item = {
                "qid": qid,
                "query": query,
                "gt_answer": gt_answer,
                "prediction": prediction,
                "em": em_eval,
                "f1_qald": f1_qald_eval,
                "f1": f1_eval['f1'],
                "reasoning_path": reasoning_path
            }

            # Add optional fields if they exist
            if 'file_id' in sample:
                result_item['file_id'] = sample['file_id']
            if 'original_query' in sample:
                result_item['original_query'] = sample['original_query']
            if 'aggregation_function' in sample:
                result_item['aggregation_function'] = sample['aggregation_function']

            all_results.append(result_item)

            # Write to file line by line
            res_f.write(json.dumps(result_item) + '\n')
            res_f.flush()  # Ensure immediate write

    # === Print Summary Statistics ===========================
    print("\n=== Evaluation Summary ===")
    if all_results:
        em_scores = [r['em'] for r in all_results]
        f1_qald_scores = [r['f1_qald'] for r in all_results]
        f1_scores = [r['f1'] for r in all_results]

        print(f"Number of queries evaluated: {len(all_results)}")
        print(f"EM Score:      {np.mean(em_scores) * 100:.2f}%")
        print(f"F1 QALD Score: {np.mean(f1_qald_scores) * 100:.2f}%")
        print(f"F1 Score:      {np.mean(f1_scores) * 100:.2f}%")
    else:
        print("No new queries were processed.")

    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run evaluation on QA datasets with retrieval')

    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='qald10',
                        choices=['qald10', 'quest'],
                        help='Dataset to evaluate on')
    parser.add_argument('--dataset_file', type=str, default=None,
                        help='Path to dataset file (overrides dataset_name)')

    # Model arguments
    parser.add_argument('--model_name_or_path', type=str, default='openai/gpt-4o',
                        choices=[
                            "openai/gpt-4o", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
                            "deepseek/deepseek-chat-v3-0324", "qwen/qwen3-235b-a22b-2507",
                            "agentrl/ReSearch-Qwen-7B-Instruct",
                            "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3",
                            "Zill1/StepSearch-7B-Base",
                            "Qwen/Qwen2.5-7B-Instruct"
                        ])

    # Generation model arguments
    parser.add_argument('--generation_model', type=str, default='no_retrieval',
                        choices=[
                            'no_retrieval', 'single_retrieval',
                            'self_ask', 'react', 'search_o1',
                            'research', 'search_r1', 'step_search'
                        ])

    # Retriever arguments
    parser.add_argument('--retriever_name', type=str, default='contriever',
                        choices=['bm25', 'rerank_l6', 'rerank_l12', 'contriever', 'dpr', 'e5', 'bge'])
    parser.add_argument('--index_dir', type=str, default='/projects/0/prjs0834/heydars/INDICES')
    parser.add_argument('--corpus_path', type=str, default='corpus_datasets/enwiki_20251001.jsonl')
    parser.add_argument('--retrieval_topk', type=int, default=3)
    parser.add_argument('--faiss_gpu', action='store_false', help='Use GPU for FAISS computation')
    parser.add_argument('--retrieval_query_max_length', type=int, default=64)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='Use FP16 for retrieval')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)

    # Other arguments
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum iterations for multi-step models')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--run', type=str, default='run_2', help='Run identifier')
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

    # === Set model source and adjust model name if needed ====================
    # Default model source
    args.model_source = 'hf_local'

    if args.generation_model == 'research':
        args.model_name_or_path = "agentrl/ReSearch-Qwen-7B-Instruct"
        args.model_source = 'hf_local'
    elif args.generation_model == 'search_r1':
        args.model_name_or_path = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3"
        args.model_source = 'hf_local'
    elif args.generation_model == 'step_search':
        args.model_name_or_path = "Zill1/StepSearch-7B-Base"
        args.model_source = 'hf_local'
    elif args.generation_model in ['self_ask', 'react', 'search_o1']:
        args.model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
        args.model_source = 'hf_local'

    # Set model source for API-based models
    if args.model_name_or_path in [
        "openai/gpt-4o", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
        "deepseek/deepseek-chat-v3-0324", "qwen/qwen3-235b-a22b-2507"
    ]:
        args.model_source = 'api'

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
        model_short = args.model_name_or_path.split('/')[-1]
        if args.generation_model in ['no_retrieval']:
            args.output_dir = f"run_output/{args.run}/{model_short}/{args.dataset_name}/{args.generation_model}"
        else:
            args.output_dir = f"run_output/{args.run}/{model_short}/{args.dataset_name}/{args.generation_model}_{args.retriever_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = f"{args.output_dir}/evaluation_results.jsonl"

    # === Run evaluation ========================
    set_seed(args.seed)
    run_evaluation(args)

    # Example usage:
    # python c4_task_evaluation/response_generation_eval.py
