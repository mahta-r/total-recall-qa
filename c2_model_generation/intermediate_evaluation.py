import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
import requests
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.general_utils import set_seed

os.environ["OPENAI_API_KEY"] = ''


def recall_entities(args):
    print("\n== Recall Entities ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Gen. Model:  {args.generation_model}
        Retriever:   {args.retriever_name}
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === Read test dataset ================
    query_ids, test_dataset = [], {}
    if os.path.exists(args.dataset_file):
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    query_ids.append(data['qid'])
                    test_dataset[data['qid']] = data
    print(f"Test dataset size: {len(test_dataset)}")
    
    # === Functions ========================
    def get_unique_docs(reasoning_path):
        unique_docs = set()
        for step in reasoning_path:
            for doc in step.get("docs", []):
                unique_docs.add((doc["id"], doc["title"]))
        return list(unique_docs)
    
    def get_wikidata_qid(pageid=None, title=None):
        # print(pageid)
        # print(title)
        if not pageid and not title:
            raise ValueError("You must provide either a pageid or a title.")

        base_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "prop": "pageprops", "format": "json"}

        if pageid:
            params["pageids"] = pageid
        else:
            params["titles"] = title

        headers = {"User-Agent": "HeydarSoudaniBot/1.0 (heydar.soudani@gmail.com)"} 

        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None

        page_info = next(iter(pages.values()))
        if "missing" in page_info:
            return None

        return page_info.get("pageprops", {}).get("wikibase_item")
    
    def get_wikidata_ids(unique_docs):
        return [get_wikidata_qid(doc_id.split('-')[0], doc_title) for (doc_id, doc_title) in unique_docs]
            
    def set_coverage(predicted_ids, gold_ids):
        predicted_set = set(predicted_ids)
        gold_set = set(gold_ids)
        if not gold_set:
            return 0.0
        intersection = len(predicted_set & gold_set)
        coverage = intersection / len(gold_set)
        return coverage

    def plot_coverage(correctness_list, set_coverage_list, steps_list=None, title=None):
        corr = np.asarray(correctness_list).astype(bool)
        cov = np.asarray(set_coverage_list, dtype=float)

        # Add small horizontal jitter for visual separation
        rng = np.random.default_rng(0)
        jitter = rng.normal(0, 0.02, size=len(cov))

        # x = all points centered around 0 (one vertical line)
        x = np.zeros(len(cov)) + jitter

        # Colors: blue for correct, red for incorrect
        colors = np.where(corr, 'blue', 'red')

        plt.figure(figsize=(3,5), dpi=150)
        plt.scatter(x, cov, facecolors='none', edgecolors=colors)
        # plt.ylim(0, 1)
        plt.xlim(-0.1, 0.1)
        plt.xticks([])
        plt.ylabel("Coverage")
        plt.title("Coverage by Correctness")
        plt.tight_layout()
        plt.savefig("_figs/coverage_correctness.png", dpi=500, bbox_inches="tight")
    
    # === Main loop ========================
    correctness_list, set_coverage_list = [], []
    if os.path.exists(args.generation_results_file):
        with open(args.generation_results_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f)):
                # if idx == 40:
                #     break
                sample = json.loads(line)
                if 'qid' in sample:
                    file_id, qid, query, gt_answer = sample['file_id'], sample['qid'], sample['query'], sample['gt_answer']
                    correctness, reasoning_path = sample['f1_qald'], sample['reasoning_path']
                    gold_ids = test_dataset[qid].get('intermidate_list', [])
                    
                    unique_docs = get_unique_docs(reasoning_path)
                    predicted_ids = get_wikidata_ids(unique_docs)
                    set_coverage_list.append(set_coverage(predicted_ids, gold_ids))
                    correctness_list.append(correctness)
    
    plot_coverage(correctness_list, set_coverage_list)
                    


def llm_as_judge_passages(args):
    print("\n== LLM as Judge Passeges ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Gen. Model:  {args.generation_model}
        Retriever:   {args.retriever_name}
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === Read test dataset ================
    query_ids, test_dataset = [], {}
    if os.path.exists(args.dataset_file):
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    query_ids.append(data['qid'])
                    test_dataset[data['qid']] = data
    print(f"Test dataset size: {len(test_dataset)}")

    # === Functions ========================
    # query generation
    
    
    # Passage judge
    
    
    # === Main loop ========================
    correctness_list, set_coverage_list = [], []
    if os.path.exists(args.generation_results_file):
        with open(args.generation_results_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f)):
                # if idx == 40:
                #     break
                sample = json.loads(line)
                if 'qid' in sample:
                    file_id, qid, query, gt_answer = sample['file_id'], sample['qid'], sample['query'], sample['gt_answer']
                    correctness, reasoning_path = sample['f1_qald'], sample['reasoning_path']
                    gold_ids = test_dataset[qid].get('intermidate_list', [])
                    
                    # filter samples with very large gold_ids
                    
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Generation
    parser.add_argument('--model_name_or_path', type=str, default='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3', choices=[
        "openai/gpt-4o", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
        "deepseek/deepseek-chat-v3-0324", "qwen/qwen3-235b-a22b-2507",
        # 
        "agentrl/ReSearch-Qwen-7B-Instruct",                            # ReSearch
        "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3", # Search-R1
        "Zill1/StepSearch-7B-Base",                                     # StepSearch
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    
    # Retriever
    parser.add_argument('--generation_model', type=str, default='react', choices=[
        'no_retrieval', 'single_retrieval',
        'self_ask', 'react', 'search_o1',
        'research', 'search_r1', 'step_search'
    ])
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'rerank_l6', 'rerank_l12', 'contriever', 'dpr', 'e5', 'bge'
    ])
    
    parser.add_argument('--index_dir', type=str, default='/projects/0/prjs0834/heydars/INDICES')
    parser.add_argument('--corpus_path', type=str, default='corpus_datasets/enwiki_20251001.jsonl')
    parser.add_argument('--retrieval_topk', type=int, default=3)
    parser.add_argument('--faiss_gpu', action='store_false', help='Use GPU for computation')
    parser.add_argument('--retrieval_query_max_length', type=int, default=64)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    
    # Others
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1')
    parser.add_argument("--seed", type=int, default=10)
    
    args = parser.parse_args()
    
    # === Define CUDA device =======
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
    
    # === Models ====================
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

        
    if args.model_name_or_path in [
        "openai/gpt-4o", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash", 
        "deepseek/deepseek-chat-v3-0324", "qwen/qwen3-235b-a22b-2507"
    ]:
        args.model_source = 'api'
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    if args.generation_model in ['no_retrieval']:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.generation_model}"
    else:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.generation_model}_{args.retriever_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    args.generation_results_file = f"{args.output_dir}/generation_results.jsonl"
    args.dataset_file = f"corpus_datasets/qald_aggregation_samples/wikidata_totallist.jsonl"
    
    
    # === Run ========================
    set_seed(args.seed)
    recall_entities(args)
    # llm_as_judge_passages(args)
    
    # python c2_model_generation/intermediate_evaluation.py


