import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import numpy as np
from tqdm import tqdm

from utils.general_utils import set_seed
from c2_model_generation.src.retrieval_augmented_models import *
from c2_model_generation.src.correctness_evaluation import f1_qald_score, em_score

os.environ["OPENAI_API_KEY"] = ''


def generation(args):
    print("\n== Generation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset_title}
        Gen. Model:  {args.generation_model}
        Retriever:   {args.retriever_name}
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === Read input data ========================
    query_ids, test_dataset = [], {}
    if os.path.exists(args.dataset_file):
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    query_ids.append(data['qid'])
                    test_dataset[data['qid']] = data
    print(f"Test dataset size: {len(test_dataset)}")


    # === Read existing data ====================
    generated_qids, generated_em_evaluation, generated_f1_qald_evaluation = [], [], []
    if os.path.exists(args.generation_results_file):
        with open(args.generation_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
                    generated_em_evaluation.append(data['em'])
                    generated_f1_qald_evaluation.append(data['f1_qald'])
    generated_qids = set(generated_qids)
    filtered_dataset = {qid: data for qid, data in test_dataset.items() if qid not in generated_qids}
    print(f"Inference dataset size: {len(filtered_dataset)}")


    # === Models ================================
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
        raise NotImplementedError
    
    
    # === Inference =============================
    em_evaluation = generated_em_evaluation
    f1_qald_evaluation = generated_f1_qald_evaluation
    
    with open(args.generation_results_file, 'a') as res_f:
        for i, (qid, sample) in enumerate(tqdm(filtered_dataset.items(), desc=f"inferencing ...")):
            # if i == 3:
            #     break
            file_id, qid = sample.get('file_id', ''), sample.get('qid', '')
            # query, gt_answer = sample.get('query', ''), sample.get('updated_answer', '') # For QALD
            query, gt_answer = sample.get('question', ''), sample.get('answer', '').get('value', '') # For Mahta
            
            reasoning_path, prediction = generation_model.inference(query)
            
            if prediction:
                em_eval = em_score(prediction, str(gt_answer))
                f1_qald_eval = f1_qald_score(prediction, str(gt_answer))
            else:
                em_eval = f1_qald_eval = 0.0
            em_evaluation.append(em_eval)
            f1_qald_evaluation.append(f1_qald_eval)
            
            item = {
                "file_id": file_id,
                "qid": qid,
                "query": query,
                "gt_answer": gt_answer,
                "prediction": prediction,
                "f1_qald": f1_qald_eval,
                "em": em_eval,
                "reasoning_path": reasoning_path   
            }
            res_f.write(json.dumps(item) + '\n')
            
    print("\nEvaluation Result:")
    print(f"F1 (QALD): {np.mean(f1_qald_evaluation)*100}")
    print(f"EM: {np.mean(em_evaluation)*100}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Generation
    parser.add_argument('--model_name_or_path', type=str, default='google/gemini-2.5-flash', choices=[
        "openai/gpt-4o", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
        "deepseek/deepseek-chat-v3-0324", "qwen/qwen3-235b-a22b-2507",
        # 
        "agentrl/ReSearch-Qwen-7B-Instruct",                            # ReSearch
        "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3", # Search-R1
        "Zill1/StepSearch-7B-Base",                                     # StepSearch
    ])
    parser.add_argument('--dataset_title', type=str, default="mahta", choices=['qald_aggregation', 'qald_aggregation_aug', 'mahta'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    
    # Retriever
    parser.add_argument('--generation_model', type=str, default='no_retrieval', choices=[
        'no_retrieval', 'single_retrieval',
        'self_ask', 'react', 'search_o1',
        'research', 'search_r1', 'step_search'
    ])
    parser.add_argument('--retriever_name', type=str, default='contriever', choices=[
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
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset_title}/{args.generation_model}"
    else:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset_title}/{args.generation_model}_{args.retriever_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    args.generation_results_file = f"{args.output_dir}/generation_results.jsonl"
    
    if args.dataset_title == 'mahta':
        args.dataset_file = "corpus_datasets/mahta_queries/queries.jsonl"
    elif args.dataset_title == 'qald_aggregation':
        args.dataset_file = "corpus_datasets/qald_aggregation_samples/wikidata_totallist.jsonl"
    elif args.dataset_title == 'qald_aggregation_aug':
        args.dataset_file = ""
    else:
        raise NotImplementedError
    
    # === Run ========================
    set_seed(args.seed)
    generation(args)
    
    # python c2_model_generation/model_inference.py
