#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=1:30:00
#SBATCH --mem=20GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0


### === Set variables ==========================
model_name_or_path="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3"
dataset_title="mahta"
generation_model="search_r1"
retriever_name="rerank_l6"
run="run_1"


# accelerate launch --multi_gpu
srun python $HOME/total-recall-rag/c2_model_generation/model_inference.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset_title "$dataset_title" \
    --generation_model "$generation_model" \
    --retriever_name "$retriever_name" \
    --run "$run"
