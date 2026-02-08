#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h100
#SBATCH --time=2:20:00
#SBATCH --mem=720GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd "${HOME}/total-recall-rag" || exit 1
mkdir -p script_logging

# export OPENAI_API_KEY=''

### === Set variables ==========================
pipeline=generation
dataset=wikidata
subset=test

model=Qwen/Qwen2.5-7B-Instruct
generation_method=deep_research
retriever=e5
deep_research_model=react

retrieval_eval_ks="3 10 100 1000"
run=run_1

### === Run ==========================
if [ "$pipeline" = "retrieval" ]; then
  python c5_task_evaluation/run_evalution.py \
    --pipeline $pipeline \
    --dataset $dataset \
    --subset $subset \
    --retriever $retriever \
    --retrieval_eval_ks $retrieval_eval_ks \
    --run $run
else
  python c5_task_evaluation/run_evalution.py \
    --pipeline $pipeline \
    --dataset $dataset \
    --subset $subset \
    --retriever $retriever \
    --retrieval_eval_ks $retrieval_eval_ks \
    --model $model \
    --generation_method $generation_method \
    --deep_research_model $deep_research_model \
    --run $run
fi
