#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h100
#SBATCH --time=0:30:00
#SBATCH --mem=180GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd "${HOME}/total-recall-rag" || exit 1
mkdir -p script_logging

### === Set variables ==========================
pipeline=retrieval
dataset=qald10_quest
subset=test
run=run_1
retriever=contriever
retrieval_eval_ks="3 10 100 1000"

model=openai/gpt-4o
generation_method=no_retrieval
deep_research_model=react

### === Run ==========================
if [ "$pipeline" = "retrieval" ]; then
  python c5_task_evaluation/run_evalution.py --pipeline $pipeline --dataset $dataset --subset $subset --run $run --retriever $retriever --retrieval_eval_ks $retrieval_eval_ks
else
  python c5_task_evaluation/run_evalution.py --pipeline $pipeline --dataset $dataset --subset $subset --run $run --retriever $retriever --retrieval_eval_ks $retrieval_eval_ks --model $model --generation_method $generation_method --deep_research_model $deep_research_model
fi
