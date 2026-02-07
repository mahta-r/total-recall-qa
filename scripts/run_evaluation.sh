#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h100
#SBATCH --time=3:30:00
#SBATCH --mem=720GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd "${HOME}/total-recall-rag" || exit 1
mkdir -p script_logging

export OPENAI_API_KEY='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'

### === Set variables ==========================
pipeline=retrieval
dataset=qald10_quest
subset=test

model=openai/gpt-5.2
generation_method=single_retrieval
retriever=bge
deep_research_model=react

retrieval_eval_ks="3 10 100 1000"
num_workers=10
run=run_1

### === Run ==========================
if [ "$pipeline" = "retrieval" ]; then
  python c5_task_evaluation/run_evalution.py \
    --pipeline $pipeline \
    --dataset $dataset \
    --subset $subset \
    --retriever $retriever \
    --retrieval_eval_ks $retrieval_eval_ks \
    --run $run \
    --devices "0,1,2"
else
  python c5_task_evaluation/run_evalution.py \
    --pipeline $pipeline \
    --dataset $dataset \
    --subset $subset \
    --retriever $retriever \
    --retrieval_eval_ks $retrieval_eval_ks \
    --model $model \
    --generation_method $generation_method \
    --run $run
fi
