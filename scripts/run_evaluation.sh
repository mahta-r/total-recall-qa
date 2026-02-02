#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=staging
#SBATCH --time=4:00:00
#SBATCH --mem=180GB
#SBATCH --output=script_logging/slurm_eval_%j.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

### === Set variables ==========================
pipeline=retrieval
dataset=heydar
subset=test
run=run_1
retriever=bm25
retrieval_topk=100
model=openai/gpt-4o
method_type=no_retrieval
method=no_retrieval

python $HOME/total-recall-rag/c5_task_evaluation/run_evalution.py \
    --pipeline $pipeline \
    --dataset $dataset \
    --subset $subset \
    --run $run \
    --retriever $retriever \
    --retrieval_topk $retrieval_topk \
    --model $model \
    --method_type $method_type \
    --method $method
