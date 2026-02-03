#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h100
#SBATCH --time=4:30:00
#SBATCH --mem=180GB
#SBATCH --output=script_logging/slurm_%A.out


module load 2024
module load Python/3.12.3-GCCcore-13.3.0

### === Set variables ==========================
# Pipeline: retrieval (only retriever args) or generation (generation_method + optional deep_research_model)
pipeline=retrieval
dataset=heydar
subset=test
run=run_1
retriever=e5
# Generation args (used only when pipeline=generation)
model=openai/gpt-4o
generation_method=no_retrieval   # no_retrieval | single_retrieval | deep_research
deep_research_model=react        # self_ask | react | search_o1 | research | search_r1 | step_search (when generation_method=deep_research)

BASE_ARGS="--pipeline $pipeline --dataset $dataset --subset $subset --run $run --retriever $retriever"

if [ "$pipeline" = "retrieval" ]; then
    python $HOME/total-recall-rag/c5_task_evaluation/run_evalution.py $BASE_ARGS
else
    python $HOME/total-recall-rag/c5_task_evaluation/run_evalution.py $BASE_ARGS \
        --model $model \
        --generation_method $generation_method \
        --deep_research_model $deep_research_model
fi
