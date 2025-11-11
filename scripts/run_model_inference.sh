#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=0:50:00
#SBATCH --mem=120GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0


### === Set variables ==========================
model_name_or_path="agentrl/ReSearch-Qwen-7B-Instruct"
generation_model="react"
retriever_name="contriever"
run="run_1"


# accelerate launch --multi_gpu
srun python $HOME/HighRecall_DS/c2_model_generation/model_inference.py \
    --model_name_or_path "$model_name_or_path" \
    --generation_model "$generation_model" \
    --retriever_name "$retriever_name" \
    --run "$run"
