#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu_h100
#SBATCH --time=3:20:00
#SBATCH --mem=376GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0


### === Set variables ==========================
corpus_file=corpus_datasets/enwiki_20251001.jsonl
save_dir=/projects/0/prjs0834/heydars/INDICES
retriever_name=contriever

# srun 
CUDA_VISIBLE_DEVICES=0,1,2,3 python $HOME/HighRecall_DS/c2_model_generation/src/index_builder.py \
    --retrieval_method $retriever_name \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --faiss_type Flat \
    --save_embedding
