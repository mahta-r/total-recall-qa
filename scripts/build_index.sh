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
corpus_file=corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl
save_dir=/projects/0/prjs0834/heydars/CORPUS_Mahta/indices
retriever_name=bm25

# srun 
CUDA_VISIBLE_DEVICES=0,1,2,3 python $HOME/total-recall-rag/c2_corpus_creation/index_builder.py \
    --retrieval_method $retriever_name \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --faiss_type Flat \
    --save_embedding
