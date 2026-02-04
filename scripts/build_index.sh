#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=6:00:00
#SBATCH --mem=360GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Run from project root so relative corpus_path resolves
cd "${HOME}/total-recall-rag" || exit 1
mkdir -p script_logging

### === Set variables ==========================
corpus_file=corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl
save_dir=/projects/0/prjs0834/heydars/CORPUS_Mahta/indices
retriever_name=e5
embedding_path=/projects/0/prjs0834/heydars/CORPUS_Mahta/indices/emb_e5.memmap
# srun 
CUDA_VISIBLE_DEVICES=0,1,2,3 python $HOME/total-recall-rag/c2_corpus_creation/index_builder.py \
    --retrieval_method $retriever_name \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --faiss_type Flat \
    --save_embedding \
    --embedding_path $embedding_path
