#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu_h100
#SBATCH --time=20:00:00
#SBATCH --mem=40GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Run from project root so relative corpus_path resolves
cd "${HOME}/total-recall-rag" || exit 1
mkdir -p script_logging

### === Set cache to project directory (avoid home quota issues) ===
export HF_DATASETS_CACHE=/projects/0/prjs0834/heydars/.cache/huggingface
export HF_HOME=/projects/0/prjs0834/heydars/.cache/huggingface

### === Set variables ==========================
corpus_file=corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl
save_dir=/projects/0/prjs0834/heydars/CORPUS_Mahta/indices
retriever_name=spladepp

# SPLADE++ encoding with single GPU (optimized)
# - Larger batch size (512) for better GPU utilization
# - Estimated time: ~15-18 hours for 57M documents
# - Reduce batch_size to 256 if you hit GPU OOM errors

python $HOME/total-recall-rag/c2_corpus_creation/index_builder.py \
    --retrieval_method $retriever_name \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --save_embedding
