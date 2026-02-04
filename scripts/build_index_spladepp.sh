#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=staging
#SBATCH --time=10:00:00
#SBATCH --mem=180GB
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
# When set: index-only from existing vectors (no GPU needed). When commented out: encode + index.
embedding_path=/projects/0/prjs0834/heydars/CORPUS_Mahta/indices/spladepp_vectors/vectors.jsonl

# Fail fast if we expect index-only but vectors are missing
if [ -n "$embedding_path" ] && [ ! -f "$embedding_path" ]; then
    echo "ERROR: embedding_path is set but file not found: $embedding_path"
    exit 1
fi

# With embedding_path set: indexing only (no encoding, no GPU). Otherwise: SPLADE++ encode + index (~15-18h for 57M docs).

python $HOME/total-recall-rag/c2_corpus_creation/index_builder.py \
    --retrieval_method $retriever_name \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --save_embedding \
    --embedding_path $embedding_path
