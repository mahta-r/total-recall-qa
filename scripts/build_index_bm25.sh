#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=staging
#SBATCH --time=3:00:00
#SBATCH --mem=180GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

### === Set cache to project directory (avoid home quota issues) ===
export HF_DATASETS_CACHE=/projects/0/prjs0834/heydars/.cache/huggingface
export HF_HOME=/projects/0/prjs0834/heydars/.cache/huggingface

### === Set variables ==========================
corpus_file=corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl
save_dir=/projects/0/prjs0834/heydars/CORPUS_Mahta/indices
retriever_name=bm25

python $HOME/total-recall-rag/c2_corpus_creation/src/index_builder.py \
    --retrieval_method $retriever_name \
    --corpus_path $corpus_file \
    --save_dir $save_dir
