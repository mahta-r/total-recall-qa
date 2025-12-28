#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=staging
#SBATCH --time=6:00:00
#SBATCH --mem=16GB
#SBATCH --output=script_logging/slurm_%A.out

# QRel Generation Script (Single Worker)
# Generates TREC-format qrels for Total Recall RAG queries using sequential processing

# Load any required modules (adjust as needed for your cluster)
# module load python/3.9
# module load conda

# Activate conda environment if needed
# source activate your_env_name

# Print job information
echo "======================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "======================================"
echo ""

# Set OpenAI API key
export OPENAI_API_KEY='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'

# Navigate to project directory
cd /gpfs/home6/hsoudani/total-recall-rag

# Create logging directory if it doesn't exist
mkdir -p script_logging
mkdir -p qrel_logging

# Run qrel generation (single worker version)
# This version processes queries sequentially - slower but simpler and more reliable
#
# Corpus loading modes:
#   --load_corpus_mode stream  : Low memory, slower per-query (good for small query batches)
#   --load_corpus_mode memory  : High memory, faster per-query (good for large query batches)
python c2_corpus_annotation/qrel_generation.py \
    --dataset qald10 \
    --load_corpus_mode memory

# Print completion message
echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"
