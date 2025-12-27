#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=staging
#SBATCH --time=3:00:00
#SBATCH --mem=64GB
#SBATCH --output=script_logging/slurm_%A.out

# QRel Generation Script
# Generates TREC-format qrels for Total Recall RAG queries

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

# Set OpenAI API key (make sure this is set in your environment)
# export OPENAI_API_KEY="your_api_key_here"

# Navigate to project directory
cd /gpfs/home6/hsoudani/total-recall-rag

# Create logging directory if it doesn't exist
mkdir -p script_logging
mkdir -p qrel_logging

export OPENAI_API_KEY='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'

# Run qrel generation
# Modify parameters as needed:
# --dataset: Dataset name (e.g., "qald10", "quest")
# --num_workers: Number of parallel workers (adjust based on your CPUs)
# --model: LLM model to use (default: gpt-4o)
# --temperature: Temperature for LLM (default: 0.0)

python c2_corpus_annotation/qrel_generation.py \
    --dataset quest \
    --num_workers 32 \
    --model gpt-4o \
    --temperature 0.0

# Print completion message
echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"
