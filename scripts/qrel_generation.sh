#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=staging
#SBATCH --time=2:00:00
#SBATCH --mem=16GB
#SBATCH --output=script_logging/slurm_%A.out

# QRel Generation Script - Parallel Processing (OPTIMIZED)
# Generates TREC-format qrels for Total Recall RAG queries using parallel LLM calls
# Speedup: 10-20x faster than sequential processing!

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

# Run qrel generation with PARALLEL processing (DEFAULT - RECOMMENDED)
# This version uses async/await to make multiple LLM API calls concurrently
# Provides 10-20x speedup compared to sequential processing!
#
# Configuration:
#   --use_parallel         : Enable parallel LLM API calls (10-20x faster!)
#   --max_concurrent 20    : Maximum 20 concurrent API calls (adjust based on API limits)
#   --load_corpus_mode stream : Low memory mode (use 'memory' for even faster passage retrieval)
#   --dataset qald10       : Dataset to process
#
# Performance comparison (for 707 passages):
#   Sequential mode:        ~17-35 minutes per query
#   Parallel (concurrent=10): ~2-4 minutes per query (10x faster)
#   Parallel (concurrent=20): ~1-2 minutes per query (20x faster)
#
# To use sequential mode instead (NOT recommended):
#   Remove --use_parallel and --max_concurrent flags
python c3_qrel_generation/qrel_generation.py \
    --dataset qald10 \
    --use_parallel \
    --max_concurrent 20

# Print completion message
echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"
