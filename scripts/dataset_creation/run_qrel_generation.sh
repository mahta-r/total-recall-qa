#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=staging
#SBATCH --time=12:00:00
#SBATCH --mem=180GB
#SBATCH --output=script_logging/slurm_%A.out

# QRel Generation Script - Parallel Processing with 4-way Classification + Rewriting
# Generates TREC-format qrels using extract_qrels approach (YES-SAME/YES-DIFFERENT/NO-RELATED/NO-UNRELATED)
# Includes passage rewriting for YES-DIFFERENT and NO-RELATED cases
# Speedup: 5-10x faster than sequential processing!

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
export OPENAI_API_KEY=''

# Navigate to project directory
cd /gpfs/home6/hsoudani/total-recall-rag

# Create logging directory if it doesn't exist
mkdir -p script_logging
mkdir -p qrel_logging

# Run qrel generation with PARALLEL processing (DEFAULT - RECOMMENDED)
# This version uses async/await to make multiple LLM API calls concurrently
# Provides 5-10x speedup compared to sequential processing!
#
# Key Features:
#   - 4-way classification: YES-SAME, YES-DIFFERENT, NO-RELATED, NO-UNRELATED
#   - Passage rewriting for YES-DIFFERENT (REPLACE) and NO-RELATED (ADD)
#   - Parallel judging and rewriting for maximum speed
#
# Configuration:
#   --dataset quest            : Dataset to process (quest or qald10)
#   --subset val              : Subset to process (test, train, val)
#   --use_parallel            : Enable parallel LLM API calls (5-10x faster!)
#   --max_concurrent 20       : Maximum 20 concurrent API calls (adjust based on API limits)
#   --load_corpus_mode memory : High memory mode for fastest passage retrieval (use 'stream' for low memory)
#   --judge_model gpt-4o-mini : Model for relevance judgment
#   --rewrite_model gpt-4o-mini : Model for passage rewriting
#   --limit 10                : Process only first 10 queries (remove for full dataset)
#   --rewrite_passages        : Enable rewriting (default: True, use --no_rewrite to disable)
#
# Performance comparison:
#   Sequential mode:           ~20-40 minutes per query (depending on passages)
#   Parallel (concurrent=10):  ~4-8 minutes per query (5x faster)
#   Parallel (concurrent=20):  ~2-4 minutes per query (10x faster)
#
# Memory modes:
#   --load_corpus_mode memory : Loads entire corpus in RAM, fastest passage retrieval (requires ~180GB)
#   --load_corpus_mode stream : Streams from disk, slower but uses less memory (~10-20GB)
#
# Rewriting options:
#   --rewrite_passages        : Enable rewriting (default)
#   --no_rewrite             : Disable rewriting, only judge passages (faster, no rewrites in qrels)
#
# Resume functionality (continue after interruption):
#   --resume                  : Enable resume mode, skip already-completed queries
#   --reprocess_last         : Always re-process last query (safest, may waste LLM calls)
#   --trust_last             : Trust last query is complete (fastest, risky if interrupted)
#   --min_passages_threshold N : Re-process if last query has < N passages (default: 5)
#   Note: Resume cleans up both qrels and rewrites files for re-processed queries
#
# To use sequential mode instead (NOT recommended):
#   Remove --use_parallel and --max_concurrent flags

python c3_qrel_generation/qrel_generation.py \
    --dataset qald10 \
    --use_parallel \
    --max_concurrent 20 \
    --load_corpus_mode memory \
    --judge_model gpt-4o-mini \
    --judge_temperature 0.0 \
    --rewrite_model gpt-4o-mini \
    --rewrite_temperature 0.7 \
    --rewrite_passages

# Alternative configurations (uncomment to use):

# 0. Resume from interruption (default heuristic - balanced):
# python c3_qrel_generation/qrel_generation.py \
#     --dataset quest \
#     --subset val \
#     --use_parallel \
#     --max_concurrent 20 \
#     --load_corpus_mode memory \
#     --judge_model gpt-4o-mini \
#     --rewrite_model gpt-4o-mini \
#     --resume

# 0b. Resume with safe strategy (always re-process last query):
# python c3_qrel_generation/qrel_generation.py \
#     --dataset quest \
#     --subset val \
#     --use_parallel \
#     --max_concurrent 20 \
#     --load_corpus_mode memory \
#     --judge_model gpt-4o-mini \
#     --rewrite_model gpt-4o-mini \
#     --resume \
#     --reprocess_last

# 1. Full dataset processing (no limit):
# python c3_qrel_generation/qrel_generation.py \
#     --dataset quest \
#     --subset test \
#     --use_parallel \
#     --max_concurrent 20 \
#     --load_corpus_mode memory \
#     --judge_model gpt-4o-mini \
#     --rewrite_model gpt-4o-mini

# 2. Judge only (no rewriting, faster):
# python c3_qrel_generation/qrel_generation.py \
#     --dataset quest \
#     --subset val \
#     --use_parallel \
#     --max_concurrent 20 \
#     --load_corpus_mode memory \
#     --judge_model gpt-4o-mini \
#     --no_rewrite \
#     --limit 10

# 3. Streaming mode (lower memory usage):
# python c3_qrel_generation/qrel_generation.py \
#     --dataset quest \
#     --subset val \
#     --use_parallel \
#     --max_concurrent 20 \
#     --load_corpus_mode stream \
#     --judge_model gpt-4o-mini \
#     --rewrite_model gpt-4o-mini \
#     --limit 10

# 4. Sequential mode (not recommended, but available):
# python c3_qrel_generation/qrel_generation.py \
#     --dataset quest \
#     --subset val \
#     --load_corpus_mode memory \
#     --judge_model gpt-4o-mini \
#     --rewrite_model gpt-4o-mini \
#     --limit 10

# Print completion message
echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"

# Output files will be located at:
#   QRels:    corpus_datasets/dataset_creation_heydar/quest/val/qrels_val_quest.txt
#   Rewrites: corpus_datasets/dataset_creation_heydar/quest/val/passage_rewrites_val_quest.jsonl
#   Log:      qrel_logging/qrel_generation_val_quest_YYYYMMDD_HHMMSS.log
