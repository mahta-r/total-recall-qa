#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=staging
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --output=script_logging/slurm_%A.out

# Dataset Creation Pipeline Script
# Runs the unified dataset processing pipeline for QALD10 or Quest datasets

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

# Run the dataset creation pipeline
#
# Configuration options:
#   --dataset: "qald10" or "quest"
#   --model: LLM model to use (e.g., openai/gpt-4o)
#   --quest_input: Input file for Quest dataset (e.g., train.jsonl, test.jsonl)
#   --property_num: "all" or "log" (logarithmic selection)
#   --selection_strategy: "random" or "least" (prioritize rare properties)
#   --max_props: Maximum properties per query (optional)
#   --temperature: Generation temperature (default: 0.7)
#   --seed: Random seed (default: 42)
#   --resume / --no-resume: Resume from last processed entry
#
python c1_2_dataset_creation_heydar/run_pipeline.py \
    --dataset quest \
    --quest_input test.jsonl \
    --model openai/gpt-5.2 \
    --property_num log \
    --selection_strategy least \
    --max_props 3

# Print completion message
echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"
