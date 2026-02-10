#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=staging
#SBATCH --time=3:00:00
#SBATCH --mem=64GB
#SBATCH --output=script_logging/slurm_%A.out


### === Set variables ==========================
dump_date="20251001"
output_dir="/projects/0/prjs0834/heydars/INDICES"
output_path="${output_dir}/enwiki-${dump_date}-pages-articles.xml.bz2"
extract_output_dir="${output_dir}/enwiki-${dump_date}"


# ### --- Step 1: Download the Wikipedia XML dump
# echo "=== Step 1: Downloading Wikipedia XML dump ==="
# python c2_corpus_annotation/src/dump2corpus.py --step download \
#     --dump-date "$dump_date" \
#     --output-dir "$output_dir"


# ### --- Step 2: Extract clean text from Wikipedia XML dump
# echo "=== Step 2: Setting up conda environment and extracting clean text ==="

# # Create conda environment if it doesn't exist
# if ! conda env list | grep -q "wiki_env"; then
#     echo "Creating wiki_env conda environment..."
#     conda create -n wiki_env python=3.8 -y
# fi

# # Activate conda environment and install dependencies
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate wiki_env

# # Install required packages
# pip install wikiextractor SoMaJo

# # Run wikiextractor
# echo "Running wikiextractor..."
# python -m wikiextractor.WikiExtractor "$output_path" \
#     -o "$extract_output_dir" \
#     --processes 16

# # Deactivate conda environment
# conda deactivate


### --- Step 3: Convert wiki dump to jsonl corpus
echo "=== Step 3: Converting wiki dump to jsonl corpus ==="
python c2_corpus_annotation/src/dump2corpus.py --step convert \
    --dump-date "$dump_date" \
    --output-dir "$output_dir" \
    --words-per-passage 100 \
    --min-words 20 \
    --progress-every 10000

echo "=== Done ==="
