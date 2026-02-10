#!/bin/bash

# Script to merge rewritten passages into the main corpus
# This script merges all 5 rewritten passage files into the original corpus

set -e  # Exit on error

# Set default paths (relative to project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default paths
CORPUS_FILE="${CORPUS_FILE:-corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl}"

# Rewritten passage files
REWRITE_FILES=(
    "corpus_datasets/dataset_creation_heydar/qald10/passage_rewrites_v2_qald10.jsonl"
    "corpus_datasets/dataset_creation_heydar/quest/test/passage_rewrites_v2_test_quest.jsonl"
    "corpus_datasets/dataset_creation_heydar/quest/train/passage_rewrites_v2_train_quest.jsonl"
    "corpus_datasets/dataset_creation_heydar/quest/val/passage_rewrites_v2_val_quest.jsonl"
    "corpus_datasets/dataset_creation_mahta/passage_rewrites.jsonl"
)

echo "=========================================="
echo "Merge Rewritten Passages into Corpus"
echo "=========================================="
echo ""
echo "Corpus file: $CORPUS_FILE"
echo "Output file: $OUTPUT_FILE"
echo ""
echo "Rewritten passage files:"
for file in "${REWRITE_FILES[@]}"; do
    echo "  - $file"
done
echo ""
echo "=========================================="
echo ""

# Check if corpus file exists
if [ ! -f "$CORPUS_FILE" ]; then
    echo "Error: Corpus file not found: $CORPUS_FILE"
    echo "Please set CORPUS_FILE environment variable or place the file at the default location."
    exit 1
fi

# Check if at least one rewrite file exists
found_rewrite=false
for file in "${REWRITE_FILES[@]}"; do
    if [ -f "$file" ]; then
        found_rewrite=true
        break
    fi
done

if [ "$found_rewrite" = false ]; then
    echo "Error: No rewritten passage files found."
    echo "Please ensure at least one of the following files exists:"
    for file in "${REWRITE_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

# Run the merge script with qrel updates
python c2_corpus_creation/merge_rewritten_passages.py \
    --corpus "$CORPUS_FILE" \
    --output "$OUTPUT_FILE" \
    --rewrites "${REWRITE_FILES[@]}" \
    --buffer-size 10000 \
    --update-qrels

echo ""
echo "=========================================="
echo "Merge completed successfully!"
echo "Output saved to: $OUTPUT_FILE"
echo ""
echo "Qrel files and ID mappings have been updated."
echo "=========================================="
