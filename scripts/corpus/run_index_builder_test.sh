#!/bin/bash
# Run index builder test: small corpus -> index -> retrieval -> recall evaluation
# Usage: ./scripts/run_index_builder_test.sh [retriever]
# Example: ./scripts/run_index_builder_test.sh bge

set -e
cd "$(dirname "$0")/.." || exit 1
mkdir -p script_logging

retriever=${1:-bge}
corpus_file=corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl
save_dir=/tmp/index_builder_test_${retriever}
query_file=corpus_datasets/generated_datasets/queries/queries_wikidata_test.jsonl
qrel_file=corpus_datasets/generated_datasets/qrels/trec_qrels_wikidata_test.txt

echo "=== Index Builder Test: $retriever ==="
echo "Corpus: $corpus_file"
echo "Save dir: $save_dir"
echo "Query limit: 20"
echo ""

python c2_corpus_creation/index_builder_test.py \
    --retrieval_method "$retriever" \
    --corpus_path "$corpus_file" \
    --save_dir "$save_dir" \
    --query_file "$query_file" \
    --qrel_file "$qrel_file" \
    --query_limit 20 \
    --use_fp16 \
    --max_length 512 \
    --batch_size 64

echo ""
echo "Results saved to $save_dir"
