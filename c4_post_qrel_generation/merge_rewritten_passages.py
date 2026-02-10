"""
Script to merge rewritten passages into the main corpus.

Reads a single rewrite file (output of c3_qrel_generation/fix_duplication_rewrite.py)
and replaces each original passage in the corpus with its rewritten version (1:1;
passage IDs unchanged). Reports corpus length before and after merge (they must be the same).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path

from c4_post_qrel_generation.src.rewrite_merger import load_rewrites_as_map, merge_corpus_replace_only


def count_corpus_lines(path: str) -> int:
    """Count non-empty lines in a JSONL corpus file."""
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    """Main function to run the merge process."""
    parser = argparse.ArgumentParser(
        description="Merge rewritten passages into the main corpus (1:1 replacement)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--corpus', type=str, default="corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl", help='Path to the original corpus JSONL file')
    parser.add_argument('--rewrites', type=str, default="corpus_datasets/passage_rewrites_unified.jsonl", help='Path to the single rewrite JSONL file (output of fix_duplication_rewrite.py)')
    parser.add_argument('--output', type=str, default="corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl", help='Path to output merged corpus JSONL file')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Number of lines to buffer before writing (default: 10000)')

    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {args.corpus}")
    length_before = count_corpus_lines(args.corpus)
    print(f"Corpus length before merge: {length_before:,}")

    print("Loading rewritten passages...")
    rewrites_map = load_rewrites_as_map([args.rewrites], verbose=True)

    print("\nMerging corpus (replacing original passages with rewrites)...")
    merge_corpus_replace_only(
        corpus_file=args.corpus,
        output_file=args.output,
        rewrites_map=rewrites_map,
        buffer_size=args.buffer_size,
        verbose=True,
    )

    length_after = count_corpus_lines(args.output)
    print(f"\nCorpus length after merge:  {length_after:,}")
    if length_before != length_after:
        raise RuntimeError(f"Corpus length changed: before={length_before}, after={length_after}. They must be the same.")
    print("Corpus length unchanged (same before and after).")
    print(f"\n✓ Merge completed. Output: {args.output}")


if __name__ == '__main__':
    main()
    
# python c4_post_qrel_generation/merge_rewritten_passages.py
