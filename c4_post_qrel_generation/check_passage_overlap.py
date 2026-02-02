"""
Script to check for overlapping passage IDs across rewritten passage files.

This script analyzes multiple rewritten passage files and reports:
- Which passage IDs appear in multiple files
- Statistics about overlaps between file pairs
- Examples of overlapping passages
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from c4_post_qrel_generation.src.rewrite_merger import check_passage_overlap


def main():
    """Main function to check passage overlaps."""
    parser = argparse.ArgumentParser(
        description="Check for overlapping passage IDs across rewritten passage files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# Check overlap across all rewrite files
python c2_corpus_creation/src/check_passage_overlap.py \
    --rewrites corpus_datasets/dataset_creation_heydar/qald10/passage_rewrites_qald10.jsonl \
        corpus_datasets/dataset_creation_heydar/quest/test/passage_rewrites_test_quest.jsonl \
        corpus_datasets/dataset_creation_heydar/quest/train/passage_rewrites_train_quest.jsonl \
        corpus_datasets/dataset_creation_heydar/quest/val/passage_rewrites_val_quest.jsonl \
        corpus_datasets/dataset_creation_mahta/passage_rewrites.jsonl
  
  # Check overlap with quiet output (just statistics)
  python check_passage_overlap.py --rewrites file1.jsonl file2.jsonl --quiet
        """
    )
    
    parser.add_argument(
        '--rewrites',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to rewritten passage JSONL files'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output, show only summary statistics'
    )
    
    args = parser.parse_args()
    
    # Check for overlaps
    overlap_info = check_passage_overlap(args.rewrites, verbose=not args.quiet)
    
    if args.quiet:
        # Print minimal summary
        print(f"Total unique passages: {overlap_info['total_unique_passages']}")
        print(f"Overlapping passages: {overlap_info['overlap_count']}")
        if overlap_info['overlap_count'] > 0:
            overlap_pct = (overlap_info['overlap_count'] / overlap_info['total_unique_passages']) * 100
            print(f"Overlap percentage: {overlap_pct:.2f}%")


if __name__ == '__main__':
    main()
