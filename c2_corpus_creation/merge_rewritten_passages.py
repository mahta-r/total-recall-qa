"""
Script to merge rewritten passages into the main corpus.

This script:
1. Loads rewritten passages from multiple JSONL files
2. Groups multiple rewrites per passage ID
3. Streams through the large corpus file
4. Replaces passages with rewritten versions
5. If multiple rewrites exist for one passage, creates multiple entries with incremental IDs
   (e.g., 14607-0003-0, 14607-0003-1, 14607-0003-2)
"""

import argparse
from pathlib import Path

from src.rewrite_merger import (
    merge_corpus_with_rewrites, 
    check_passage_overlap, 
    RewriteMerger,
    update_qrels_and_create_mappings
)


def main():
    """Main function to run the merge process."""
    parser = argparse.ArgumentParser(
        description="Merge rewritten passages into the main corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all rewritten passages into the corpus
  python c2_corpus_creation/merge_rewritten_passages.py \
    --corpus corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl \
    --output corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl \
    --rewrites corpus_datasets/dataset_creation_heydar/qald10/passage_rewrites_v2_qald10.jsonl \
               corpus_datasets/dataset_creation_heydar/quest/test/passage_rewrites_v2_test_quest.jsonl \
               corpus_datasets/dataset_creation_heydar/quest/train/passage_rewrites_v2_train_quest.jsonl \
               corpus_datasets/dataset_creation_heydar/quest/val/passage_rewrites_v2_val_quest.jsonl \
               corpus_datasets/dataset_creation_mahta/passage_rewrites.jsonl
        """
    )
    
    parser.add_argument(
        '--corpus',
        type=str,
        required=True,
        help='Path to the original corpus JSONL file'
    )
    
    parser.add_argument(
        '--rewrites',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to rewritten passage JSONL files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output merged corpus JSONL file'
    )
    
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=10000,
        help='Number of lines to buffer before writing (default: 10000)'
    )
    
    parser.add_argument(
        '--check-overlap',
        action='store_true',
        help='Check for overlapping passage IDs between files before merging'
    )
    
    parser.add_argument(
        '--overlap-only',
        action='store_true',
        help='Only check overlap and exit without merging'
    )
    
    parser.add_argument(
        '--update-qrels',
        action='store_true',
        help='Update qrel files with new passage IDs'
    )
    
    args = parser.parse_args()
    
    # Check overlap if requested
    if args.check_overlap or args.overlap_only:
        print("\nChecking for overlapping passage IDs...\n")
        overlap_info = check_passage_overlap(args.rewrites, verbose=True)
        
        if args.overlap_only:
            print(f"\n✓ Overlap check completed!")
            return
        else:
            print()
    
    # Merge corpus with rewrites
    merger = RewriteMerger(verbose=True)
    merger.load_rewritten_passages(args.rewrites)
    merger.merge_corpus(
        corpus_file=args.corpus,
        output_file=args.output,
        buffer_size=args.buffer_size
    )
    
    print(f"\n✓ Merge completed successfully!")
    print(f"  Output saved to: {args.output}")
    
    # Update qrel files if requested
    if args.update_qrels:
        # Define rewrite configs with qrel files
        rewrite_configs = []
        
        for rewrite_file in args.rewrites:
            rewrite_path = Path(rewrite_file)
            config = {'rewrite_file': rewrite_file}
            
            # Check if this is a heydar dataset with qrel file
            if 'dataset_creation_heydar' in rewrite_file:
                # Construct qrel file path
                qrel_filename = rewrite_path.name.replace('passage_rewrites_v2_', 'qrels_v2_').replace('.jsonl', '.txt')
                qrel_file = str(rewrite_path.parent / qrel_filename)
                if Path(qrel_file).exists():
                    config['qrel_file'] = qrel_file
                    config['qrel_output'] = qrel_file.replace('.txt', '_rewritten.txt')
            
            # Check if this is mahta dataset (create mapping file)
            if 'dataset_creation_mahta' in rewrite_file:
                config['create_mapping'] = True
                config['mapping_file'] = str(rewrite_path.parent / 'passage_id_mapping.txt')
            
            rewrite_configs.append(config)
        
        # Update qrels and create mappings
        update_qrels_and_create_mappings(rewrite_configs, merger, verbose=True)


if __name__ == '__main__':
    main()
