"""
Script to update qrel files with new passage IDs after rewriting.

This script:
1. Loads rewritten passages to build passage ID mappings
2. Updates qrel files with new passage IDs (creating duplicates for multiple rewrites)
3. Creates ID mapping files for datasets without qrels
"""

import argparse
from pathlib import Path
from rewrite_merger import RewriteMerger, update_qrels_and_create_mappings


def main():
    """Main function to update qrel files."""
    parser = argparse.ArgumentParser(
        description="Update qrel files with new passage IDs from rewritten passages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update qrels for heydar datasets (auto-detect qrel files)
  python c2_corpus_creation/src/update_qrels.py \\
    --rewrites corpus_datasets/dataset_creation_heydar/qald10/passage_rewrites_v2_qald10.jsonl \\
               corpus_datasets/dataset_creation_heydar/quest/test/passage_rewrites_v2_test_quest.jsonl
  
  # Create mapping file for mahta dataset
  python c2_corpus_creation/src/update_qrels.py \\
    --rewrites corpus_datasets/dataset_creation_mahta/passage_rewrites.jsonl \\
    --create-mappings
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
        '--create-mappings',
        action='store_true',
        help='Create ID mapping files for datasets without qrels'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("UPDATING QREL FILES WITH REWRITTEN PASSAGE IDS")
    print("="*60)
    print()
    
    # Load rewritten passages to build ID mappings
    print("Step 1: Loading rewritten passages...")
    merger = RewriteMerger(verbose=True)
    merger.load_rewritten_passages(args.rewrites)
    
    # Build rewrite configs
    rewrite_configs = []
    
    for rewrite_file in args.rewrites:
        rewrite_path = Path(rewrite_file)
        config = {'rewrite_file': rewrite_file}
        
        # Check if this is a heydar dataset with qrel file
        if 'dataset_creation_heydar' in rewrite_file:
            # Construct qrel file path by replacing passage_rewrites with qrels
            qrel_filename = rewrite_path.name.replace('passage_rewrites_v2_', 'qrels_v2_').replace('.jsonl', '.txt')
            qrel_file = str(rewrite_path.parent / qrel_filename)
            
            if Path(qrel_file).exists():
                config['qrel_file'] = qrel_file
                config['qrel_output'] = qrel_file.replace('.txt', '_rewritten.txt')
                print(f"  Found qrel file: {Path(qrel_file).name}")
            else:
                print(f"  Warning: Qrel file not found: {qrel_filename}")
        
        # Check if this is mahta dataset or user requested mappings
        if 'dataset_creation_mahta' in rewrite_file or args.create_mappings:
            config['create_mapping'] = True
            config['mapping_file'] = str(rewrite_path.parent / 'passage_id_mapping.txt')
            print(f"  Will create mapping file: passage_id_mapping.txt")
        
        rewrite_configs.append(config)
    
    print()
    print("Step 2: Processing qrel files and creating mappings...")
    
    # Update qrels and create mappings
    stats = update_qrels_and_create_mappings(rewrite_configs, merger, verbose=True)
    
    print()
    print("✓ Qrel update completed successfully!")
    print(f"  Qrel files updated: {stats['qrels_updated']}")
    print(f"  Mapping files created: {stats['mappings_created']}")


if __name__ == '__main__':
    main()
