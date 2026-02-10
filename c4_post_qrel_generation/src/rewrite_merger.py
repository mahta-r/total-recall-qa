"""
Utility module for merging rewritten passages into corpus.

This module provides reusable functions for loading rewritten passages
and merging them into a corpus file.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

from tqdm import tqdm


def check_passage_overlap(rewrite_files: List[str], verbose: bool = True) -> Dict[str, any]:
    """
    Check for overlapping passage IDs across multiple rewrite files.
    
    Args:
        rewrite_files: List of paths to rewritten passage JSONL files
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary containing:
        - 'file_passage_ids': Dict mapping file path to set of passage IDs
        - 'passage_to_files': Dict mapping passage ID to list of files containing it
        - 'overlapping_passages': Set of passage IDs that appear in multiple files
        - 'overlap_count': Number of passages that appear in multiple files
        - 'total_unique_passages': Total number of unique passage IDs across all files
        
    Example:
        >>> overlap_info = check_passage_overlap([
        ...     'rewrites1.jsonl',
        ...     'rewrites2.jsonl'
        ... ])
        >>> print(f"Found {overlap_info['overlap_count']} overlapping passages")
    """
    if verbose:
        print("="*60)
        print("CHECKING PASSAGE ID OVERLAP ACROSS FILES")
        print("="*60)
        print()
    
    file_passage_ids: Dict[str, Set[str]] = {}
    passage_to_files: Dict[str, List[str]] = defaultdict(list)
    file_rewrite_counts: Dict[str, int] = {}
    
    # Load passage IDs from each file
    for rewrite_file in rewrite_files:
        rewrite_path = Path(rewrite_file)
        if not rewrite_path.exists():
            if verbose:
                print(f"Warning: File not found: {rewrite_file}")
            continue
        
        if verbose:
            print(f"Loading passage IDs from: {rewrite_file}")
        
        passage_ids = set()
        rewrite_count = 0
        
        with open(rewrite_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    passage_id = data.get('passage_id') or data.get('passage', {}).get('id')
                    if passage_id:
                        passage_ids.add(passage_id)
                        rewrite_count += 1
        
        # Add file to passage_to_files mapping (once per passage per file)
        for passage_id in passage_ids:
            passage_to_files[passage_id].append(rewrite_file)
        
        file_passage_ids[rewrite_file] = passage_ids
        file_rewrite_counts[rewrite_file] = rewrite_count
        
        if verbose:
            print(f"  - Found {len(passage_ids)} unique passage IDs ({rewrite_count} total rewrites)")
    
    # Find overlapping passages
    overlapping_passages = {
        pid for pid, files in passage_to_files.items() if len(files) > 1
    }
    
    # Calculate statistics
    all_unique_passages = set(passage_to_files.keys())
    
    if verbose:
        print()
        print("="*60)
        print("OVERLAP ANALYSIS")
        print("="*60)
        print(f"Total files processed:          {len(file_passage_ids)}")
        print(f"Total unique passage IDs:       {len(all_unique_passages)}")
        print(f"Overlapping passage IDs:        {len(overlapping_passages)}")
        
        if overlapping_passages:
            print()
            print("Files with overlaps:")
            
            # Create file pairs with overlap counts
            file_list = list(file_passage_ids.keys())
            for i, file1 in enumerate(file_list):
                for file2 in file_list[i+1:]:
                    overlap = file_passage_ids[file1] & file_passage_ids[file2]
                    if overlap:
                        print(f"\n  {Path(file1).name} ∩ {Path(file2).name}")
                        print(f"    Overlapping passages: {len(overlap)}")
            
            # Show some example overlapping passages
            print()
            print("Examples of overlapping passage IDs:")
            for pid in list(overlapping_passages)[:10]:
                files_with_pid = [Path(f).name for f in passage_to_files[pid]]
                print(f"  - {pid}: appears in {len(files_with_pid)} files")
                for fname in files_with_pid:
                    print(f"      • {fname}")
            
            if len(overlapping_passages) > 10:
                print(f"  ... and {len(overlapping_passages) - 10} more")
        else:
            print("\n✓ No overlapping passage IDs found - all passages are unique across files!")
        
        print("="*60)
    
    return {
        'file_passage_ids': file_passage_ids,
        'passage_to_files': dict(passage_to_files),
        'overlapping_passages': overlapping_passages,
        'overlap_count': len(overlapping_passages),
        'total_unique_passages': len(all_unique_passages),
        'file_rewrite_counts': file_rewrite_counts
    }


def load_rewrites_as_map(rewrite_files: List[str], verbose: bool = True) -> Dict[str, Dict]:
    """
    Load rewritten passages from JSONL files into a single passage_id -> passage dict.
    Assumes check_passage_overlap has been run and there are no overlaps; each passage_id
    appears at most once. If a passage_id appears in multiple files, the last occurrence wins.

    Args:
        rewrite_files: Paths to rewritten passage JSONL files
        verbose: Whether to print progress

    Returns:
        Dict mapping passage_id -> rewritten passage (the passage dict to write to corpus)
    """
    rewrites_map: Dict[str, Dict] = {}
    for rewrite_file in rewrite_files:
        rewrite_path = Path(rewrite_file)
        if not rewrite_path.exists():
            if verbose:
                print(f"Warning: File not found: {rewrite_file}")
            continue
        if verbose:
            print(f"  Loading: {rewrite_file}")
        with open(rewrite_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    passage_id = data.get('passage_id') or data.get('passage', {}).get('id')
                    if passage_id and 'rewritten_passage' in data:
                        rewrites_map[passage_id] = data['rewritten_passage']
    if verbose:
        print(f"  Loaded {len(rewrites_map)} passage rewrites")
    return rewrites_map

def merge_corpus_replace_only(
    corpus_file: str,
    output_file: str,
    rewrites_map: Dict[str, Dict],
    buffer_size: int = 10000,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Stream the corpus and replace each passage with its rewritten version when present.
    Passage IDs are unchanged (1:1 replacement). Use after check_passage_overlap confirms no overlaps.

    Args:
        corpus_file: Path to the original corpus JSONL file
        output_file: Path to output merged corpus JSONL file
        rewrites_map: Dict mapping passage_id -> rewritten passage dict
        buffer_size: Write buffer size
        verbose: Whether to show progress

    Returns:
        Dict with keys: total_passages, replaced_passages, unchanged_passages
    """
    corpus_path = Path(corpus_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {'total_passages': 0, 'replaced_passages': 0, 'unchanged_passages': 0}
    write_buffer = []
    file_size = corpus_path.stat().st_size

    with open(corpus_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc='Merging corpus', disable=not verbose)
            try:
                for line in f_in:
                    pbar.update(len(line.encode('utf-8')))
                    if not line.strip():
                        continue
                    stats['total_passages'] += 1
                    passage = json.loads(line)
                    passage_id = passage.get('id')
                    if passage_id and passage_id in rewrites_map:
                        write_buffer.append(
                            json.dumps(rewrites_map[passage_id], ensure_ascii=False) + '\n'
                        )
                        stats['replaced_passages'] += 1
                    else:
                        write_buffer.append(line)
                        stats['unchanged_passages'] += 1
                    if len(write_buffer) >= buffer_size:
                        f_out.writelines(write_buffer)
                        write_buffer = []
                if write_buffer:
                    f_out.writelines(write_buffer)
            finally:
                pbar.close()

    if verbose:
        print(f"Total passages: {stats['total_passages']:,}")
        print(f"Replaced:       {stats['replaced_passages']:,}")
        print(f"Unchanged:      {stats['unchanged_passages']:,}")
    return stats
