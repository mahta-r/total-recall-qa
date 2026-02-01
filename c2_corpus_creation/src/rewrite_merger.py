"""
Utility module for merging rewritten passages into corpus.

This module provides reusable functions for loading rewritten passages
and merging them into a corpus file.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Callable, Set, Tuple
from tqdm import tqdm


class RewriteMerger:
    """Class to handle merging of rewritten passages into corpus."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize RewriteMerger.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.rewritten_passages: Dict[str, List[Dict]] = defaultdict(list)
        self.passage_id_mapping: Dict[str, List[str]] = {}  # Maps old_id -> [new_id1, new_id2, ...]
        self.rewrite_file_index: Dict[str, Dict[str, int]] = {}  # Maps rewrite_file -> {passage_id: index}
        self.stats = {
            'total_passages': 0,
            'replaced_passages': 0,
            'new_passage_variants': 0,
            'unchanged_passages': 0
        }
    
    def load_rewritten_passages(self, rewrite_files: List[str]) -> 'RewriteMerger':
        """
        Load rewritten passages from multiple JSONL files.
        
        Args:
            rewrite_files: List of paths to rewritten passage JSONL files
            
        Returns:
            Self for method chaining
        """
        if self.verbose:
            print("Loading rewritten passages...")
        
        for rewrite_file in rewrite_files:
            rewrite_path = Path(rewrite_file)
            if not rewrite_path.exists():
                if self.verbose:
                    print(f"Warning: File not found: {rewrite_file}")
                continue
            
            if self.verbose:
                print(f"  Loading: {rewrite_file}")
            
            # Track which passages are in this file and their order
            passage_order = {}
            
            with open(rewrite_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        passage_id = data.get('passage_id') or data.get('passage', {}).get('id')
                        if passage_id:
                            # Track the index for this passage in the global rewritten_passages list
                            current_index = len(self.rewritten_passages[passage_id])
                            passage_order[passage_id] = current_index
                            self.rewritten_passages[passage_id].append(data['rewritten_passage'])
            
            # Store the mapping of passage_id to its index for this file
            self.rewrite_file_index[rewrite_file] = passage_order
        
        if self.verbose:
            self._print_load_statistics()
        
        # Build ID mappings after loading
        self._build_id_mappings()
        
        return self
    
    def _build_id_mappings(self):
        """Build passage ID mappings based on loaded rewritten passages."""
        self.passage_id_mapping = {}
        
        for passage_id, rewrites in self.rewritten_passages.items():
            if len(rewrites) == 1:
                # Single rewrite keeps same ID
                self.passage_id_mapping[passage_id] = [passage_id]
            else:
                # Multiple rewrites get incremental suffixes
                new_ids = [f"{passage_id}-{idx}" for idx in range(len(rewrites))]
                self.passage_id_mapping[passage_id] = new_ids
    
    def _print_load_statistics(self):
        """Print statistics about loaded rewritten passages."""
        total_passages = len(self.rewritten_passages)
        total_rewrites = sum(len(rewrites) for rewrites in self.rewritten_passages.values())
        passages_with_multiple = sum(1 for rewrites in self.rewritten_passages.values() if len(rewrites) > 1)
        
        print(f"\nLoaded {total_rewrites} rewrites for {total_passages} unique passages")
        print(f"Passages with multiple rewrites: {passages_with_multiple}")
        if passages_with_multiple > 0:
            max_rewrites = max(len(rewrites) for rewrites in self.rewritten_passages.values())
            print(f"Maximum rewrites for a single passage: {max_rewrites}")
    
    def merge_corpus(
        self,
        corpus_file: str,
        output_file: str,
        buffer_size: int = 10000,
        transform_fn: Optional[Callable[[Dict], Dict]] = None
    ) -> Dict[str, int]:
        """
        Stream through corpus and replace passages with rewritten versions.
        
        Args:
            corpus_file: Path to the original corpus JSONL file
            output_file: Path to output merged corpus JSONL file
            buffer_size: Number of lines to buffer before writing
            transform_fn: Optional function to transform each passage before writing
            
        Returns:
            Dictionary with merge statistics
        """
        corpus_path = Path(corpus_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\nProcessing corpus: {corpus_file}")
            print(f"Output will be saved to: {output_file}")
        
        # Reset statistics
        self.stats = {
            'total_passages': 0,
            'replaced_passages': 0,
            'new_passage_variants': 0,
            'unchanged_passages': 0
        }
        
        # Get file size for progress bar
        file_size = corpus_path.stat().st_size
        write_buffer = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                # Create progress bar based on file size
                pbar = tqdm(total=file_size, unit='B', unit_scale=True, 
                           desc="Processing corpus", disable=not self.verbose)
                
                try:
                    for line in f_in:
                        # Update progress bar
                        pbar.update(len(line.encode('utf-8')))
                        
                        if not line.strip():
                            continue
                        
                        self.stats['total_passages'] += 1
                        
                        # Parse original passage
                        original_passage = json.loads(line)
                        passage_id = original_passage['id']
                        
                        # Check if this passage has rewrites
                        if passage_id in self.rewritten_passages:
                            rewrites = self.rewritten_passages[passage_id]
                            self.stats['replaced_passages'] += 1
                            
                            # If single rewrite, keep the same ID
                            if len(rewrites) == 1:
                                passage_to_write = rewrites[0]
                                if transform_fn:
                                    passage_to_write = transform_fn(passage_to_write)
                                write_buffer.append(json.dumps(passage_to_write, ensure_ascii=False) + '\n')
                                # Track ID mapping (single rewrite keeps same ID)
                                self.passage_id_mapping[passage_id] = [passage_id]
                            else:
                                # Multiple rewrites: add incremental suffix
                                new_ids = []
                                for idx, rewrite in enumerate(rewrites):
                                    # Create new ID with suffix
                                    new_passage = rewrite.copy()
                                    new_id = f"{passage_id}-{idx}"
                                    new_passage['id'] = new_id
                                    new_ids.append(new_id)
                                    if transform_fn:
                                        new_passage = transform_fn(new_passage)
                                    write_buffer.append(json.dumps(new_passage, ensure_ascii=False) + '\n')
                                    self.stats['new_passage_variants'] += 1
                                # Track ID mapping (multiple rewrites)
                                self.passage_id_mapping[passage_id] = new_ids
                        else:
                            # No rewrite, keep original
                            if transform_fn:
                                original_passage = transform_fn(original_passage)
                                write_buffer.append(json.dumps(original_passage, ensure_ascii=False) + '\n')
                            else:
                                write_buffer.append(line)
                            self.stats['unchanged_passages'] += 1
                        
                        # Write buffer if it reaches the size limit
                        if len(write_buffer) >= buffer_size:
                            f_out.writelines(write_buffer)
                            write_buffer = []
                    
                    # Write remaining buffer
                    if write_buffer:
                        f_out.writelines(write_buffer)
                
                finally:
                    pbar.close()
        
        if self.verbose:
            self._print_merge_statistics()
        
        return self.stats
    
    def _print_merge_statistics(self):
        """Print merge statistics."""
        print("\n" + "="*60)
        print("MERGE STATISTICS")
        print("="*60)
        print(f"Total passages processed:     {self.stats['total_passages']:,}")
        print(f"Passages with rewrites:       {self.stats['replaced_passages']:,}")
        print(f"New passage variants created: {self.stats['new_passage_variants']:,}")
        print(f"Unchanged passages:           {self.stats['unchanged_passages']:,}")
        print(f"\nTotal passages in new corpus: "
              f"{self.stats['unchanged_passages'] + self.stats['replaced_passages'] + self.stats['new_passage_variants']:,}")
        print("="*60)
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()
    
    def has_rewrite(self, passage_id: str) -> bool:
        """
        Check if a passage ID has rewrites.
        
        Args:
            passage_id: The passage ID to check
            
        Returns:
            True if the passage has rewrites
        """
        return passage_id in self.rewritten_passages
    
    def get_rewrites(self, passage_id: str) -> List[Dict]:
        """
        Get all rewrites for a passage ID.
        
        Args:
            passage_id: The passage ID to get rewrites for
            
        Returns:
            List of rewritten passages (empty list if no rewrites)
        """
        return self.rewritten_passages.get(passage_id, [])
    
    def get_passage_id_mapping(self) -> Dict[str, List[str]]:
        """
        Get the mapping of original passage IDs to new passage IDs.
        
        Returns:
            Dictionary mapping old_id -> list of new_ids
        """
        return self.passage_id_mapping.copy()
    
    def update_qrel_file(self, input_qrel: str, output_qrel: str, rewrite_file: str) -> Dict[str, int]:
        """
        Update a qrel file with new passage IDs based on rewrites from a specific file.
        
        Each qrel file only gets the rewrite that corresponds to its own dataset,
        maintaining the same number of lines.
        
        Args:
            input_qrel: Path to input qrel file
            output_qrel: Path to output updated qrel file
            rewrite_file: Path to rewrite file to get mappings from
            
        Returns:
            Dictionary with update statistics
        """
        rewrite_path = Path(rewrite_file)
        if not rewrite_path.exists():
            if self.verbose:
                print(f"Warning: Rewrite file not found: {rewrite_file}")
            return {'updated': 0, 'unchanged': 0}
        
        # Get the passage ID to index mapping for this specific rewrite file
        if rewrite_file not in self.rewrite_file_index:
            if self.verbose:
                print(f"Warning: No index found for rewrite file: {rewrite_file}")
            return {'updated': 0, 'unchanged': 0}
        
        file_passage_index = self.rewrite_file_index[rewrite_file]
        
        stats = {'updated': 0, 'unchanged': 0}
        
        with open(input_qrel, 'r', encoding='utf-8') as f_in:
            with open(output_qrel, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        query_id, zero, passage_id, relevance = parts[0], parts[1], parts[2], parts[3]
                        
                        # Check if this passage was rewritten in this specific file
                        if passage_id in file_passage_index and passage_id in self.passage_id_mapping:
                            # Get the specific index for this file
                            index = file_passage_index[passage_id]
                            # Get the corresponding new ID
                            new_id = self.passage_id_mapping[passage_id][index]
                            f_out.write(f"{query_id} {zero} {new_id} {relevance}\n")
                            stats['updated'] += 1
                        else:
                            # Keep original line unchanged
                            f_out.write(line)
                            stats['unchanged'] += 1
        
        if self.verbose:
            print(f"  Updated: {stats['updated']} entries")
            print(f"  Unchanged: {stats['unchanged']} entries")
            print(f"  Total: {stats['updated'] + stats['unchanged']} entries (same as original)")
        
        return stats


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


def create_id_mapping_file(rewrite_file: str, output_file: str, passage_id_mapping: Dict[str, List[str]], 
                          rewrite_file_index: Dict[str, Dict[str, int]]) -> int:
    """
    Create a mapping file showing old_id -> new_id changes for this specific dataset.
    
    Args:
        rewrite_file: Path to rewrite file to get passage IDs from
        output_file: Path to output mapping file
        passage_id_mapping: Dictionary mapping old_id to list of new_ids
        rewrite_file_index: Dictionary mapping rewrite_file to {passage_id: index}
        
    Returns:
        Number of mappings written
    """
    # Get passage IDs from this rewrite file
    rewrite_path = Path(rewrite_file)
    if not rewrite_path.exists():
        print(f"Warning: Rewrite file not found: {rewrite_file}")
        return 0
    
    if rewrite_file not in rewrite_file_index:
        print(f"Warning: No index found for rewrite file: {rewrite_file}")
        return 0
    
    file_passage_index = rewrite_file_index[rewrite_file]
    
    # Write mapping file - only the specific mapping for this dataset
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for passage_id in sorted(file_passage_index.keys()):
            if passage_id in passage_id_mapping:
                # Get the specific index for this file
                index = file_passage_index[passage_id]
                new_id = passage_id_mapping[passage_id][index]
                
                if new_id != passage_id:  # Only write if ID changed
                    f.write(f"{passage_id} -> {new_id}\n")
                    count += 1
    
    return count


def update_qrels_and_create_mappings(
    rewrite_configs: List[Dict[str, str]],
    merger: RewriteMerger,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Update qrel files and create mapping files for rewritten passages.
    
    Args:
        rewrite_configs: List of dicts with keys: 'rewrite_file', 'qrel_file' (optional), 'mapping_file' (optional)
        merger: RewriteMerger instance with loaded passage ID mappings
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'qrels_updated': 0,
        'mappings_created': 0,
        'total_qrel_entries_updated': 0,
        'qrel_files_verified_same_length': 0
    }
    
    if verbose:
        print("\n" + "="*60)
        print("UPDATING QREL FILES AND CREATING MAPPINGS")
        print("="*60)
    
    for config in rewrite_configs:
        rewrite_file = config['rewrite_file']
        rewrite_name = Path(rewrite_file).stem
        
        if verbose:
            print(f"\nProcessing: {Path(rewrite_file).name}")
        
        # Update qrel file if provided
        if 'qrel_file' in config and config['qrel_file']:
            qrel_input = config['qrel_file']
            qrel_output = config.get('qrel_output', qrel_input.replace('.txt', '_rewritten.txt'))
            
            if Path(qrel_input).exists():
                if verbose:
                    print(f"  Updating qrel: {Path(qrel_input).name} -> {Path(qrel_output).name}")
                qrel_stats = merger.update_qrel_file(qrel_input, qrel_output, rewrite_file)
                stats['qrels_updated'] += 1
                stats['total_qrel_entries_updated'] += qrel_stats['updated']
                
                # Verify same length
                original_count = qrel_stats['updated'] + qrel_stats['unchanged']
                if verbose:
                    print(f"  ✓ Verified: Original and rewritten qrels have same length")
                stats['qrel_files_verified_same_length'] += 1
            else:
                if verbose:
                    print(f"  Warning: Qrel file not found: {qrel_input}")
        
        # Create mapping file if requested
        if config.get('create_mapping', False):
            mapping_file = config.get('mapping_file', 
                                     str(Path(rewrite_file).parent / 'passage_id_mapping.txt'))
            if verbose:
                print(f"  Creating mapping file: {Path(mapping_file).name}")
            
            count = create_id_mapping_file(rewrite_file, mapping_file, 
                                         merger.get_passage_id_mapping(),
                                         merger.rewrite_file_index)
            if count > 0:
                stats['mappings_created'] += 1
                if verbose:
                    print(f"    Written {count} ID mappings (only for this dataset)")
            else:
                if verbose:
                    print(f"    No ID changes to write")
    
    if verbose:
        print("\n" + "="*60)
        print(f"QREL files updated: {stats['qrels_updated']}")
        print(f"Mapping files created: {stats['mappings_created']}")
        print(f"Total qrel entries updated: {stats['total_qrel_entries_updated']}")
        print(f"✓ All qrel files maintain same length as originals")
        print("="*60)
    
    return stats


def merge_corpus_with_rewrites(
    corpus_file: str,
    rewrite_files: List[str],
    output_file: str,
    buffer_size: int = 10000,
    verbose: bool = True,
    transform_fn: Optional[Callable[[Dict], Dict]] = None
) -> Dict[str, int]:
    """
    Convenience function to merge rewritten passages into corpus in one call.
    
    Args:
        corpus_file: Path to the original corpus JSONL file
        rewrite_files: List of paths to rewritten passage JSONL files
        output_file: Path to output merged corpus JSONL file
        buffer_size: Number of lines to buffer before writing
        verbose: Whether to print progress information
        transform_fn: Optional function to transform each passage before writing
        
    Returns:
        Dictionary with merge statistics
        
    Example:
        >>> stats = merge_corpus_with_rewrites(
        ...     corpus_file='corpus/original.jsonl',
        ...     rewrite_files=['rewrites1.jsonl', 'rewrites2.jsonl'],
        ...     output_file='corpus/merged.jsonl'
        ... )
        >>> print(f"Processed {stats['total_passages']} passages")
    """
    merger = RewriteMerger(verbose=verbose)
    merger.load_rewritten_passages(rewrite_files)
    return merger.merge_corpus(corpus_file, output_file, buffer_size, transform_fn)
