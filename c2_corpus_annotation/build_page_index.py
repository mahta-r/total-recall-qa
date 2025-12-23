"""
Build Page-to-Passage Index for Efficient Corpus Lookup

This script creates an index mapping Wikipedia page IDs to line numbers in the corpus.
This allows O(1) lookup instead of O(n) full corpus scan for each entity.

Usage:
    python c2_corpus_annotation/build_page_index.py --corpus_jsonl /path/to/corpus.jsonl --output_index page_index.json

Output format:
    {
        "pageid1": [0, 1, 2, 3],  # Line numbers where passages for this page appear
        "pageid2": [4, 5, 6],
        ...
    }
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict


def build_page_to_line_index(corpus_jsonl_path: str, show_progress: bool = True) -> Dict[str, List[int]]:
    """
    Build an index mapping page IDs to line numbers in the corpus.

    Args:
        corpus_jsonl_path: Path to the corpus JSONL file
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping page IDs to lists of line numbers
    """
    page_to_lines = defaultdict(list)

    # First pass: count lines for progress bar
    if show_progress:
        print("Counting lines in corpus...")
        with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"Total lines: {total_lines:,}")
    else:
        total_lines = None

    # Second pass: build index
    print("Building page-to-passage index...")
    with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
        iterator = enumerate(f)
        if show_progress and total_lines:
            iterator = tqdm(iterator, total=total_lines, desc="Indexing passages")

        for line_num, line in iterator:
            try:
                passage = json.loads(line.strip())
                passage_id = passage.get('id', '')

                # Extract page ID from passage ID (format: "pageid-0000")
                if '-' in passage_id:
                    pageid = passage_id.split('-')[0]
                    page_to_lines[pageid].append(line_num)

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue

    # Convert defaultdict to regular dict for JSON serialization
    page_to_lines = dict(page_to_lines)

    return page_to_lines


def save_index(index: Dict[str, List[int]], output_path: str):
    """
    Save the index to a JSON file.

    Args:
        index: The page-to-lines mapping
        output_path: Path to save the JSON file
    """
    print(f"Saving index to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index, f)
    print("Index saved successfully!")


def print_index_stats(index: Dict[str, List[int]]):
    """
    Print statistics about the index.

    Args:
        index: The page-to-lines mapping
    """
    total_pages = len(index)
    total_passages = sum(len(lines) for lines in index.values())
    avg_passages_per_page = total_passages / total_pages if total_pages > 0 else 0

    passages_per_page = [len(lines) for lines in index.values()]
    max_passages = max(passages_per_page) if passages_per_page else 0
    min_passages = min(passages_per_page) if passages_per_page else 0

    print("\n" + "="*80)
    print("INDEX STATISTICS")
    print("="*80)
    print(f"Total unique pages: {total_pages:,}")
    print(f"Total passages indexed: {total_passages:,}")
    print(f"Average passages per page: {avg_passages_per_page:.2f}")
    print(f"Max passages for a single page: {max_passages}")
    print(f"Min passages for a single page: {min_passages}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Build page-to-passage index for efficient corpus lookup"
    )

    parser.add_argument(
        "--corpus_jsonl",
        type=str,
        required=True,
        help="Path to corpus JSONL file"
    )

    parser.add_argument(
        "--output_index",
        type=str,
        default=None,
        help="Path to save the index JSON file (default: same directory as corpus with .index.json suffix)"
    )

    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar"
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output_index is None:
        corpus_path = Path(args.corpus_jsonl)
        args.output_index = str(corpus_path.parent / f"{corpus_path.stem}.index.json")

    # Create output directory if needed
    output_dir = Path(args.output_index).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Page-to-Passage Index Builder")
    print("="*80)
    print(f"Corpus: {args.corpus_jsonl}")
    print(f"Output index: {args.output_index}")
    print("="*80)
    print()

    # Build index
    index = build_page_to_line_index(
        args.corpus_jsonl,
        show_progress=not args.no_progress
    )

    # Save index
    save_index(index, args.output_index)

    # Print statistics
    print_index_stats(index)

    print(f"\nIndex file: {args.output_index}")
    print(f"You can now use this index with qrel_generation.py using:")
    print(f"  --page2passage_mapping {args.output_index}")


if __name__ == "__main__":
    main()
