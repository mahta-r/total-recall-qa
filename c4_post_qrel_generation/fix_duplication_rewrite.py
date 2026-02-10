"""
Fix Duplication in Rewritten Passages

This script consolidates duplicate passages across multiple datasets by:
1. Reading all rewrite files in order: mahta -> qald10 -> val_quest -> test_quest -> train_quest
2. For within-dataset duplicates: keeping only the last occurrence (most updated)
3. For cross-dataset duplicates: rewriting on top of the latest version using the normal REWRITE_PROMPT
4. Outputting a single unified file where each passage_id appears exactly once with:
   - passage_id
   - original_passage (the first base version)
   - rewritten_passage (the latest version after all rewrites)
   - rewrite_history (list of all rewrite operations applied)

The script follows the same approach as load_existing_rewrites_if_exists:
each new rewrite is applied on top of the current latest version.
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions
from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file, encode_datetime
from c1_1_dataset_creation_mahta.wikidata.data_utils import format_time_for_prompt

# Import rewrite prompt
from c1_1_dataset_creation_mahta.query_generation.prompts.LLM_as_relevance_judge.prompt_rewrite import (
    REWRITE_PROMPT,
    REWRITE_EXPLANATION
)

# Import input template for rewrite
from c1_1_dataset_creation_mahta.query_generation.prompts.LLM_as_relevance_judge.prompt_combined import (
    PROPERTY_CHECK_INPUT_TEMPLATE
)


def read_rewrite_file(file_path: str) -> List[Dict]:
    """Read a rewrite JSONL file and return all entries."""
    return read_jsonl_from_file(file_path)


def get_all_occurrences_grouped(entries: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group entries by passage_id, preserving order of occurrences.
    
    Args:
        entries: List of rewrite entries
        
    Returns:
        Dict mapping passage_id to list of all entries with that passage_id (in order)
    """
    passage_dict = defaultdict(list)
    for entry in entries:
        passage_id = entry["rewritten_passage"]["id"]
        passage_dict[passage_id].append(entry)
    return dict(passage_dict)


def rewrite_passage_on_latest(
    current_passage: Dict,
    new_rewrite_entry: Dict,
    client: OpenAI,
    model: str,
    temperature: float
) -> str:
    """
    Apply a rewrite on top of the current latest passage version.
    
    This follows the same approach as the normal rewrite flow:
    - Take the latest rewritten passage
    - Apply a new rewrite on top of it using REWRITE_PROMPT
    
    Args:
        current_passage: The current latest passage dict (with 'id', 'title', 'section', 'contents')
        new_rewrite_entry: The new rewrite entry to apply
        client: OpenAI client
        model: Model name
        temperature: Temperature for generation
        
    Returns:
        New rewritten passage content
    """
    # Get metadata for the rewrite
    entity_name = new_rewrite_entry.get("entity_label", "")
    property_name = new_rewrite_entry.get("property_label", "")
    property_value = new_rewrite_entry.get("property_value", "")
    property_description = ""  # Not available in rewrite files, use empty string
    
    # Determine rewrite type (default to ADD if not present)
    rewrite_type = new_rewrite_entry.get("rewrite_type", "ADD")
    
    # Format time information if present
    shared_time = new_rewrite_entry.get("shared_time")
    if shared_time is not None:
        shared_time_str = format_time_for_prompt(shared_time=shared_time)
        statement_time = f"this statement is valid as of {shared_time_str}."
    else:
        statement_time = ""
    
    # Prepare input using the current passage (which may already be rewritten)
    input_instance = PROPERTY_CHECK_INPUT_TEMPLATE.format(
        entity_name=entity_name,
        property_name=property_name,
        property_description=property_description,
        property_value=property_value,
        time_of_statement=statement_time,
        passage_title=current_passage.get('title', ''),
        sections=" - ".join(current_passage.get('section', [])),
        passage=current_passage.get('contents', '')
    )
    
    # Prepare prompt with rewrite explanation
    prompt = REWRITE_PROMPT.format(
        rewrite_explanation=REWRITE_EXPLANATION[rewrite_type]
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_instance}
            ],
            temperature=temperature,
        )
        
        response = resp.choices[0].message.content.strip()
        response = re.sub(r'\s+', ' ', response).strip()
        
        return response
    
    except Exception as e:
        print(f"Error in LLM rewriting: {e}", file=sys.stderr)
        # Fallback: return current passage unchanged
        return current_passage.get('contents', '')


def create_unified_entry(passage_id: str, first_entry: Dict) -> Dict:
    """
    Create a unified entry structure for the output.
    
    Args:
        passage_id: The passage ID
        first_entry: The first rewrite entry for this passage
        
    Returns:
        Dict with structure: {
            "passage_id": str,
            "original_passage": Dict,
            "rewritten_passage": Dict,
            "rewrite_history": List[Dict]
        }
    """
    return {
        "passage_id": passage_id,
        "original_passage": first_entry["passage"],
        "rewritten_passage": first_entry["rewritten_passage"],
        "rewrite_history": [
            {
                "query_id": first_entry.get("query_id"),
                "entity_label": first_entry.get("entity_label"),
                "entity_qid": first_entry.get("entity_qid"),
                "property_label": first_entry.get("property_label"),
                "property_id": first_entry.get("property_id"),
                "property_value": first_entry.get("property_value"),
                "shared_time": first_entry.get("shared_time"),
                "rewrite_type": first_entry.get("rewrite_type")
            }
        ]
    }


def process_dataset(
    dataset_name: str,
    entries: List[Dict],
    output_dict: Dict[str, Dict],
    client: OpenAI,
    model: str,
    temperature: float,
    verbose: bool = False
) -> Tuple[int, int, int]:
    """
    Process a dataset's rewrite entries.
    
    Args:
        dataset_name: Name of the dataset being processed
        entries: List of rewrite entries from this dataset
        output_dict: Dict of passage_id -> unified_entry for output
        client: OpenAI client for rewriting
        model: Model name
        temperature: Temperature for generation
        verbose: Whether to print detailed progress
        
    Returns:
        Tuple of (num_unique_added, num_within_dataset_dups, num_cross_dataset_dups)
    """
    # Group all occurrences by passage_id
    passage_groups = get_all_occurrences_grouped(entries)
    
    # Count statistics
    num_within_dataset_dups = len(entries) - len(passage_groups)
    num_unique_added = 0
    num_cross_dataset_dups = 0
    
    # Process each unique passage in this dataset
    pbar = tqdm(passage_groups.items(), desc=f"Processing {dataset_name}", disable=not verbose)
    for passage_id, entry_list in pbar:
        if passage_id not in output_dict:
            # New passage - process all occurrences sequentially
            # Start with the first entry
            output_dict[passage_id] = create_unified_entry(passage_id, entry_list[0])
            num_unique_added += 1
            
            # If there are within-dataset duplicates, apply them sequentially
            if len(entry_list) > 1:
                if verbose:
                    pbar.set_postfix({"within-rewriting": passage_id})
                
                for entry in entry_list[1:]:
                    # Get current latest
                    current_latest = output_dict[passage_id]["rewritten_passage"]
                    
                    # Apply new rewrite on top
                    new_rewritten_content = rewrite_passage_on_latest(
                        current_latest,
                        entry,
                        client,
                        model,
                        temperature
                    )
                    
                    # Update rewritten passage
                    output_dict[passage_id]["rewritten_passage"] = {
                        "id": passage_id,
                        "title": current_latest.get("title", ""),
                        "section": current_latest.get("section", []),
                        "contents": new_rewritten_content
                    }
                    
                    # Add to rewrite history
                    output_dict[passage_id]["rewrite_history"].append({
                        "query_id": entry.get("query_id"),
                        "entity_label": entry.get("entity_label"),
                        "entity_qid": entry.get("entity_qid"),
                        "property_label": entry.get("property_label"),
                        "property_id": entry.get("property_id"),
                        "property_value": entry.get("property_value"),
                        "shared_time": entry.get("shared_time"),
                        "rewrite_type": entry.get("rewrite_type")
                    })
        else:
            # Cross-dataset duplicate - apply all occurrences sequentially on top of existing
            num_cross_dataset_dups += len(entry_list)
            if verbose:
                pbar.set_postfix({"cross-rewriting": passage_id})
            
            for entry in entry_list:
                # Get the current latest rewritten passage
                current_latest = output_dict[passage_id]["rewritten_passage"]
                
                # Apply new rewrite on top of it
                new_rewritten_content = rewrite_passage_on_latest(
                    current_latest,
                    entry,
                    client,
                    model,
                    temperature
                )
                
                # Update the rewritten passage content
                output_dict[passage_id]["rewritten_passage"] = {
                    "id": passage_id,
                    "title": current_latest.get("title", ""),
                    "section": current_latest.get("section", []),
                    "contents": new_rewritten_content
                }
                
                # Add to rewrite history
                output_dict[passage_id]["rewrite_history"].append({
                    "query_id": entry.get("query_id"),
                    "entity_label": entry.get("entity_label"),
                    "entity_qid": entry.get("entity_qid"),
                    "property_label": entry.get("property_label"),
                    "property_id": entry.get("property_id"),
                    "property_value": entry.get("property_value"),
                    "shared_time": entry.get("shared_time"),
                    "rewrite_type": entry.get("rewrite_type")
                })
    
    return num_unique_added, num_within_dataset_dups, num_cross_dataset_dups


def main():
    parser = argparse.ArgumentParser(description="Fix duplication in rewritten passages across datasets")
    parser.add_argument("--output", type=str, 
                       default="corpus_datasets/passage_rewrites_unified.jsonl",
                       help="Output path for unified rewrite file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model to use for merging passages")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress information")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Set OPENAI_API_KEY env var or use --api_key", 
              file=sys.stderr)
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    
    # Define input files in processing order
    dataset_files = [
        ("mahta", "corpus_datasets/dataset_creation_mahta/passage_rewrites.jsonl"),
        ("qald10", "corpus_datasets/dataset_creation_heydar/qald10/passage_rewrites_qald10.jsonl"),
        ("val_quest", "corpus_datasets/dataset_creation_heydar/quest/val/passage_rewrites_val_quest.jsonl"),
        ("test_quest", "corpus_datasets/dataset_creation_heydar/quest/test/passage_rewrites_test_quest.jsonl"),
        ("train_quest", "corpus_datasets/dataset_creation_heydar/quest/train/passage_rewrites_train_quest.jsonl"),
    ]
    
    # Initialize output dictionary
    output_dict: Dict[str, Dict] = {}
    
    # Statistics tracking
    total_stats = {
        "total_entries_read": 0,
        "total_unique_passages": 0,
        "within_dataset_duplicates": 0,
        "cross_dataset_duplicates": 0
    }
    
    print(f"Starting duplication fix process...")
    print(f"Using model: {args.model}")
    print(f"Output will be written to: {args.output}\n")
    
    # Process each dataset
    for dataset_name, file_path in dataset_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}: {file_path}")
        print(f"{'='*60}")
        
        # Read entries
        entries = read_rewrite_file(file_path)
        total_stats["total_entries_read"] += len(entries)
        print(f"Read {len(entries)} entries from {dataset_name}")
        
        if dataset_name == "mahta":
            # Special handling for mahta: group occurrences but take last without LLM calls
            mahta_groups = get_all_occurrences_grouped(entries)
            within_dups = len(entries) - len(mahta_groups)
            
            # Create unified entries for mahta passages (using last occurrence only)
            # For mahta, we trust that it already has the most updated version in the last occurrence
            for passage_id, entry_list in mahta_groups.items():
                # Take the last entry for mahta (skip LLM rewriting for within-dataset dups)
                last_entry = entry_list[-1]
                
                # For mahta with duplicates, we need to build the complete history
                if len(entry_list) > 1:
                    # Create entry with first occurrence
                    output_dict[passage_id] = create_unified_entry(passage_id, entry_list[0])
                    
                    # Add all subsequent occurrences (including last) to history
                    # We're NOT doing LLM rewrites, just tracking the history
                    # The last entry already has the final rewritten passage
                    for entry in entry_list[1:]:
                        output_dict[passage_id]["rewrite_history"].append({
                            "query_id": entry.get("query_id"),
                            "entity_label": entry.get("entity_label"),
                            "entity_qid": entry.get("entity_qid"),
                            "property_label": entry.get("property_label"),
                            "property_id": entry.get("property_id"),
                            "property_value": entry.get("property_value"),
                            "shared_time": entry.get("shared_time"),
                            "rewrite_type": entry.get("rewrite_type")
                        })
                    
                    # Update the rewritten passage to use the last entry's rewritten version
                    output_dict[passage_id]["rewritten_passage"] = last_entry["rewritten_passage"]
                else:
                    # No duplicates, just create entry normally
                    output_dict[passage_id] = create_unified_entry(passage_id, last_entry)
            
            print(f"  - Unique passages: {len(mahta_groups)}")
            print(f"  - Within-dataset duplicates: {within_dups}")
            print(f"  - Added {len(mahta_groups)} passages to output")
            
            total_stats["within_dataset_duplicates"] += within_dups
            total_stats["total_unique_passages"] = len(output_dict)
        else:
            # Process other datasets with LLM merging
            num_unique, num_within, num_cross = process_dataset(
                dataset_name,
                entries,
                output_dict,
                client,
                args.model,
                args.temperature,
                args.verbose
            )
            
            print(f"  - Unique passages added: {num_unique}")
            print(f"  - Within-dataset duplicates: {num_within}")
            print(f"  - Cross-dataset duplicates (merged): {num_cross}")
            print(f"  - Total passages in output so far: {len(output_dict)}")
            
            total_stats["within_dataset_duplicates"] += num_within
            total_stats["cross_dataset_duplicates"] += num_cross
            total_stats["total_unique_passages"] = len(output_dict)
    
    # Write unified output file
    print(f"\n{'='*60}")
    print(f"Writing unified output to {args.output}")
    print(f"{'='*60}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for passage_id in sorted(output_dict.keys()):
            f.write(json.dumps(output_dict[passage_id], ensure_ascii=False, default=encode_datetime) + '\n')
    
    print(f"Wrote {len(output_dict)} passages to {args.output}")
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Total entries read: {total_stats['total_entries_read']}")
    print(f"Total unique passages in output: {total_stats['total_unique_passages']}")
    print(f"Within-dataset duplicates removed: {total_stats['within_dataset_duplicates']}")
    print(f"Cross-dataset duplicates merged: {total_stats['cross_dataset_duplicates']}")
    print(f"Total duplicates handled: {total_stats['within_dataset_duplicates'] + total_stats['cross_dataset_duplicates']}")
    print(f"\nDuplication fix complete!")


if __name__ == "__main__":
    main()
