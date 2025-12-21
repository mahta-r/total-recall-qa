#!/usr/bin/env python3
"""
Unified Dataset Processing Pipeline

This script orchestrates the dataset processing pipeline for multiple datasets (QALD10, Quest).
The pipeline consists of:
1. Get annotations (QIDs and properties) - Dataset-specific component
2. Extract aggregatable properties - Shared component (QALD10 only)
3. Generate Total Recall queries - Shared component

Usage:
    # For QALD10
    python c1_2_dataset_creation_heydar/run_pipeline.py --dataset qald10 --model openai/gpt-4o

    # For Quest
    python c1_2_dataset_creation_heydar/run_pipeline.py --dataset quest --quest_input test.jsonl --model openai/gpt-4o
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
pipeline_dir = Path(__file__).parent
sys.path.insert(0, str(pipeline_dir))


def run_qald10_pipeline(args):
    """
    Run the QALD10 dataset processing pipeline.

    Steps:
    1. Get annotations from SPARQL queries
    2. Extract aggregatable properties
    3. Generate Total Recall queries
    """
    print("=" * 70)
    print("QALD10 Dataset Processing Pipeline")
    print("=" * 70)
    print(f"Model: {args.model}")
    print()

    # Get paths
    base_dir = pipeline_dir.parent  # /gpfs/home6/hsoudani/total-recall-rag

    # File paths
    input_file = base_dir / "corpus_datasets/qald_datasets/wikidata_aggregation.jsonl"
    output_dir = base_dir / "corpus_datasets/dataset_creation_heydar/qald10"
    step1_output = output_dir / "qald10_annotations.jsonl"
    step1_entity_types = output_dir / "qald10_entity_types.json"
    step2_output = output_dir / "qald10_with_properties.jsonl"
    step3_output = output_dir / "qald10_queries.jsonl"
    prompt_template = base_dir / "c1_2_dataset_creation_heydar" / "prompts/query_generation_v1.txt"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import modules
    try:
        from importlib import import_module
        step1 = import_module('1_get_annotation')
        step2 = import_module('2_get_properties')
        step3 = import_module('3_query_generation')
    except ImportError as e:
        print(f"Error: Could not import pipeline components: {e}")
        return 1

    # Step 1: Get Annotations
    print("=" * 70)
    print("STEP 1: Get Annotations")
    print("=" * 70)
    try:
        result = step1.process_qald10_annotations(
            input_file=str(input_file),
            output_file=str(step1_output),
            entity_type_output_file=str(step1_entity_types),
            model_name=args.model
        )
        if result != 0:
            print("✗ Step 1 failed")
            return 1
        print("✓ Step 1 completed\n")
    except Exception as e:
        print(f"✗ Step 1 failed: {e}")
        return 1

    # Step 2: Get Properties
    print("=" * 70)
    print("STEP 2: Get Properties")
    print("=" * 70)
    try:
        step2.process_dataset_with_aggregatable_properties(str(step1_output), str(step2_output))
        print("✓ Step 2 completed\n")
    except Exception as e:
        print(f"✗ Step 2 failed: {e}")
        return 1

    # Step 3: Generate Queries
    print("=" * 70)
    print("STEP 3: Generate Queries")
    print("=" * 70)
    try:
        step3.process_dataset_for_valid_pairs(
            str(step2_output),
            str(step3_output),
            str(prompt_template),
            args.model,
            resume=True
        )
        print("✓ Step 3 completed\n")
    except Exception as e:
        print(f"✗ Step 3 failed: {e}")
        return 1

    # Done
    print("=" * 70)
    print("QALD10 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Annotations: {step1_output}")
    print(f"With properties: {step2_output}")
    print(f"Final queries: {step3_output}")

    return 0


def run_quest_pipeline(args):
    """
    Run the Quest dataset processing pipeline.

    Steps:
    1. Get annotations (QIDs and intermediate QIDs) from Quest dataset
    2. Extract aggregatable properties
    3. Generate Total Recall queries
    """
    print("=" * 70)
    print("Quest Dataset Processing Pipeline")
    print("=" * 70)
    print(f"Model: {args.model}")
    print()

    # Check required arguments
    if not args.quest_input:
        print("Error: --quest_input is required for Quest dataset")
        return 1

    # Get paths
    base_dir = pipeline_dir.parent  # /gpfs/home6/hsoudani/total-recall-rag
    input_file = base_dir / "corpus_datasets" / "quest_dataset" / args.quest_input
    output_dir = base_dir / "corpus_datasets" / "dataset_creation_heydar" / "quest"

    # Output files for each step
    step1_output = output_dir / f"{Path(args.quest_input).stem}_quest_annotations.jsonl"
    step2_output = output_dir / f"{Path(args.quest_input).stem}_quest_with_properties.jsonl"
    step3_output = output_dir / f"{Path(args.quest_input).stem}_quest_queries.jsonl"

    prompt_template = base_dir / "c1_2_dataset_creation_heydar" / "prompts/query_generation_v1.txt"

    print(f"Input: {input_file}")
    print(f"Step 1 output: {step1_output}")
    print(f"Step 2 output: {step2_output}")
    print(f"Step 3 output: {step3_output}")
    print()

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import modules
    try:
        from importlib import import_module
        step1 = import_module('1_get_annotation')
        step2 = import_module('2_get_properties')
        step3 = import_module('3_query_generation')
    except ImportError as e:
        print(f"Error: Could not import pipeline components: {e}")
        return 1

    # Step 1: Get Annotations
    print("=" * 70)
    print("STEP 1: Get Annotations")
    print("=" * 70)

    result = step1.process_quest_annotations(
        input_file=str(input_file),
        output_file=str(step1_output),
        subsample=args.subsample,
        limit=args.limit
    )

    if result != 0:
        print("\n✗ Step 1 failed")
        return 1

    print("\n✓ Step 1 completed successfully")
    print()

    # Step 2: Get Properties
    print("=" * 70)
    print("STEP 2: Get Properties")
    print("=" * 70)

    try:
        step2.process_dataset_with_aggregatable_properties(str(step1_output), str(step2_output))
        print("\n✓ Step 2 completed successfully")
    except Exception as e:
        print(f"\n✗ Step 2 failed: {e}")
        return 1

    print()

    # Step 3: Generate Total Recall Queries
    print("=" * 70)
    print("STEP 3: Generate Queries")
    print("=" * 70)

    try:
        step3.process_dataset_for_valid_pairs(
            dataset_file=str(step2_output),
            output_file=str(step3_output),
            prompt_template_path=str(prompt_template),
            model_name=args.model,
            resume=args.resume
        )
        print("\n✓ Step 3 completed successfully")
    except Exception as e:
        print(f"\n✗ Step 3 failed: {e}")
        return 1

    # Pipeline completed
    print()
    print("=" * 70)
    print("QUEST PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Annotations: {step1_output}")
    print(f"With properties: {step2_output}")
    print(f"Final queries: {step3_output}")

    return 0


def main():
    parser = argparse.ArgumentParser(description='Run the unified dataset processing pipeline')

    # Common arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['qald10', 'quest'],
                        help='Dataset type to process (qald10 or quest)')
    parser.add_argument('--model', type=str, default='openai/gpt-4o',
                        help='Model to use for LLM steps (default: openai/gpt-4o)')

    # Quest-specific arguments
    parser.add_argument('--quest_input', type=str,
                        help='Input file name for Quest dataset (e.g., train.jsonl, test.jsonl)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of entries to process (Quest only, for testing)')
    parser.add_argument('--subsample', type=float, default=50,
                        help='Number of samples to process (Quest only): -1 for all, 0-1 for percentage, >1 for absolute number (default: 200)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume query generation from last processed entry (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Start query generation fresh, overwrite output file')

    args = parser.parse_args()

    # Route to appropriate pipeline
    if args.dataset == 'qald10':
        return run_qald10_pipeline(args)
    elif args.dataset == 'quest':
        return run_quest_pipeline(args)
    else:
        print(f"Error: Unknown dataset type '{args.dataset}'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
