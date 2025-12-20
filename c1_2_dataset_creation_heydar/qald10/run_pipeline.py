#!/usr/bin/env python3
"""
QALD10 Dataset Augmentation Pipeline Runner

This script runs the complete pipeline for QALD10 dataset augmentation:
1. Get annotations from SPARQL queries (1_get_annotation.py)
2. Extract aggregatable properties (2_get_properties.py)
3. Generate Total Recall queries (3_query_generation.py)

Usage:
    python run_pipeline.py --model openai/gpt-4o
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run the QALD10 dataset augmentation pipeline')
    parser.add_argument('--model', type=str, default='openai/gpt-4o', help='Model to use for LLM steps (default: openai/gpt-4o)')
    args = parser.parse_args()

    # Get paths
    pipeline_dir = Path(__file__).parent
    base_dir = pipeline_dir.parent.parent

    # File paths
    input_file = base_dir / "corpus_datasets/qald_aggregation_samples/wikidata_aggregation.jsonl"
    output_dir = base_dir / "corpus_datasets/dataset_creation_heydar/qald10"
    step1_output = output_dir / "wikidata_totallist.jsonl"
    step1_entity_types = output_dir / "entity_types_mapping.jsonl"
    step2_output = output_dir / "wikidata_totallist_with_properties.jsonl"
    step3_output = output_dir / "wikidata_total_recall_queries.jsonl"
    prompt_template = pipeline_dir / "prompts/query_generation_v1.txt"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add pipeline dir to path
    sys.path.insert(0, str(pipeline_dir))

    print("=" * 70)
    print("QALD10 Dataset Augmentation Pipeline")
    print("=" * 70)
    print(f"Model: {args.model}")
    print()

    # Step 1: Get Annotations
    print("=" * 70)
    print("STEP 1: Get Annotations")
    print("=" * 70)
    try:
        from importlib import import_module
        step1 = import_module('1_get_annotation')

        class Args:
            model_name_or_path = args.model

        step1.args = Args()
        step1.get_annotations(str(input_file), str(step1_output), str(step1_entity_types))
        print("✓ Step 1 completed\n")
    except Exception as e:
        print(f"✗ Step 1 failed: {e}")
        return 1

    # Step 2: Get Properties
    print("=" * 70)
    print("STEP 2: Get Properties")
    print("=" * 70)
    try:
        step2 = import_module('2_get_properties')
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
        step3 = import_module('3_query_generation')
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
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Final output: {step3_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
