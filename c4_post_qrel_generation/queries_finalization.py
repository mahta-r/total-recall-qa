"""
Queries Finalization for Total Recall RAG

Aggregates valid queries from qald10 and quest sets. A valid query is one that has
at least one passage for all its entities (same definition as coverage).

Outputs:
- Validation set: valid queries from quest_train + quest_val
- Test set: valid queries from qald10 + quest_test

Uses the same coverage logic as c3_qrel_generation (calculate_coverage) to ensure
consistency with reported coverage numbers.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file, write_jsonl_to_file

# Use same coverage logic as qrel_generation (see qrel_generation.py)
from c3_qrel_generation.src.qrel_analysis import calculate_coverage


# Subset config: (dataset, subset) -> short label for logging and output "src"
SUBSET_CONFIG = {
    "validation": [
        ("quest", "train"),   # quest_train
        ("quest", "val"),     # quest_val
    ],
    "test": [
        ("qald10", None),     # qald10 (subset None -> dataset name as subset)
        ("quest", "test"),   # quest_test
    ],
}


def _subset_to_src(dataset: str, subset: Optional[str]) -> str:
    """Return src label for output: quest_train, quest_val, qald10, quest_test."""
    if subset:
        return f"{dataset}_{subset}" if "_" not in subset else subset
    return dataset


def _to_output_entry(q: Dict[str, Any], src: str) -> Dict[str, Any]:
    """Reduce a generation record to output format: id, question, answer, src."""
    return {
        "id": q.get("qid", ""),
        "question": q.get("question", ""),  # Use 'question' field (from queries file), not 'original_query' (from generations file)
        "answer": q.get("ground_truth"),
        "src": src,
    }


def _resolve_generations_path(dataset: str, subset: Optional[str], base_dir: Path) -> Path:
    """Resolve path to generations file (same logic as qrel_analysis)."""
    if subset:
        if "_" in subset:
            subset_name = subset
            subdir = subset.split("_")[0]
        else:
            subdir = subset
            subset_name = f"{subset}_{dataset}"
    else:
        subset_name = dataset
        subdir = ""
    if subdir:
        return base_dir / dataset / subdir / f"{subset_name}_generations.jsonl"
    return base_dir / dataset / f"{subset_name}_generations.jsonl"


def _resolve_queries_path(dataset: str, subset: Optional[str], base_dir: Path) -> Path:
    """Resolve path to queries file (parallel to generations file)."""
    if subset:
        if "_" in subset:
            subset_name = subset
            subdir = subset.split("_")[0]
        else:
            subdir = subset
            subset_name = f"{subset}_{dataset}"
    else:
        subset_name = dataset
        subdir = ""
    if subdir:
        return base_dir / dataset / subdir / f"{subset_name}_queries.jsonl"
    return base_dir / dataset / f"{subset_name}_queries.jsonl"


def get_valid_queries_for_subset(
    dataset: str,
    subset: Optional[str],
    base_dir: Path,
    qrel_file_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Load generations for a subset, run coverage, and return only valid queries.
    Uses the same logic as calculate_coverage (valid = all entities have at least one passage).
    Loads questions from *_queries.jsonl (not from *_generations.jsonl).
    Returns (valid_queries, total_queries, valid_count).
    """
    if qrel_file_path is None:
        if subset:
            if "_" in subset:
                subset_name = subset
                subdir = subset.split("_")[0]
            else:
                subdir = subset
                subset_name = f"{subset}_{dataset}"
        else:
            subset_name = dataset
            subdir = ""
        if subdir:
            qrel_file_path = base_dir / dataset / subdir / f"qrels_{subset_name}.txt"
        else:
            qrel_file_path = base_dir / dataset / f"qrels_{subset_name}.txt"

    qrel_str = str(qrel_file_path) if qrel_file_path else None
    results = calculate_coverage(
        dataset=dataset,
        qrel_file_path=qrel_str,
        subset=subset,
    )
    valid_indices = results.get("valid_query_indices", [])
    
    # Load generations (for ground_truth and qid)
    generations_path = _resolve_generations_path(dataset, subset, base_dir)
    if not generations_path.exists():
        raise FileNotFoundError(f"Generations file not found: {generations_path}")
    generations = read_jsonl_from_file(str(generations_path))
    
    # Load queries (for correct question text)
    queries_path = _resolve_queries_path(dataset, subset, base_dir)
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    queries = read_jsonl_from_file(str(queries_path))
    
    # Create mapping from id to question
    id_to_question = {q["id"]: q["question"] for q in queries}
    
    # Merge: add correct question from queries to generations data
    merged_queries = []
    for gen in generations:
        qid = gen.get("qid", "")
        if qid in id_to_question:
            gen["question"] = id_to_question[qid]  # Add correct question
        else:
            print(f"WARNING: qid={qid} not found in queries file")
            gen["question"] = gen.get("original_query", "")  # Fallback to original_query
        merged_queries.append(gen)
    
    # Filter to valid queries only
    valid_queries = [merged_queries[i] for i in valid_indices]
    total = len(merged_queries)
    valid_count = len(valid_queries)
    return valid_queries, total, valid_count


def main(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_queries: List[Dict[str, Any]] = []
    test_queries: List[Dict[str, Any]] = []
    stats: List[Tuple[str, int, int]] = []  # (label, total, valid)

    # Validation set: quest_train + quest_val
    print("\n" + "=" * 80)
    print("VALIDATION SET (quest_train, quest_val)")
    print("=" * 80)
    for dataset, subset in SUBSET_CONFIG["validation"]:
        label = f"{dataset}/{subset}" if subset else dataset
        src = _subset_to_src(dataset, subset)
        print(f"\n--- {label} ---")
        valid, total, valid_count = get_valid_queries_for_subset(dataset, subset, base_dir)
        validation_queries.extend([_to_output_entry(q, src) for q in valid])
        stats.append((label, total, valid_count))
        print(f"Valid queries: {valid_count} / {total}")

    # Test set: qald10 + quest_test
    print("\n" + "=" * 80)
    print("TEST SET (qald10, quest_test)")
    print("=" * 80)
    for dataset, subset in SUBSET_CONFIG["test"]:
        label = f"{dataset}/{subset}" if subset else dataset
        src = _subset_to_src(dataset, subset)
        print(f"\n--- {label} ---")
        valid, total, valid_count = get_valid_queries_for_subset(dataset, subset, base_dir)
        test_queries.extend([_to_output_entry(q, src) for q in valid])
        stats.append((label, total, valid_count))
        print(f"Valid queries: {valid_count} / {total}")

    # Write outputs
    validation_path = out_dir / args.validation_filename
    test_path = out_dir / args.test_filename
    write_jsonl_to_file(str(validation_path), validation_queries)
    write_jsonl_to_file(str(test_path), test_queries)

    print("\n" + "=" * 80)
    print("OUTPUTS")
    print("=" * 80)
    print(f"Validation queries: {len(validation_queries)} -> {validation_path}")
    print(f"Test queries:       {len(test_queries)} -> {test_path}")
    print("=" * 80)

    # Final statistics
    total_validation = sum(s[1] for s in stats[:2])
    valid_validation = sum(s[2] for s in stats[:2])
    total_test = sum(s[1] for s in stats[2:])
    valid_test = sum(s[2] for s in stats[2:])
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    for label, total, valid_count in stats:
        pct = (valid_count / total * 100) if total else 0.0
        print(f"  {label:20s}  total: {total:5d}   valid: {valid_count:5d}   coverage: {pct:5.2f}%")
    print("-" * 80)
    print(f"  {'Validation (train+val)':20s}  total: {total_validation:5d}   valid: {valid_validation:5d}   coverage: {(valid_validation/total_validation*100) if total_validation else 0:.2f}%")
    print(f"  {'Test (qald10+test)':20s}  total: {total_test:5d}   valid: {valid_test:5d}   coverage: {(valid_test/total_test*100) if total_test else 0:.2f}%")
    print("-" * 80)
    print(f"  Output validation file: {len(validation_queries)} queries -> {validation_path.name}")
    print(f"  Output test file:      {len(test_queries)} queries -> {test_path.name}")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate valid queries from qald10 and quest (same logic as coverage).")
    parser.add_argument("--base-dir", type=str, default="corpus_datasets/dataset_creation_heydar", help="Base directory containing qald10/ and quest/ (default: corpus_datasets/dataset_creation_heydar)")
    parser.add_argument("--output-dir", type=str, default="corpus_datasets/dataset_creation_heydar", help="Directory for output files (default: same as base-dir)")
    parser.add_argument("--validation-filename", type=str, default="queries_validation_final.jsonl", help="Output filename for validation set")
    parser.add_argument("--test-filename", type=str, default="queries_test_final.jsonl", help="Output filename for test set")
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    main(_args)



# python c4_post_qrel_generation/queries_finalization.py