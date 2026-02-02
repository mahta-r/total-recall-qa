"""
Queries Finalization for Total Recall RAG

Aggregates valid queries from qald10 and quest sets. A valid query is one that has
**per-query, per-entity coverage**: for that specific query, for each entity in that
query, at least one qrel row (with that query_id) must reference a passage from that
entity's Wikipedia page.

Outputs:
- Validation set: valid queries from quest_train + quest_val
- Test set: valid queries from qald10 + quest_test
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file, write_jsonl_to_file
from c3_qrel_generation.src.qrel_analysis import (
    get_wikipedia_info_from_qid,
    extract_page_id_from_passage_id,
)
from tqdm import tqdm


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
    """Reduce a generation record to output format: id, question, answer, src. Answer from *_queries file when available."""
    return {
        "id": q.get("qid", ""),
        "question": q.get("question", ""),  # From *_queries file
        "answer": q.get("answer", q.get("ground_truth")),  # Prefer answer from *_queries file
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


def _entities_values_for_query(query_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract entities_values from a query record (QALD10 or Quest structure)."""
    if "property" in query_obj and isinstance(query_obj.get("property"), dict):
        return query_obj["property"].get("entities_values", [])
    return query_obj.get("entities_values", [])


def calculate_query_entity_coverage(
    generations: List[Dict[str, Any]],
    qrel_file_path: Path,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Coverage definition: for each query, for each entity in that query, at least one
    qrel row with that query_id must reference a passage from that entity's Wikipedia page.

    Returns (valid_query_indices, stats_dict).
    """
    # Collect all entity IDs from generations
    entity_list = set()
    for query_obj in generations:
        for entity in _entities_values_for_query(query_obj):
            eid = entity.get("entity_id")
            if eid:
                entity_list.add(eid)
    entity_list = list(entity_list)

    # Map entity_id -> Wikipedia page_id
    entity_to_pageid: Dict[str, str] = {}
    entities_without_wikipedia: Set[str] = set()
    for qid in tqdm(entity_list, desc="Mapping QIDs to page IDs"):
        wiki_info = get_wikipedia_info_from_qid(qid)
        if wiki_info:
            entity_to_pageid[qid] = str(wiki_info["pageid"])
        else:
            entities_without_wikipedia.add(qid)

    # Build per-query coverage: query_id -> set of page_ids that have at least one passage for that query
    # (from qrel rows with that query_id)
    query_to_covered_page_ids: Dict[str, Set[str]] = defaultdict(set)
    if not qrel_file_path.exists():
        return [], {
            "total_queries": len(generations),
            "valid_count": 0,
            "entities_without_wikipedia": len(entities_without_wikipedia),
        }

    with open(qrel_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                passage_id = parts[2]
                relevance = int(parts[3])
                if relevance > 0:
                    page_id = extract_page_id_from_passage_id(passage_id)
                    if page_id:
                        query_to_covered_page_ids[query_id].add(page_id)

    # For each query, valid iff every entity (with a Wikipedia page) has its page in this query's covered set
    valid_query_indices: List[int] = []
    for idx, query_obj in enumerate(generations):
        entities_values = _entities_values_for_query(query_obj)
        qid = query_obj.get("qid", "")
        covered_page_ids = query_to_covered_page_ids.get(qid, set())

        all_covered = True
        for entity in entities_values:
            entity_id = entity.get("entity_id")
            if not entity_id:
                continue
            if entity_id not in entity_to_pageid:
                all_covered = False
                break
            page_id = entity_to_pageid[entity_id]
            if page_id not in covered_page_ids:
                all_covered = False
                break

        if all_covered:
            valid_query_indices.append(idx)

    stats = {
        "total_queries": len(generations),
        "valid_count": len(valid_query_indices),
        "entities_without_wikipedia": len(entities_without_wikipedia),
    }
    return valid_query_indices, stats


def get_valid_queries_for_subset(
    dataset: str,
    subset: Optional[str],
    base_dir: Path,
    qrel_file_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Load generations for a subset, run coverage, and return only valid queries.

    Coverage: for each query, for each entity in that query, at least one qrel row
    with that query_id must reference a passage from that entity's Wikipedia page.
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

    # Load generations first (needed for coverage)
    generations_path = _resolve_generations_path(dataset, subset, base_dir)
    if not generations_path.exists():
        raise FileNotFoundError(f"Generations file not found: {generations_path}")
    generations = read_jsonl_from_file(str(generations_path))

    # Coverage: per-query, per-entity (at least one passage for that query from each entity's page)
    valid_indices, _stats = calculate_query_entity_coverage(generations, qrel_file_path)

    # Load queries (for correct question text)
    queries_path = _resolve_queries_path(dataset, subset, base_dir)
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    queries = read_jsonl_from_file(str(queries_path))

    # Create mappings from id to question and answer (from *_queries file)
    id_to_question = {q["id"]: q["question"] for q in queries}
    id_to_answer = {q["id"]: q.get("answer", q.get("ground_truth")) for q in queries}

    # Merge: add question and answer from queries to generations data
    merged_queries = []
    for gen in generations:
        qid = gen.get("qid", "")
        if qid in id_to_question:
            gen["question"] = id_to_question[qid]
            gen["answer"] = id_to_answer.get(qid, gen.get("ground_truth"))
        else:
            print(f"WARNING: qid={qid} not found in queries file")
            gen["question"] = gen.get("original_query", "")  # Fallback to original_query
            gen["answer"] = gen.get("ground_truth")
        merged_queries.append(gen)

    # Filter to valid queries only
    valid_queries = [merged_queries[i] for i in valid_indices]
    total = len(merged_queries)
    valid_count = len(valid_queries)
    return valid_queries, total, valid_count


def main_queries(args: argparse.Namespace) -> None:
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


def _src_to_qrel_path(src: str, base_dir: Path) -> Path:
    """Return path to source qrel file for a given src (qald10, quest_train, quest_val, quest_test)."""
    if src == "qald10":
        return base_dir / "qald10" / "qrels_qald10.txt"
    if src == "quest_train":
        return base_dir / "quest" / "train" / "qrels_train_quest.txt"
    if src == "quest_val":
        return base_dir / "quest" / "val" / "qrels_val_quest.txt"
    if src == "quest_test":
        return base_dir / "quest" / "test" / "qrels_test_quest.txt"
    raise ValueError(f"Unknown src for qrel path: {src}")


def main_qrels(args: argparse.Namespace) -> None:
    """
    Split qrels into validation and test sets using the final query files.

    Reads queries_validation_final.jsonl and queries_test_final.jsonl to get (id, src)
    for each query, then copies the corresponding rows from each source qrel file
    into qrels_validation_final.txt and qrels_test_final.txt.
    """
    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_queries_path = Path(args.validation_queries_path)
    test_queries_path = Path(args.test_queries_path)
    if not validation_queries_path.exists():
        raise FileNotFoundError(f"Validation queries file not found: {validation_queries_path}")
    if not test_queries_path.exists():
        raise FileNotFoundError(f"Test queries file not found: {test_queries_path}")

    validation_queries = read_jsonl_from_file(str(validation_queries_path))
    test_queries = read_jsonl_from_file(str(test_queries_path))

    # (id, src) -> we need query ids per src to know which rows to copy from each source qrel
    validation_by_src: Dict[str, Set[str]] = defaultdict(set)
    for q in validation_queries:
        qid = q.get("id", "")
        src = q.get("src", "")
        if qid and src:
            validation_by_src[src].add(qid)

    test_by_src: Dict[str, Set[str]] = defaultdict(set)
    for q in test_queries:
        qid = q.get("id", "")
        src = q.get("src", "")
        if qid and src:
            test_by_src[src].add(qid)

    def copy_qrel_rows(
        query_ids_by_src: Dict[str, Set[str]],
        output_path: Path,
    ) -> int:
        total_rows = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            for src, ids in query_ids_by_src.items():
                qrel_path = _src_to_qrel_path(src, base_dir)
                if not qrel_path.exists():
                    print(f"WARNING: Source qrel not found: {qrel_path}, skipping src={src}")
                    continue
                with open(qrel_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        # TREC format: query_id 0 passage_id relevance
                        parts = line.split()
                        if len(parts) >= 1 and parts[0] in ids:
                            out_f.write(line + "\n")
                            total_rows += 1
        return total_rows

    validation_qrel_path = out_dir / args.validation_qrel_filename
    test_qrel_path = out_dir / args.test_qrel_filename
    validation_rows = copy_qrel_rows(validation_by_src, validation_qrel_path)
    test_rows = copy_qrel_rows(test_by_src, test_qrel_path)

    # Report how many queries have at least one qrel row (for sanity check)
    def count_queries_with_rows(qrel_path: Path) -> Set[str]:
        seen: Set[str] = set()
        with open(qrel_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(line.split()[0])
        return seen

    val_qrel_ids = count_queries_with_rows(validation_qrel_path)
    test_qrel_ids = count_queries_with_rows(test_qrel_path)
    val_query_ids = {q["id"] for q in validation_queries if q.get("id")}
    test_query_ids = {q["id"] for q in test_queries if q.get("id")}
    val_missing = val_query_ids - val_qrel_ids
    test_missing = test_query_ids - test_qrel_ids

    print("\n" + "=" * 80)
    print("QREL SPLIT OUTPUTS")
    print("=" * 80)
    print(f"Validation qrels: {validation_rows} rows -> {validation_qrel_path}")
    print(f"Test qrels:       {test_rows} rows -> {test_qrel_path}")
    if val_missing:
        print(f"  (Validation: {len(val_missing)} queries in JSONL have no rows in source qrels)")
    if test_missing:
        print(f"  (Test: {len(test_missing)} queries in JSONL have no rows in source qrels)")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate valid queries from qald10 and quest (same logic as coverage). Split qrels into val/test using final query files.")
    parser.add_argument("--base-dir", type=str, default="corpus_datasets/dataset_creation_heydar", help="Base directory containing qald10/ and quest/ (default: corpus_datasets/dataset_creation_heydar)")
    parser.add_argument("--output-dir", type=str, default="corpus_datasets/dataset_creation_heydar", help="Directory for output files (default: same as base-dir)")
    parser.add_argument("--validation-filename", type=str, default="queries_validation_final.jsonl", help="Output filename for validation queries")
    parser.add_argument("--test-filename", type=str, default="queries_test_final.jsonl", help="Output filename for test queries")
    # Qrel split: inputs (final query files) and outputs (qrel filenames)
    parser.add_argument("--validation-queries-path", type=str, default=None, help="Path to validation queries JSONL (for qrel split). Default: output-dir/queries_validation_final.jsonl")
    parser.add_argument("--test-queries-path", type=str, default=None, help="Path to test queries JSONL (for qrel split). Default: output-dir/queries_test_final.jsonl")
    parser.add_argument("--validation-qrel-filename", type=str, default="qrels_validation_final.txt", help="Output filename for validation qrels")
    parser.add_argument("--test-qrel-filename", type=str, default="qrels_test_final.txt", help="Output filename for test qrels")
    parser.add_argument("--run", type=str, default="queries", choices=["queries", "qrels", "all"], help="Run: queries only, qrels only, or both (default: queries)")
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    if _args.validation_queries_path is None:
        _args.validation_queries_path = str(Path(_args.output_dir) / "queries_validation_final.jsonl")
    if _args.test_queries_path is None:
        _args.test_queries_path = str(Path(_args.output_dir) / "queries_test_final.jsonl")
    if _args.run in ("queries", "all"):
        main_queries(_args)
    if _args.run in ("qrels", "all"):
        main_qrels(_args)



# python c4_post_qrel_generation/dataset_finalization.py --run all