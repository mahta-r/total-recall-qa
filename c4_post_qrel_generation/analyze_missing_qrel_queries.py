"""
Analyze queries that appear in final query files but have no rows in the final qrel files.

For each such query, looks up the corresponding *_generations.jsonl and reports how many
entities (entities_values) the query has. This helps find the root cause:

- If n_entities == 0: coverage treats "all entities covered" as vacuously true, so the
  query is marked valid and included in the final list, but qrel generation writes 0 rows.
- If n_entities > 0: the query has entities but no qrel rows (e.g. qrel file was
  regenerated, or qrel generation failed/skipped for those queries).

Usage:
  python c4_post_qrel_generation/analyze_missing_qrel_queries.py
  python c4_post_qrel_generation/analyze_missing_qrel_queries.py --base-dir corpus_datasets/dataset_creation_heydar
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file


def _src_to_generations_path(src: str, base_dir: Path) -> Path:
    """Path to generations JSONL for a given src."""
    if src == "qald10":
        return base_dir / "qald10" / "qald10_generations.jsonl"
    if src == "quest_train":
        return base_dir / "quest" / "train" / "train_quest_generations.jsonl"
    if src == "quest_val":
        return base_dir / "quest" / "val" / "val_quest_generations.jsonl"
    if src == "quest_test":
        return base_dir / "quest" / "test" / "test_quest_generations.jsonl"
    raise ValueError(f"Unknown src: {src}")


def _entities_count(record: dict) -> int:
    """Number of entities for this query (same logic as qrel_analysis)."""
    if "property" in record and isinstance(record.get("property"), dict):
        entities_values = record["property"].get("entities_values", [])
    else:
        entities_values = record.get("entities_values", [])
    return len(entities_values)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze missing qrel queries via generations files")
    parser.add_argument("--base-dir", type=str, default="corpus_datasets/dataset_creation_heydar", help="Base directory")
    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    validation_queries = read_jsonl_from_file(str(base_dir / "queries_validation_final.jsonl"))
    test_queries = read_jsonl_from_file(str(base_dir / "queries_test_final.jsonl"))

    def qrel_query_ids(path: Path) -> set:
        out = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.add(line.split()[0])
        return out

    val_qrel_ids = qrel_query_ids(base_dir / "qrels_validation_final.txt")
    test_qrel_ids = qrel_query_ids(base_dir / "qrels_test_final.txt")

    val_ids = {q["id"] for q in validation_queries if q.get("id")}
    test_ids = {q["id"] for q in test_queries if q.get("id")}

    val_missing = [(q["id"], q["src"]) for q in validation_queries if q.get("id") and q["id"] not in val_qrel_ids]
    test_missing = [(q["id"], q["src"]) for q in test_queries if q.get("id") and q["id"] not in test_qrel_ids]

    # Load generations by src: src -> { qid -> record }
    def load_generations_by_qid(src: str) -> dict:
        path = _src_to_generations_path(src, base_dir)
        if not path.exists():
            return {}
        rows = read_jsonl_from_file(str(path))
        return {r.get("qid"): r for r in rows if r.get("qid")}

    # Collect unique srcs for missing queries
    srcs_needed = set()
    for _id, src in val_missing + test_missing:
        if src:
            srcs_needed.add(src)
    gen_by_src = {src: load_generations_by_qid(src) for src in srcs_needed}

    def report(name: str, missing: list) -> None:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  Total missing: {len(missing)}")
        print("=" * 60)
        zero_entity = []
        nonzero_entity = []
        not_in_gen = []
        for qid, src in missing:
            gen = gen_by_src.get(src, {})
            rec = gen.get(qid) if gen else None
            if rec is None:
                not_in_gen.append((qid, src))
                continue
            n = _entities_count(rec)
            if n == 0:
                zero_entity.append((qid, src))
            else:
                nonzero_entity.append((qid, src, n))
        if zero_entity:
            print(f"\n  Queries with 0 entities (vacuously valid in coverage → 0 qrel rows): {len(zero_entity)}")
            for qid, src in zero_entity[:15]:
                print(f"    {qid}  src={src}")
            if len(zero_entity) > 15:
                print(f"    ... and {len(zero_entity) - 15} more")
        if nonzero_entity:
            print(f"\n  Queries with >0 entities but still 0 qrel rows: {len(nonzero_entity)}")
            for qid, src, n in nonzero_entity[:15]:
                print(f"    {qid}  src={src}  n_entities={n}")
            if len(nonzero_entity) > 15:
                print(f"    ... and {len(nonzero_entity) - 15} more")
        if not_in_gen:
            print(f"\n  Queries not found in generations file: {len(not_in_gen)}")
            for qid, src in not_in_gen[:10]:
                print(f"    {qid}  src={src}")
        print()

    report("VALIDATION (queries in queries_validation_final.jsonl with no qrel rows)", val_missing)
    report("TEST (queries in queries_test_final.jsonl with no qrel rows)", test_missing)

    # Summary
    all_missing = val_missing + test_missing
    zero = sum(1 for qid, src in all_missing if gen_by_src.get(src, {}).get(qid) and _entities_count(gen_by_src[src][qid]) == 0)
    nonzero = sum(1 for qid, src in all_missing if gen_by_src.get(src, {}).get(qid) and _entities_count(gen_by_src[src][qid]) > 0)
    not_found = len(all_missing) - zero - nonzero
    print("SUMMARY")
    print("-" * 40)
    print(f"  Missing with 0 entities (vacuously valid): {zero}")
    print(f"  Missing with >0 entities (investigate):     {nonzero}")
    print(f"  Missing not in generations file:            {not_found}")


if __name__ == "__main__":
    main()
