"""
Response Refinement for Total Recall RAG

This script creates two versions of answers for each query:
1. wikidata: The original answer from Wikidata (all entities)
2. wikipedia: The refined answer using only entities that have relevant passages in the corpus

The idea is that some entities may not have relevant passages in Wikipedia, so
the wikipedia-based answer only considers entities that do have supporting passages.

Output format (JSONL):
{
    "id": "query_id",
    "question": "query text",
    "answer": {
        "wikidata": {"n_entities": N, "value": X},
        "wikipedia": {"n_entities": M, "value": Y}
    },
    "is_different": true/false
}
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions
from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file


def extract_page_id_from_passage_id(passage_id: str) -> Optional[str]:
    """
    Extract page ID from passage ID.

    Args:
        passage_id: Passage ID in format "pageid-0000", "pageid-0001", etc.

    Returns:
        Page ID string, or None if format is invalid
    """
    dash_pos = passage_id.rfind('-')
    if dash_pos != -1:
        return passage_id[:dash_pos]
    return None


def get_wikipedia_info_from_qid(qid: str) -> Optional[Dict[str, str]]:
    """
    Get Wikipedia page title and page ID from Wikidata QID.

    Args:
        qid: Wikidata QID (e.g., "Q4193501")

    Returns:
        Dict with 'title' and 'pageid' keys, or None if not found
    """
    try:
        headers = {
            "User-Agent": "TotalRecallRAG/1.0 (Research Project; Contact: your-email@example.com)"
        }

        # Query Wikidata API to get Wikipedia page title
        wikidata_url = "https://www.wikidata.org/w/api.php"
        wikidata_params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks",
            "sitefilter": "enwiki",
            "format": "json"
        }

        response = requests.get(wikidata_url, params=wikidata_params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if qid not in data.get("entities", {}):
            return None

        entity = data["entities"][qid]
        if "sitelinks" not in entity or "enwiki" not in entity["sitelinks"]:
            return None

        enwiki_title = entity["sitelinks"]["enwiki"]["title"]

        # Get Wikipedia page ID from title
        wikipedia_url = "https://en.wikipedia.org/w/api.php"
        wikipedia_params = {
            "action": "query",
            "titles": enwiki_title,
            "format": "json"
        }

        response = requests.get(wikipedia_url, params=wikipedia_params, headers=headers)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None

        page_info = next(iter(pages.values()))
        if "missing" in page_info:
            return None

        return {
            "title": enwiki_title,
            "pageid": str(page_info.get("pageid", ""))
        }

    except Exception as e:
        print(f"Error getting Wikipedia info for {qid}: {e}", file=sys.stderr)
        return None


def load_qrel_relevant_pages(qrel_file_path: str) -> Set[str]:
    """
    Load qrel file and extract all page IDs that have relevant passages.

    This returns a GLOBAL set of all pages with relevant passages across all queries,
    matching the coverage definition in qrel_analysis.py.

    Args:
        qrel_file_path: Path to qrel file in TREC format

    Returns:
        Set of page IDs that have at least one relevant passage in the qrel file
    """
    relevant_page_ids = set()

    if not Path(qrel_file_path).exists():
        print(f"Warning: qrel file does not exist: {qrel_file_path}")
        return relevant_page_ids

    with open(qrel_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                passage_id = parts[2]
                relevance = int(parts[3])

                if relevance > 0:
                    page_id = extract_page_id_from_passage_id(passage_id)
                    if page_id:
                        relevant_page_ids.add(page_id)

    return relevant_page_ids


def apply_operation(values: List[Any], operation: str) -> Any:
    """
    Apply the aggregation operation to a list of values.

    Args:
        values: List of numeric values
        operation: One of 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX'

    Returns:
        The computed result, or None if values is empty
    """
    if not values:
        return None

    # Filter out None values
    numeric_values = [v for v in values if v is not None]
    if not numeric_values:
        return None

    op = operation.upper()

    if op == 'COUNT':
        return len(numeric_values)
    elif op == 'SUM':
        return sum(numeric_values)
    elif op == 'AVG':
        return sum(numeric_values) / len(numeric_values)
    elif op == 'MIN':
        return min(numeric_values)
    elif op == 'MAX':
        return max(numeric_values)
    else:
        # Default to COUNT for unknown operations
        print(f"Warning: Unknown operation '{operation}', defaulting to COUNT", file=sys.stderr)
        return len(numeric_values)


def process_query(
    query_obj: Dict,
    qid_to_pageid: Dict[str, str],
    relevant_page_ids: Set[str]
) -> Dict:
    """
    Process a single query to generate both wikidata and wikipedia versions of the answer.

    Args:
        query_obj: Query object from generations JSONL
        qid_to_pageid: Mapping from Wikidata QID to Wikipedia page ID
        relevant_page_ids: Global set of page IDs that have relevant passages (across all queries)

    Returns:
        Result dictionary with id, question, answer (wikidata/wikipedia), and is_different flag
    """
    query_id = query_obj['qid']
    question = query_obj.get('question', query_obj.get('original_query', 'N/A'))
    operation = query_obj.get('operation', 'COUNT')

    # Extract entities_values (handle both QALD10 and Quest structures)
    if 'property' in query_obj and isinstance(query_obj['property'], dict):
        entities_values = query_obj['property'].get('entities_values', [])
    else:
        entities_values = query_obj.get('entities_values', [])

    # Wikidata version: use all entities
    wikidata_values = [entity.get('value') for entity in entities_values]
    wikidata_n_entities = len(entities_values)
    wikidata_answer = apply_operation(wikidata_values, operation)

    # Wikipedia version: use only entities that have relevant passages
    # An entity is covered if its Wikipedia page has relevant passages anywhere in the dataset
    wikipedia_values = []
    wikipedia_entities_count = 0

    for entity in entities_values:
        entity_id = entity.get('entity_id')
        entity_value = entity.get('value')

        if entity_id and entity_id in qid_to_pageid:
            page_id = qid_to_pageid[entity_id]
            if page_id in relevant_page_ids:
                # This entity's Wikipedia page has relevant passages in the corpus
                wikipedia_values.append(entity_value)
                wikipedia_entities_count += 1

    wikipedia_answer = apply_operation(wikipedia_values, operation)

    # Determine if answers are different
    # Handle floating point comparison for AVG
    if wikidata_answer is None and wikipedia_answer is None:
        is_different = False
    elif wikidata_answer is None or wikipedia_answer is None:
        is_different = True
    elif isinstance(wikidata_answer, float) or isinstance(wikipedia_answer, float):
        # Use tolerance for floating point comparison
        is_different = abs(wikidata_answer - wikipedia_answer) > 1e-6
    else:
        is_different = wikidata_answer != wikipedia_answer

    return {
        'id': query_id,
        'question': question,
        'answer': {
            'wikidata': {
                'n_entities': wikidata_n_entities,
                'value': wikidata_answer
            },
            'wikipedia': {
                'n_entities': wikipedia_entities_count,
                'value': wikipedia_answer
            }
        },
        'is_different': is_different
    }


def main(args):
    """Main function to run response refinement."""

    # Use subset if provided, otherwise use default mapping
    if args.subset:
        subset_name = args.subset
    else:
        # Default mapping: quest -> test_quest, qald10 -> qald10
        subset_name = "test_quest" if args.dataset == "quest" else args.dataset

    # Construct file paths
    generations_file = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/{subset_name}_generations.jsonl"
    qrel_file = args.qrel_file or f"corpus_datasets/dataset_creation_heydar/{args.dataset}/qrels_{subset_name}.txt"
    output_file = args.output_file or f"corpus_datasets/dataset_creation_heydar/{args.dataset}/{subset_name}_refined_answers.jsonl"

    print("="*80)
    print("Response Refinement Configuration")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Subset: {subset_name}")
    print(f"Generations file: {generations_file}")
    print(f"Qrel file: {qrel_file}")
    print(f"Output file: {output_file}")
    print("="*80 + "\n")

    # Load queries
    print(f"Loading queries from {generations_file}...")
    queries = read_jsonl_from_file(generations_file)
    print(f"Loaded {len(queries)} queries")

    # Collect all unique entity IDs
    print("Collecting entity IDs...")
    all_qids = set()
    for query_obj in queries:
        if 'property' in query_obj and isinstance(query_obj['property'], dict):
            entities_values = query_obj['property'].get('entities_values', [])
        else:
            entities_values = query_obj.get('entities_values', [])

        for entity in entities_values:
            entity_id = entity.get('entity_id')
            if entity_id:
                all_qids.add(entity_id)

    print(f"Found {len(all_qids)} unique entity IDs")

    # Build QID to Wikipedia page ID mapping
    print("Mapping entity IDs to Wikipedia page IDs...")
    qid_to_pageid = {}

    for qid in tqdm(all_qids, desc="Mapping QIDs"):
        wiki_info = get_wikipedia_info_from_qid(qid)
        if wiki_info:
            qid_to_pageid[qid] = wiki_info['pageid']

    print(f"Successfully mapped {len(qid_to_pageid)} entities to Wikipedia pages")
    print(f"Entities without Wikipedia pages: {len(all_qids) - len(qid_to_pageid)}")

    # Load qrel file - get global set of all pages with relevant passages
    print(f"\nLoading qrel file: {qrel_file}")
    relevant_page_ids = load_qrel_relevant_pages(qrel_file)
    print(f"Found {len(relevant_page_ids)} unique pages with relevant passages")

    # Process each query
    print("\nProcessing queries...")
    results = []
    different_count = 0

    for query_obj in tqdm(queries, desc="Processing queries"):
        result = process_query(
            query_obj=query_obj,
            qid_to_pageid=qid_to_pageid,
            relevant_page_ids=relevant_page_ids
        )
        results.append(result)

        if result['is_different']:
            different_count += 1

    # Write output
    print(f"\nWriting results to {output_file}...")
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print summary
    print("\n" + "="*80)
    print("RESPONSE REFINEMENT SUMMARY")
    print("="*80)
    print(f"Total queries processed: {len(queries)}")
    print(f"Queries with different answers: {different_count}")
    print(f"Queries with same answers: {len(queries) - different_count}")
    print(f"Percentage different: {different_count / len(queries) * 100:.2f}%")
    print(f"\nOutput written to: {output_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate refined answers comparing Wikidata (all entities) vs Wikipedia (entities with passages)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="qald10",
        choices=["quest", "qald10"],
        help="Dataset name (e.g., 'quest', 'qald10')"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Subset name (e.g., 'test_quest', 'train_quest'). If not provided, defaults to 'test_quest' for quest dataset"
    )
    parser.add_argument(
        "--qrel_file",
        type=str,
        default=None,
        help="Optional: Path to qrel file. If not provided, uses default path based on dataset/subset"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Path for output JSONL. If not provided, uses default path"
    )

    args = parser.parse_args()
    main(args)


# ============================================================================
# Usage Examples
# ============================================================================
#
# 1. Run with default paths for QALD10:
#    python c3_qrel_generation/response_refinement.py --dataset qald10
#
# 2. Run with default paths for Quest (uses test_quest subset by default):
#    python c3_qrel_generation/response_refinement.py --dataset quest
#
# 3. Run with specific subset:
#    python c3_qrel_generation/response_refinement.py --dataset quest --subset train_quest
#    python c3_qrel_generation/response_refinement.py --dataset quest --subset test_quest
#
# 4. Run with custom qrel file:
#    python c3_qrel_generation/response_refinement.py --dataset qald10 \
#           --qrel_file path/to/custom_qrels.txt
#
# 5. Run with custom output file:
#    python c3_qrel_generation/response_refinement.py --dataset qald10 \
#           --output_file path/to/custom_output.jsonl
#
# Output format (JSONL):
# {
#     "id": "13_p569",
#     "question": "What is the average year of birth of the General Secretaries...",
#     "answer": {
#         "wikidata": {"n_entities": 2, "value": 1886},
#         "wikipedia": {"n_entities": 1, "value": 1878}
#     },
#     "is_different": true
# }
#
# ============================================================================
