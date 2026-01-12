"""
QRel Analysis for Total Recall RAG

This script provides analysis functions for qrel files:
1. Calculate entity coverage - percentage of entities with at least one relevant passage
2. Find entities without relevant passages for each query
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Optional, Set
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions
from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file, read_json_from_file


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
        # Set User-Agent header as required by Wikidata/Wikipedia API etiquette
        headers = {
            "User-Agent": "TotalRecallRAG/1.0 (Research Project; Contact: your-email@example.com)"
        }

        # Query Wikidata API to get Wikipedia page title
        wikidata_url = f"https://www.wikidata.org/w/api.php"
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


def calculate_coverage(
    dataset: str,
    qrel_file_path: Optional[str] = None
) -> Dict:
    """
    Calculate entity coverage given a dataset name and qrel file.

    Entity coverage is defined as the percentage of entities that have at least
    one relevant passage in the qrel file.

    Args:
        dataset: Dataset name (e.g., 'quest', 'qald10')
        qrel_file_path: Optional path to the qrel file. If not provided, uses default path
                       based on dataset name

    Returns:
        Dictionary containing:
            - total_entities: Total number of entities in the list
            - entities_with_passages: Number of entities with at least one relevant passage
            - coverage_percentage: Percentage of entities covered
            - covered_entities: Set of entity IDs that have coverage
            - entities_without_wikipedia: Set of entity IDs that don't have Wikipedia pages
    """
    # Map dataset name to directory name (quest -> test_quest)
    dataset_name = "test_quest" if dataset == "quest" else dataset

    # Construct file paths
    generations_file = f"corpus_datasets/dataset_creation_heydar/{dataset}/{dataset_name}_generations.jsonl"
    if qrel_file_path is None:
        qrel_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/qrels_{dataset_name}.txt"

    print(f"Loading queries from {generations_file}...")
    queries = read_jsonl_from_file(generations_file)
    print(f"Loaded {len(queries)} queries")

    # Extract all unique entity IDs from queries
    entity_list = set()
    for query_obj in queries:
        # Handle different dataset structures:
        # QALD10: has nested structure with query_obj['property']['entities_values']
        # Quest: has flat structure with query_obj['entities_values']
        if 'property' in query_obj and isinstance(query_obj['property'], dict):
            # QALD10 structure
            property_info = query_obj['property']
            entities_values = property_info.get('entities_values', [])
        else:
            # Quest structure
            entities_values = query_obj.get('entities_values', [])

        for entity in entities_values:
            entity_id = entity.get('entity_id')
            if entity_id:
                entity_list.add(entity_id)

    entity_list = list(entity_list)
    print(f"Found {len(entity_list)} unique entities across all queries")

    # Map Wikipedia page IDs for all entities
    print(f"Mapping {len(entity_list)} entities to Wikipedia page IDs...")
    entity_to_pageid = {}
    entities_without_wikipedia = set()

    for qid in tqdm(entity_list, desc="Mapping QIDs to page IDs"):
        wiki_info = get_wikipedia_info_from_qid(qid)
        if wiki_info:
            entity_to_pageid[qid] = wiki_info['pageid']
        else:
            entities_without_wikipedia.add(qid)

    print(f"Successfully mapped {len(entity_to_pageid)} entities to Wikipedia pages")
    print(f"Entities without Wikipedia pages: {len(entities_without_wikipedia)}")

    # Read qrel file and extract all passage IDs with relevance > 0
    print(f"Reading qrel file: {qrel_file_path}")
    relevant_passage_ids = set()

    if not Path(qrel_file_path).exists():
        print(f"Warning: qrel file does not exist: {qrel_file_path}")
        return {
            'total_entities': len(entity_list),
            'entities_with_passages': 0,
            'coverage_percentage': 0.0,
            'covered_entities': set(),
            'entities_without_wikipedia': entities_without_wikipedia
        }

    with open(qrel_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                passage_id = parts[2]
                relevance = int(parts[3])
                if relevance > 0:
                    relevant_passage_ids.add(passage_id)

    print(f"Found {len(relevant_passage_ids)} relevant passages in qrel file")

    # Extract page IDs from relevant passage IDs
    relevant_page_ids = set()
    for passage_id in relevant_passage_ids:
        page_id = extract_page_id_from_passage_id(passage_id)
        if page_id:
            relevant_page_ids.add(page_id)

    print(f"These passages belong to {len(relevant_page_ids)} unique pages")

    # Check which entities have at least one relevant passage
    covered_entities = set()
    for qid, pageid in entity_to_pageid.items():
        if pageid in relevant_page_ids:
            covered_entities.add(qid)

    # Calculate coverage
    total_entities = len(entity_list)
    entities_with_passages = len(covered_entities)
    coverage_percentage = (entities_with_passages / total_entities * 100) if total_entities > 0 else 0.0

    results = {
        'total_entities': total_entities,
        'entities_with_passages': entities_with_passages,
        'coverage_percentage': coverage_percentage,
        'covered_entities': covered_entities,
        'entities_without_wikipedia': entities_without_wikipedia
    }

    # Print results
    print("\n" + "="*80)
    print("ENTITY COVERAGE ANALYSIS")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Total entities: {total_entities}")
    print(f"Entities with Wikipedia pages: {len(entity_to_pageid)}")
    print(f"Entities without Wikipedia pages: {len(entities_without_wikipedia)}")
    print(f"Entities with at least one relevant passage: {entities_with_passages}")
    print(f"Entity coverage: {coverage_percentage:.2f}%")
    print("="*80 + "\n")

    return results


def analyze_entities_without_passages(
    dataset: str,
    qrel_file_path: Optional[str] = None,
    output_file_path: Optional[str] = None
) -> None:
    """
    Analyze queries to find entities without relevant passages.

    For each query, identifies which entities don't have any relevant passages
    and writes the results to a JSONL file.

    Args:
        dataset: Dataset name (e.g., 'quest', 'qald10')
        qrel_file_path: Optional path to the qrel file. If not provided, uses default path
        output_file_path: Optional path for output JSONL. If not provided, uses default path
    """
    # Map dataset name to directory name (quest -> test_quest)
    dataset_name = "test_quest" if dataset == "quest" else dataset

    # Construct file paths
    generations_file = f"corpus_datasets/dataset_creation_heydar/{dataset}/{dataset_name}_generations.jsonl"
    if qrel_file_path is None:
        qrel_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/qrels_{dataset_name}.txt"
    if output_file_path is None:
        output_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/entities_without_passages_{dataset_name}.jsonl"

    print(f"Loading queries from {generations_file}...")
    queries = read_jsonl_from_file(generations_file)
    print(f"Loaded {len(queries)} queries")

    # Build a mapping from QID to Wikipedia page ID
    print("Building QID to Wikipedia page ID mapping...")
    qid_to_pageid = {}
    all_qids = set()

    # First collect all unique QIDs
    for query_obj in queries:
        if 'property' in query_obj and isinstance(query_obj['property'], dict):
            entities_values = query_obj['property'].get('entities_values', [])
        else:
            entities_values = query_obj.get('entities_values', [])

        for entity in entities_values:
            entity_id = entity.get('entity_id')
            if entity_id:
                all_qids.add(entity_id)

    # Map QIDs to page IDs
    for qid in tqdm(all_qids, desc="Mapping QIDs to page IDs"):
        wiki_info = get_wikipedia_info_from_qid(qid)
        if wiki_info:
            qid_to_pageid[qid] = wiki_info['pageid']

    print(f"Mapped {len(qid_to_pageid)} QIDs to Wikipedia page IDs")

    # Read qrel file and build mapping from query_id to set of page IDs with relevant passages
    print(f"Reading qrel file: {qrel_file_path}")
    query_to_relevant_pages = {}

    if not Path(qrel_file_path).exists():
        print(f"Warning: qrel file does not exist: {qrel_file_path}")
        return

    with open(qrel_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                passage_id = parts[2]
                relevance = int(parts[3])

                if relevance > 0:
                    page_id = extract_page_id_from_passage_id(passage_id)
                    if page_id:
                        if query_id not in query_to_relevant_pages:
                            query_to_relevant_pages[query_id] = set()
                        query_to_relevant_pages[query_id].add(page_id)

    print(f"Found relevant passages for {len(query_to_relevant_pages)} queries")

    # Analyze each query to find entities without relevant passages
    print("Analyzing entities without relevant passages...")
    results = []
    queries_with_missing_entities = 0
    total_missing_entities = 0

    for query_obj in tqdm(queries, desc="Processing queries"):
        qid = query_obj['qid']
        query_text = query_obj.get('question', query_obj.get('original_query', 'N/A'))

        # Extract entities for this query
        if 'property' in query_obj and isinstance(query_obj['property'], dict):
            entities_values = query_obj['property'].get('entities_values', [])
        else:
            entities_values = query_obj.get('entities_values', [])

        # Get the set of relevant page IDs for this query
        relevant_pages = query_to_relevant_pages.get(qid, set())

        # Check which entities don't have relevant passages
        entities_without_passages = []
        for entity in entities_values:
            entity_id = entity.get('entity_id')
            entity_label = entity.get('entity_label', 'Unknown')

            if entity_id:
                # Check if entity has a Wikipedia page
                if entity_id not in qid_to_pageid:
                    entities_without_passages.append({
                        'entity_id': entity_id,
                        'entity_label': entity_label,
                        'reason': 'no_wikipedia_page'
                    })
                else:
                    # Check if the page has relevant passages
                    page_id = qid_to_pageid[entity_id]
                    if page_id not in relevant_pages:
                        entities_without_passages.append({
                            'entity_id': entity_id,
                            'entity_label': entity_label,
                            'reason': 'no_relevant_passages'
                        })

        # Only record if there are entities without passages
        if entities_without_passages:
            queries_with_missing_entities += 1
            total_missing_entities += len(entities_without_passages)
            results.append({
                'qid': qid,
                'query': query_text,
                'entities_without_passages': entities_without_passages,
                'total_entities': len(entities_values),
                'missing_count': len(entities_without_passages)
            })

    # Write results to JSONL file
    print(f"\nWriting results to {output_file_path}...")
    output_dir = Path(output_file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print summary
    print("\n" + "="*80)
    print("ENTITIES WITHOUT PASSAGES ANALYSIS")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Total queries: {len(queries)}")
    print(f"Queries with entities missing passages: {queries_with_missing_entities}")
    print(f"Total entities without relevant passages: {total_missing_entities}")
    print(f"Average missing entities per affected query: {total_missing_entities / queries_with_missing_entities:.2f}" if queries_with_missing_entities > 0 else "Average: 0")
    print(f"Results written to: {output_file_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze qrel files for entity coverage and missing passages")

    parser.add_argument("--dataset", type=str, required=True, choices=["quest", "qald10"], help="Dataset name (e.g., 'quest', 'qald10')")
    parser.add_argument("--qrel_file", type=str, default=None, help="Optional: Path to qrel file. If not provided, uses default path based on dataset")
    parser.add_argument("--analysis", type=str, default="coverage", choices=["coverage", "missing", "both"], help="Type of analysis to run: 'coverage' (entity coverage), 'missing' (entities without passages), 'both' (default: coverage)")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: Path for output file (only used for 'missing' analysis). If not provided, uses default path")

    args = parser.parse_args()

    if args.analysis in ["coverage", "both"]:
        print("\n" + "="*80)
        print("Running COVERAGE analysis...")
        print("="*80 + "\n")
        calculate_coverage(dataset=args.dataset, qrel_file_path=args.qrel_file)

    if args.analysis in ["missing", "both"]:
        print("\n" + "="*80)
        print("Running MISSING ENTITIES analysis...")
        print("="*80 + "\n")
        analyze_entities_without_passages(
            dataset=args.dataset,
            qrel_file_path=args.qrel_file,
            output_file_path=args.output_file
        )


# ============================================================================
# Usage Examples
# ============================================================================
#
# 1. Calculate entity coverage:
#    python c3_qrel_generation/qrel_analysis.py --dataset quest --analysis coverage
#
# 2. Find entities without relevant passages:
#    python c3_qrel_generation/qrel_analysis.py --dataset quest --analysis missing
#
# 3. Run both analyses:
#    python c3_qrel_generation/qrel_analysis.py --dataset quest --analysis both
#
# 4. With custom qrel file path:
#    python c3_qrel_generation/qrel_analysis.py --dataset qald10 --analysis both \
#           --qrel_file path/to/custom_qrels.txt
#
# 5. With custom output file for missing entities analysis:
#    python c3_qrel_generation/qrel_analysis.py --dataset quest --analysis missing \
#           --output_file path/to/custom_output.jsonl
#
# ============================================================================
