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

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
    qrel_file_path: Optional[str] = None,
    subset: Optional[str] = None
) -> Dict:
    """
    Calculate entity coverage given a dataset name and qrel file.

    Entity coverage is defined as the percentage of entities that have at least
    one relevant passage in the qrel file.

    Args:
        dataset: Dataset name (e.g., 'quest', 'qald10')
        qrel_file_path: Optional path to the qrel file. If not provided, uses default path
                       based on dataset name
        subset: Optional subset name (e.g., 'test_quest', 'train_quest'). If not provided,
               defaults to 'test_quest' for quest dataset

    Returns:
        Dictionary containing:
            - total_entities: Total number of entities in the list
            - entities_with_passages: Number of entities with at least one relevant passage
            - coverage_percentage: Percentage of entities covered
            - covered_entities: Set of entity IDs that have coverage
            - entities_without_wikipedia: Set of entity IDs that don't have Wikipedia pages
    """
    # Determine subset and subdirectory
    # User can pass short form (e.g., "test") or full form (e.g., "test_quest")
    # For quest dataset: test -> test/, train -> train/, val -> val/
    # For other datasets (e.g., qald10): no subdirectory
    if subset:
        # If subset contains underscore, assume it's full form (e.g., "test_quest")
        if "_" in subset:
            subset_name = subset
            subdir = subset.split("_")[0]  # Extract prefix as subdir
        else:
            # Short form (e.g., "test") - construct full name
            subdir = subset
            subset_name = f"{subset}_{dataset}"  # e.g., "test_quest"
    else:
        # Default mapping: quest -> test_quest, qald10 -> qald10
        if dataset == "quest":
            subset_name = "test_quest"
            subdir = "test"
        else:
            subset_name = dataset
            subdir = ""

    # Construct file paths
    if subdir:
        generations_file = f"corpus_datasets/dataset_creation_heydar/{dataset}/{subdir}/{subset_name}_generations.jsonl"
    else:
        generations_file = f"corpus_datasets/dataset_creation_heydar/{dataset}/{subset_name}_generations.jsonl"

    if qrel_file_path is None:
        if subdir:
            qrel_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/{subdir}/qrels_{subset_name}.txt"
        else:
            qrel_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/qrels_{subset_name}.txt"

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
            'entities_without_wikipedia': entities_without_wikipedia,
            'total_queries': len(queries),
            'queries_with_full_coverage': 0,
            'query_coverage_percentage': 0.0,
            'valid_query_indices': []
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

    # Calculate query-level coverage: how many queries have all entities covered
    queries_with_full_coverage = 0
    total_queries = len(queries)
    valid_query_indices: List[int] = []

    for idx, query_obj in enumerate(queries):
        # Extract entities for this query
        if 'property' in query_obj and isinstance(query_obj['property'], dict):
            entities_values = query_obj['property'].get('entities_values', [])
        else:
            entities_values = query_obj.get('entities_values', [])

        # Check if all entities in this query are covered
        all_covered = True
        for entity in entities_values:
            entity_id = entity.get('entity_id')
            if entity_id:
                # Entity is covered if it has a Wikipedia page AND that page has relevant passages
                if entity_id not in entity_to_pageid:
                    all_covered = False
                    break
                pageid = entity_to_pageid[entity_id]
                if pageid not in relevant_page_ids:
                    all_covered = False
                    break

        if all_covered:
            queries_with_full_coverage += 1
            valid_query_indices.append(idx)

    query_coverage_percentage = (queries_with_full_coverage / total_queries * 100) if total_queries > 0 else 0.0

    # Calculate entity-level coverage
    total_entities = len(entity_list)
    entities_with_passages = len(covered_entities)
    coverage_percentage = (entities_with_passages / total_entities * 100) if total_entities > 0 else 0.0

    results = {
        'total_entities': total_entities,
        'entities_with_passages': entities_with_passages,
        'coverage_percentage': coverage_percentage,
        'covered_entities': covered_entities,
        'entities_without_wikipedia': entities_without_wikipedia,
        'total_queries': total_queries,
        'queries_with_full_coverage': queries_with_full_coverage,
        'query_coverage_percentage': query_coverage_percentage,
        'valid_query_indices': valid_query_indices
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
    print("-"*80)
    print(f"Total queries: {total_queries}")
    print(f"Queries with full entity coverage: {queries_with_full_coverage}")
    print(f"Query coverage (all entities covered): {query_coverage_percentage:.2f}%")
    print("="*80 + "\n")

    return results


def analyze_entity_list_distribution(
    dataset: str,
    properties_file_path: Optional[str] = None,
    subset: Optional[str] = None
) -> Dict:
    """
    Analyze the distribution of entity list lengths from the properties file.

    Reads the intermediate_qids field from *_properties.jsonl files and computes
    statistics about the distribution of entity list lengths across queries.

    Args:
        dataset: Dataset name (e.g., 'quest', 'qald10')
        properties_file_path: Optional path to the properties file. If not provided,
                             uses default path based on dataset name
        subset: Optional subset name (e.g., 'test_quest', 'train_quest'). If not provided,
               defaults to 'test_quest' for quest dataset

    Returns:
        Dictionary containing:
            - total_queries: Total number of queries
            - length_distribution: Dict mapping length to count
            - min_length: Minimum entity list length
            - max_length: Maximum entity list length
            - mean_length: Mean entity list length
            - median_length: Median entity list length
    """
    # Determine subset and subdirectory
    # User can pass short form (e.g., "test") or full form (e.g., "test_quest")
    # For quest dataset: test -> test/, train -> train/, val -> val/
    # For other datasets (e.g., qald10): no subdirectory
    if subset:
        # If subset contains underscore, assume it's full form (e.g., "test_quest")
        if "_" in subset:
            subset_name = subset
            subdir = subset.split("_")[0]  # Extract prefix as subdir
        else:
            # Short form (e.g., "test") - construct full name
            subdir = subset
            subset_name = f"{subset}_{dataset}"  # e.g., "test_quest"
    else:
        # Default mapping: quest -> test_quest, qald10 -> qald10
        if dataset == "quest":
            subset_name = "test_quest"
            subdir = "test"
        else:
            subset_name = dataset
            subdir = ""

    # Construct file path
    if properties_file_path is None:
        if subdir:
            properties_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/{subdir}/{subset_name}_with_properties.jsonl"
        else:
            properties_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/{subset_name}_with_properties.jsonl"

    print(f"Loading properties from {properties_file_path}...")

    if not Path(properties_file_path).exists():
        print(f"Error: Properties file does not exist: {properties_file_path}")
        return {}

    queries = read_jsonl_from_file(properties_file_path)
    print(f"Loaded {len(queries)} queries")

    # Collect entity list lengths
    lengths = []
    for query_obj in queries:
        intermediate_qids = query_obj.get('intermediate_qids', [])
        lengths.append(len(intermediate_qids))

    if not lengths:
        print("No queries found with intermediate_qids")
        return {}

    # Compute distribution
    length_distribution = {}
    for length in lengths:
        length_distribution[length] = length_distribution.get(length, 0) + 1

    # Compute statistics
    min_length = min(lengths)
    max_length = max(lengths)
    mean_length = sum(lengths) / len(lengths)
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    if n % 2 == 0:
        median_length = (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2
    else:
        median_length = sorted_lengths[n // 2]

    results = {
        'total_queries': len(queries),
        'length_distribution': length_distribution,
        'min_length': min_length,
        'max_length': max_length,
        'mean_length': mean_length,
        'median_length': median_length
    }

    # Print results
    print("\n" + "=" * 80)
    print("ENTITY LIST LENGTH DISTRIBUTION")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Total queries: {len(queries)}")
    print(f"Min entity list length: {min_length}")
    print(f"Max entity list length: {max_length}")
    print(f"Mean entity list length: {mean_length:.2f}")
    print(f"Median entity list length: {median_length:.1f}")
    print("\nLength Distribution:")
    for length in sorted(length_distribution.keys()):
        count = length_distribution[length]
        percentage = count / len(queries) * 100
        bar = "#" * int(percentage / 2)
        print(f"  {length:4d}: {count:5d} ({percentage:5.1f}%) {bar}")
    print("=" * 80 + "\n")

    return results


def analyze_entities_without_passages(
    dataset: str,
    qrel_file_path: Optional[str] = None,
    output_file_path: Optional[str] = None,
    subset: Optional[str] = None
) -> None:
    """
    Analyze queries to find entities without relevant passages.

    For each query, identifies which entities don't have any relevant passages
    and writes the results to a JSONL file.

    Args:
        dataset: Dataset name (e.g., 'quest', 'qald10')
        qrel_file_path: Optional path to the qrel file. If not provided, uses default path
        output_file_path: Optional path for output JSONL. If not provided, uses default path
        subset: Optional subset name (e.g., 'test_quest', 'train_quest'). If not provided,
               defaults to 'test_quest' for quest dataset
    """
    # Determine subset and subdirectory
    # User can pass short form (e.g., "test") or full form (e.g., "test_quest")
    # For quest dataset: test -> test/, train -> train/, val -> val/
    # For other datasets (e.g., qald10): no subdirectory
    if subset:
        # If subset contains underscore, assume it's full form (e.g., "test_quest")
        if "_" in subset:
            subset_name = subset
            subdir = subset.split("_")[0]  # Extract prefix as subdir
        else:
            # Short form (e.g., "test") - construct full name
            subdir = subset
            subset_name = f"{subset}_{dataset}"  # e.g., "test_quest"
    else:
        # Default mapping: quest -> test_quest, qald10 -> qald10
        if dataset == "quest":
            subset_name = "test_quest"
            subdir = "test"
        else:
            subset_name = dataset
            subdir = ""

    # Construct file paths
    if subdir:
        generations_file = f"corpus_datasets/dataset_creation_heydar/{dataset}/{subdir}/{subset_name}_generations.jsonl"
    else:
        generations_file = f"corpus_datasets/dataset_creation_heydar/{dataset}/{subset_name}_generations.jsonl"

    if qrel_file_path is None:
        if subdir:
            qrel_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/{subdir}/qrels_{subset_name}.txt"
        else:
            qrel_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/qrels_{subset_name}.txt"

    if output_file_path is None:
        if subdir:
            output_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/{subdir}/entities_without_passages_{subset_name}.jsonl"
        else:
            output_file_path = f"corpus_datasets/dataset_creation_heydar/{dataset}/entities_without_passages_{subset_name}.jsonl"

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
    parser.add_argument("--dataset", type=str, default="qald10", choices=["quest", "qald10"], help="Dataset name (e.g., 'quest', 'qald10')")
    parser.add_argument("--subset", type=str, default=None, help="Subset name (e.g., 'test_quest', 'train_quest'). If not provided, defaults to 'test_quest' for quest dataset")
    parser.add_argument("--qrel_file", type=str, default=None, help="Optional: Path to qrel file. If not provided, uses default path based on dataset/subset")
    parser.add_argument("--analysis", type=str, default="coverage", choices=["coverage", "missing", "entity_distribution", "both", "all"], help="Type of analysis to run: 'coverage' (entity coverage), 'missing' (entities without passages), 'entity_distribution' (entity list length distribution), 'both' (coverage + missing), 'all' (all analyses) (default: coverage)")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: Path for output file (only used for 'missing' analysis). If not provided, uses default path")

    args = parser.parse_args()

    if args.analysis in ["coverage", "both", "all"]:
        print("\n" + "="*80)
        print("Running COVERAGE analysis...")
        print("="*80 + "\n")
        calculate_coverage(dataset=args.dataset, qrel_file_path=args.qrel_file, subset=args.subset)

    if args.analysis in ["missing", "both", "all"]:
        print("\n" + "="*80)
        print("Running MISSING ENTITIES analysis...")
        print("="*80 + "\n")
        analyze_entities_without_passages(
            dataset=args.dataset,
            qrel_file_path=args.qrel_file,
            output_file_path=args.output_file,
            subset=args.subset
        )

    if args.analysis in ["entity_distribution", "all"]:
        print("\n" + "="*80)
        print("Running ENTITY LIST DISTRIBUTION analysis...")
        print("="*80 + "\n")
        analyze_entity_list_distribution(dataset=args.dataset, subset=args.subset)


# ============================================================================
# Usage Examples
# ============================================================================
#
# 1. Calculate entity coverage:
#    python c3_qrel_generation/qrel_analysis.py --dataset qald10 --analysis coverage
#
# 2. Find entities without relevant passages:
#    python c3_qrel_generation/qrel_analysis.py --dataset quest --subset val_quest --analysis coverage
#
# 3. Analyze entity list length distribution:
#    python c3_qrel_generation/qrel_analysis.py --dataset qald10 --analysis entity_distribution
#
# 4. Run coverage and missing analyses:
#    python c3_qrel_generation/qrel_analysis.py --dataset quest --analysis both
#
# 5. Run all analyses:
#    python c3_qrel_generation/qrel_analysis.py --dataset quest --analysis all
#
# 6. With custom qrel file path:
#    python c3_qrel_generation/qrel_analysis.py --dataset qald10 --analysis both \
#           --qrel_file path/to/custom_qrels.txt
#
# 7. With custom output file for missing entities analysis:
#    python c3_qrel_generation/qrel_analysis.py --dataset quest --analysis missing \
#           --output_file path/to/custom_output.jsonl
#
# ============================================================================
