"""
QRel Generation for Total Recall RAG

This script generates TREC-format qrels by:
1. Reading queries from generations JSONL file (e.g., quest_generations.jsonl)
   - Input format from c1_2_dataset_creation_heydar pipeline
   - Contains full metadata including entity IDs, labels, property info, and values
2. For each query, extracting entity QIDs from property['entities_values']
3. Mapping Wikidata QIDs to Wikipedia pages
4. Retrieving passages from those Wikipedia pages
5. Using LLM to judge relevance of passages for the given property
6. Writing qrels in TREC format
7. Tracking entity coverage for Total Recall analysis

Outputs:
- qrels_*.txt: TREC-format qrels (query_id 0 passage_id relevance)
- qrels_*_entity_coverage.jsonl: Per-query entity coverage analysis
  Each line contains:
  {
    "query_id": "quest_1_p136",
    "query": "How many films...",
    "property": "genre (P136)",
    "total_entities": 10,
    "entities_with_coverage": 8,
    "entities_without_coverage": 2,
    "entities_with_relevant_passages": [
      {"entity_id": "Q123", "entity_label": "Film 1", "entity_value": ["thriller"], "relevant_passage_count": 3},
      ...
    ],
    "entities_without_relevant_passages": [
      {"entity_id": "Q456", "entity_label": "Film 2", "entity_value": ["drama"], "reason": "no_relevant_passages_found"},
      ...
    ]
  }

Input Format (from c1_2_dataset_creation_heydar):
{
    "qid": "quest_1_p136",
    "original_query": "...",
    "property": {
        "property_info": {
            "label": "genre",
            "property_id": "P136",
            "description": "...",
            "datatype": "WikibaseItem"
        },
        "entities_values": [
            {
                "entity_id": "Q4193501",
                "entity_label": "1",
                "value": ["thriller film"]
            },
            ...
        ]
    },
    "operation": "COUNT",
    "ground_truth": 5,
    ...
}
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool, Manager, Lock
from functools import partial

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file, read_json_from_file
from c1_1_dataset_creation_mahta.query_generation.prompts.LLM_as_relevance_judge.prompt_property_check import (
    PROPERTY_PROMPT,
    PROPERTY_INPUT_TEMPLATE
)


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

def get_passages_for_page(
    pageid: str,
    corpus_jsonl_path: str,
    page2passage_mapping: Optional[Dict] = None
) -> List[Dict]:
    """
    Get all passages for a given Wikipedia page ID from corpus.

    Args:
        pageid: Wikipedia page ID
        corpus_jsonl_path: Path to corpus JSONL file
        page2passage_mapping: Optional pre-built mapping from page ID to line numbers

    Returns:
        List of passage dictionaries with 'id', 'title', 'contents'
    """
    passages = []

    # If we have a pre-built mapping, use it for efficiency
    if page2passage_mapping and pageid in page2passage_mapping:
        line_numbers = page2passage_mapping[pageid]

        # Sort line numbers to read sequentially (more efficient file access)
        sorted_line_numbers = sorted(line_numbers)
        target_lines = set(sorted_line_numbers)

        with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
            current_line = 0
            for target_line_num in sorted_line_numbers:
                # Skip lines until we reach the target
                while current_line < target_line_num:
                    f.readline()
                    current_line += 1

                # Read the target line
                line = f.readline()
                current_line += 1

                if line:
                    passage = json.loads(line.strip())
                    passages.append(passage)

        return passages

    # Otherwise, scan the corpus file (SLOW - not recommended for large corpora)
    # Passages have IDs like "pageid-0000", "pageid-0001", etc.
    print(f"Warning: No index mapping found for page {pageid}. Falling back to full corpus scan (slow).", file=sys.stderr)
    with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            passage = json.loads(line.strip())
            passage_id = passage.get('id', '')
            # Check if passage belongs to this page
            if passage_id.startswith(f"{pageid}-"):
                passages.append(passage)

    return passages

def judge_passage_relevance(
    passage_content: str,
    entity_label: str,
    property_label: str,
    property_description: str,
    client: OpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0
) -> str:
    """
    Use LLM to judge if passage contains information about the property.

    Args:
        passage_content: The text content of the passage
        entity_label: Name of the entity (e.g., film title)
        property_label: Property name (e.g., "publication date")
        property_description: Description of the property
        client: OpenAI client
        model: Model to use for judgment
        temperature: Temperature for generation

    Returns:
        "YES" or "NO"
    """
    input_instance = PROPERTY_INPUT_TEMPLATE.format(
        entity_name=entity_label,
        property_name=property_label,
        property_description=property_description,
        passage=passage_content
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROPERTY_PROMPT},
                {"role": "user", "content": input_instance}
            ],
            temperature=temperature,
        )

        response = resp.choices[0].message.content.strip()
        if response not in ["YES", "NO"]:
            print(f"Warning: Unexpected response '{response}', defaulting to NO", file=sys.stderr)
            return "NO"
        return response

    except Exception as e:
        print(f"Error in LLM judgment: {e}", file=sys.stderr)
        return "NO"

def write_trec_qrel(
    output_file,
    query_id: str,
    passage_id: str,
    relevance: int
):
    """
    Write a single qrel line in TREC format.
    Format: query_id iteration passage_id relevance

    Args:
        output_file: Open file handle to write to
        query_id: Query identifier
        passage_id: Document/passage identifier
        relevance: Relevance score (0 or 1)
    """
    # TREC format: query_id 0 doc_id relevance
    output_file.write(f"{query_id} 0 {passage_id} {relevance}\n")

def process_query_core(
    query_obj: Dict,
    corpus_jsonl_path: str,
    client: OpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    page2passage_mapping: Optional[Dict] = None,
    qid2wikipedia_cache: Optional[Dict] = None
) -> Tuple[Dict, List[str], List[str], str]:
    """
    Process a single query to generate qrels (core logic without file I/O).

    Args:
        query_obj: Query object from JSONL (generations format from c1_2_dataset_creation_heydar)
        corpus_jsonl_path: Path to corpus JSONL file
        client: OpenAI client
        model: LLM model to use
        temperature: Temperature for LLM
        page2passage_mapping: Optional mapping for efficiency
        qid2wikipedia_cache: Optional cache for QID->Wikipedia mappings

    Returns:
        Tuple of (stats, qrel_lines, log_lines, entity_coverage_json)
        - stats: Statistics dict with entity coverage information
        - qrel_lines: List of qrel lines to write
        - log_lines: List of log lines to write
        - entity_coverage_json: JSON string of entity coverage
    """
    query_id = query_obj['qid']

    # Extract entity QIDs and their values from the new nested structure
    property_info = query_obj['property']
    entities_values = property_info.get('entities_values', [])

    # Build list of QIDs and their labels/values
    total_recall_qids = [entity['entity_id'] for entity in entities_values]
    total_recall_qids_with_values = {
        entity['entity_id']: {
            'label': entity['entity_label'],
            'value': entity.get('value', 'N/A')
        }
        for entity in entities_values
    }

    # Extract property metadata from nested property_info structure
    property_metadata = property_info['property_info']
    property_label = property_metadata['label']
    property_description = property_metadata['description']
    property_id = property_metadata['property_id']
    property_datatype = property_metadata.get('datatype', 'Unknown')

    stats = {
        'total_qids': len(total_recall_qids),
        'qids_with_wikipedia': 0,
        'total_passages_checked': 0,
        'relevant_passages': 0
    }

    # Track entity coverage for this query
    entity_coverage = {
        'query_id': query_id,
        'query': query_obj.get('question', query_obj.get('original_query', 'N/A')),
        'property': f"{property_label} ({property_id})",
        'total_entities': len(total_recall_qids),
        'entities_with_relevant_passages': [],
        'entities_without_relevant_passages': []
    }

    # Collect log lines and qrel lines instead of writing directly
    log_lines = []
    qrel_lines = []

    log_lines.append(f"\n{'='*80}\n")
    log_lines.append(f"Query ID: {query_id}\n")
    log_lines.append(f"Query: {query_obj.get('question', query_obj.get('original_query', 'N/A'))}\n")
    log_lines.append(f"Property: {property_label} ({property_id}) - {property_description}\n")
    log_lines.append(f"Property Datatype: {property_datatype}\n")
    log_lines.append(f"Operation: {query_obj.get('operation', 'N/A')}\n")
    log_lines.append(f"Subclass: {query_obj.get('subclass_label', 'N/A')} ({query_obj.get('subclass_id', 'N/A')})\n")
    log_lines.append(f"Ground Truth: {query_obj.get('ground_truth', 'N/A')}\n")
    log_lines.append(f"{'='*80}\n\n")

    # Process each QID
    for qid in total_recall_qids:
        qid_info = total_recall_qids_with_values.get(qid, {})
        qid_label = qid_info.get('label', 'N/A') if isinstance(qid_info, dict) else qid_info
        qid_value = qid_info.get('value', 'N/A') if isinstance(qid_info, dict) else 'N/A'
        log_lines.append(f"\n--- Processing QID: {qid} | Label: {qid_label} | Value: {qid_value} ---\n")

        # Track if this entity has at least one relevant passage
        entity_has_relevant_passage = False

        # Get Wikipedia page info
        wiki_info = None
        if qid2wikipedia_cache and qid in qid2wikipedia_cache:
            wiki_info = qid2wikipedia_cache[qid]
        else:
            wiki_info = get_wikipedia_info_from_qid(qid)
            if qid2wikipedia_cache is not None:
                qid2wikipedia_cache[qid] = wiki_info

        if not wiki_info:
            log_lines.append(f"No Wikipedia page found for {qid}\n")
            entity_coverage['entities_without_relevant_passages'].append({
                'entity_id': qid,
                'entity_label': qid_label,
                'entity_value': qid_value,
                'reason': 'no_wikipedia_page'
            })
            continue

        stats['qids_with_wikipedia'] += 1
        entity_title = wiki_info['title']
        pageid = wiki_info['pageid']

        log_lines.append(f"Wikipedia Title: {entity_title}\n")
        log_lines.append(f"Wikipedia Page ID: {pageid}\n")

        # Get passages for this page
        passages = get_passages_for_page(pageid, corpus_jsonl_path, page2passage_mapping)

        if not passages:
            log_lines.append(f"No passages found for page {pageid}\n")
            entity_coverage['entities_without_relevant_passages'].append({
                'entity_id': qid,
                'entity_label': qid_label,
                'entity_value': qid_value,
                'reason': 'no_passages_in_corpus'
            })
            continue

        log_lines.append(f"Found {len(passages)} passages\n")
        stats['total_passages_checked'] += len(passages)

        # Track relevant passages for this entity
        entity_relevant_passage_count = 0

        # Check each passage with LLM
        for passage in passages:
            passage_id = passage['id']
            passage_content = passage['contents']

            # Judge relevance
            judgment = judge_passage_relevance(
                passage_content=passage_content,
                entity_label=entity_title,
                property_label=property_label,
                property_description=property_description,
                client=client,
                model=model,
                temperature=temperature
            )

            # Write qrel only when relevance is 1
            relevance = 1 if judgment == "YES" else 0

            if relevance == 1:
                log_lines.append(f"✓ RELEVANT - Passage ID: {passage_id}\n")
                log_lines.append(f"Full Content:\n{passage_content}\n\n")

                # Add qrel line instead of writing directly
                qrel_lines.append(f"{query_id} 0 {passage_id} {relevance}\n")
                stats['relevant_passages'] += 1
                entity_has_relevant_passage = True
                entity_relevant_passage_count += 1
            else:
                # For non-relevant passages, just log the ID
                log_lines.append(f"✗ NOT RELEVANT - Passage ID: {passage_id} - Content: {passage_content[:100]}...\n")

        # Record entity coverage results
        if entity_has_relevant_passage:
            entity_coverage['entities_with_relevant_passages'].append({
                'entity_id': qid,
                'entity_label': qid_label,
                'entity_value': qid_value,
                'relevant_passage_count': entity_relevant_passage_count
            })
        else:
            entity_coverage['entities_without_relevant_passages'].append({
                'entity_id': qid,
                'entity_label': qid_label,
                'entity_value': qid_value,
                'reason': 'no_relevant_passages_found'
            })

    # Add coverage counts to entity_coverage
    entity_coverage['entities_with_coverage'] = len(entity_coverage['entities_with_relevant_passages'])
    entity_coverage['entities_without_coverage'] = len(entity_coverage['entities_without_relevant_passages'])

    # Log entity coverage summary for this query
    log_lines.append(f"\n{'='*80}\n")
    log_lines.append(f"ENTITY COVERAGE SUMMARY FOR QUERY: {query_id}\n")
    log_lines.append(f"{'='*80}\n")
    log_lines.append(f"Total entities: {entity_coverage['total_entities']}\n")
    log_lines.append(f"Entities with relevant passages: {entity_coverage['entities_with_coverage']} ({100*entity_coverage['entities_with_coverage']/entity_coverage['total_entities']:.1f}%)\n")
    log_lines.append(f"Entities without relevant passages: {entity_coverage['entities_without_coverage']} ({100*entity_coverage['entities_without_coverage']/entity_coverage['total_entities']:.1f}%)\n")

    if entity_coverage['entities_without_relevant_passages']:
        log_lines.append(f"\nEntities without coverage:\n")
        for entity in entity_coverage['entities_without_relevant_passages']:
            log_lines.append(f"  - {entity['entity_id']} ({entity['entity_label']}): {entity['reason']}\n")

    # Also add to stats for backward compatibility
    stats['entity_coverage'] = entity_coverage

    # Return data instead of writing to files
    entity_coverage_json = json.dumps(entity_coverage)
    return stats, qrel_lines, log_lines, entity_coverage_json


def process_query_wrapper(
    query_obj: Dict,
    corpus_jsonl_path: str,
    model: str,
    temperature: float,
    page2passage_mapping: Optional[Dict],
    qid2wikipedia_cache: Optional[Dict],
    api_key: str
) -> Tuple[Dict, List[str], List[str], str]:
    """
    Wrapper function for multiprocessing that creates its own OpenAI client.

    Args:
        query_obj: Query object
        corpus_jsonl_path: Path to corpus
        model: LLM model
        temperature: Temperature
        page2passage_mapping: Page to passage mapping
        qid2wikipedia_cache: QID to Wikipedia cache
        api_key: OpenAI API key

    Returns:
        Tuple of (stats, qrel_lines, log_lines, entity_coverage_json)
    """
    # Create a new OpenAI client for this process
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    return process_query_core(
        query_obj=query_obj,
        corpus_jsonl_path=corpus_jsonl_path,
        client=client,
        model=model,
        temperature=temperature,
        page2passage_mapping=page2passage_mapping,
        qid2wikipedia_cache=qid2wikipedia_cache
    )


def main(args):
    """Main function to generate qrels."""

    # Load queries
    print(f"Loading queries from {args.query_file}...")
    queries = read_jsonl_from_file(args.query_file)
    print(f"Loaded {len(queries)} queries")

    # Load optional page2passage mapping if provided
    page2passage_mapping = None
    if args.page2passage_mapping:
        print(f"Loading page2passage mapping from {args.page2passage_mapping}...")
        page2passage_mapping = read_json_from_file(args.page2passage_mapping)
        print(f"Loaded mapping for {len(page2passage_mapping)} pages")
    else:
        print("\nWARNING: No page2passage mapping provided!")
        print("This will result in VERY SLOW performance as the entire corpus will be scanned for each entity.")
        print("To build an index, run:")
        print(f"  python c2_corpus_annotation/build_page_index.py --corpus_jsonl {args.corpus_jsonl}")
        print("Then use the generated index file with --page2passage_mapping\n")

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Create output directory if needed
    output_dir = Path(args.output_qrel).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create entity coverage output file path
    entity_coverage_file = args.output_qrel.replace('.txt', '_entity_coverage.jsonl')

    # Use multiprocessing Manager for shared cache
    with Manager() as manager:
        # Create a shared cache for QID to Wikipedia mappings
        qid2wikipedia_cache = manager.dict()

        # Create partial function with fixed parameters
        process_func = partial(
            process_query_wrapper,
            corpus_jsonl_path=args.corpus_jsonl,
            model=args.model,
            temperature=args.temperature,
            page2passage_mapping=page2passage_mapping,
            qid2wikipedia_cache=qid2wikipedia_cache,
            api_key=api_key
        )

        # Process queries in parallel
        all_stats = []
        all_qrel_lines = []
        all_log_lines = []
        all_coverage_jsons = []

        print(f"Processing {len(queries)} queries with {args.num_workers} parallel workers...")

        if args.num_workers > 1:
            # Use multiprocessing Pool with imap_unordered for faster progress updates
            with Pool(processes=args.num_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(process_func, queries, chunksize=1),
                    total=len(queries),
                    desc="Processing queries"
                ))
        else:
            # Sequential processing (for debugging)
            results = [process_func(q) for q in tqdm(queries, desc="Processing queries")]

        # Unpack results
        for stats, qrel_lines, log_lines, coverage_json in results:
            all_stats.append(stats)
            all_qrel_lines.extend(qrel_lines)
            all_log_lines.extend(log_lines)
            all_coverage_jsons.append(coverage_json)

    # Write all results to files
    print("Writing results to files...")
    with open(args.output_qrel, 'w', encoding='utf-8') as qrel_file, \
         open(args.log_file, 'w', encoding='utf-8') as log_file, \
         open(entity_coverage_file, 'w', encoding='utf-8') as coverage_file:

        # Write log header
        log_file.write(f"QRel Generation Log\n")
        log_file.write(f"Query file: {args.query_file}\n")
        log_file.write(f"Corpus: {args.corpus_jsonl}\n")
        log_file.write(f"Model: {args.model}\n")
        log_file.write(f"Temperature: {args.temperature}\n")
        log_file.write(f"Parallel workers: {args.num_workers}\n")
        log_file.write(f"\n{'='*80}\n")

        # Write all qrel lines
        qrel_file.writelines(all_qrel_lines)

        # Write all log lines
        log_file.writelines(all_log_lines)

        # Write all coverage JSONs
        for coverage_json in all_coverage_jsons:
            coverage_file.write(coverage_json + '\n')

        # Write summary
        log_file.write(f"\n\n{'='*80}\n")
        log_file.write("SUMMARY\n")
        log_file.write(f"{'='*80}\n")

        total_queries = len(queries)
        total_qids = sum(s['total_qids'] for s in all_stats)
        total_with_wiki = sum(s['qids_with_wikipedia'] for s in all_stats)
        total_passages = sum(s['total_passages_checked'] for s in all_stats)
        total_relevant = sum(s['relevant_passages'] for s in all_stats)

        # Entity coverage statistics
        total_entities_with_coverage = sum(
            s['entity_coverage']['entities_with_coverage'] for s in all_stats if 'entity_coverage' in s
        )
        total_entities_without_coverage = sum(
            s['entity_coverage']['entities_without_coverage'] for s in all_stats if 'entity_coverage' in s
        )

        # Count queries by coverage status
        queries_full_coverage = sum(
            1 for s in all_stats
            if 'entity_coverage' in s and s['entity_coverage']['entities_without_coverage'] == 0
        )
        queries_partial_coverage = sum(
            1 for s in all_stats
            if 'entity_coverage' in s
            and s['entity_coverage']['entities_with_coverage'] > 0
            and s['entity_coverage']['entities_without_coverage'] > 0
        )
        queries_no_coverage = sum(
            1 for s in all_stats
            if 'entity_coverage' in s and s['entity_coverage']['entities_with_coverage'] == 0
        )

        log_file.write(f"Total queries processed: {total_queries}\n")
        log_file.write(f"Total QIDs: {total_qids}\n")
        log_file.write(f"QIDs with Wikipedia pages: {total_with_wiki}\n")
        log_file.write(f"Total passages checked: {total_passages}\n")
        log_file.write(f"Relevant passages found: {total_relevant}\n")
        log_file.write(f"\n--- Entity Coverage (Total Recall) ---\n")
        log_file.write(f"Total entities: {total_qids}\n")
        log_file.write(f"Entities with relevant passages: {total_entities_with_coverage} ({100*total_entities_with_coverage/total_qids:.1f}%)\n")
        log_file.write(f"Entities without relevant passages: {total_entities_without_coverage} ({100*total_entities_without_coverage/total_qids:.1f}%)\n")
        log_file.write(f"\n--- Query Coverage Analysis ---\n")
        log_file.write(f"Queries with full entity coverage: {queries_full_coverage} ({100*queries_full_coverage/total_queries:.1f}%)\n")
        log_file.write(f"Queries with partial entity coverage: {queries_partial_coverage} ({100*queries_partial_coverage/total_queries:.1f}%)\n")
        log_file.write(f"Queries with no entity coverage: {queries_no_coverage} ({100*queries_no_coverage/total_queries:.1f}%)\n")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total queries processed: {total_queries}")
        print(f"Total QIDs: {total_qids}")
        print(f"QIDs with Wikipedia pages: {total_with_wiki}")
        print(f"Total passages checked: {total_passages}")
        print(f"Relevant passages found: {total_relevant}")
        print(f"\n--- Entity Coverage (Total Recall) ---")
        print(f"Total entities: {total_qids}")
        print(f"Entities with relevant passages: {total_entities_with_coverage} ({100*total_entities_with_coverage/total_qids:.1f}%)")
        print(f"Entities without relevant passages: {total_entities_without_coverage} ({100*total_entities_without_coverage/total_qids:.1f}%)")
        print(f"\n--- Query Coverage Analysis ---")
        print(f"Queries with full entity coverage: {queries_full_coverage} ({100*queries_full_coverage/total_queries:.1f}%)")
        print(f"Queries with partial entity coverage: {queries_partial_coverage} ({100*queries_partial_coverage/total_queries:.1f}%)")
        print(f"Queries with no entity coverage: {queries_no_coverage} ({100*queries_no_coverage/total_queries:.1f}%)")
        print(f"\nQRels written to: {args.output_qrel}")
        print(f"Entity coverage written to: {entity_coverage_file}")
        print(f"Log written to: {args.log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TREC-format qrels for Total Recall RAG queries"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="qald10",
        choices=["quest", "qald10"],
        help="Dataset name (e.g., 'quest', 'custom'). Used to construct file paths."
    )

    parser.add_argument(
        "--corpus_jsonl",
        type=str,
        default="/projects/0/prjs0834/heydars/INDICES/enwiki_20251001.jsonl",
        help="Path to corpus JSONL file with passages"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="qrel_logging",
        help="Directory to write log file (log filename will be auto-generated based on dataset)"
    )

    parser.add_argument(
        "--page2passage_mapping",
        type=str,
        default="/projects/0/prjs0834/heydars/INDICES/enwiki_20251001.index.json",
        help="Optional: Path to JSON file mapping page IDs to passage indices for faster lookup"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for relevance judgment (default: gpt-4o)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (default: 0.0 for consistency)"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing queries (default: 1 for sequential processing)"
    )

    args = parser.parse_args()

    # Construct file paths based on dataset name
    dataset_name = "test_quest" if args.dataset == "quest" else args.dataset
    # Map dataset name to directory name (quest -> test_quest)
    dir_name = args.dataset
    # Use the generations file which contains full metadata including entity values
    args.query_file = f"corpus_datasets/dataset_creation_heydar/{dir_name}/{dataset_name}_generations.jsonl"
    args.output_qrel = f"corpus_datasets/dataset_creation_heydar/{dir_name}/qrels_{dataset_name}.txt"

    # Create log file path based on dataset name with timestamp
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_file = str(log_dir / f"qrel_generation_{dataset_name}_{timestamp}.log")

    # Print configuration
    print("="*80)
    print("QRel Generation Configuration")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Query file: {args.query_file}")
    print(f"Output qrel: {args.output_qrel}")
    print(f"Log file: {args.log_file}")
    print(f"Corpus: {args.corpus_jsonl}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Parallel workers: {args.num_workers}")
    print("="*80)
    print()

    main(args)

    # Usage examples:
    # Sequential (1 worker):
    # python c2_corpus_annotation/qrel_generation.py --dataset qald10

    # Parallel (20 workers):
    # python c2_corpus_annotation/qrel_generation.py --dataset qald10 --num_workers 32

    # Parallel (50 workers):
    # python c2_corpus_annotation/qrel_generation.py --dataset qald10 --num_workers 50 
