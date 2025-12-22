"""
QRel Generation for Total Recall RAG

This script generates TREC-format qrels by:
1. Reading queries from JSONL file (e.g., test_quest_queries.jsonl)
2. For each query, mapping Wikidata QIDs to Wikipedia pages
3. Retrieving passages from those Wikipedia pages
4. Using LLM to judge relevance of passages for the given property
5. Writing qrels in TREC format
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions from c1_1_dataset_creation_mahta
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
        page2passage_mapping: Optional pre-built mapping from page ID to passage indices

    Returns:
        List of passage dictionaries with 'id', 'title', 'contents'
    """
    passages = []

    # If we have a pre-built mapping, use it for efficiency
    if page2passage_mapping and pageid in page2passage_mapping:
        passage_indices = page2passage_mapping[pageid]
        with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num in passage_indices:
                    passage = json.loads(line.strip())
                    passages.append(passage)
        return passages

    # Otherwise, scan the corpus file
    # Passages have IDs like "pageid-0000", "pageid-0001", etc.
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


def process_query(
    query_obj: Dict,
    corpus_jsonl_path: str,
    client: OpenAI,
    qrel_output_file,
    log_file,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    page2passage_mapping: Optional[Dict] = None,
    qid2wikipedia_cache: Optional[Dict] = None
) -> Dict:
    """
    Process a single query to generate qrels.

    Args:
        query_obj: Query object from JSONL
        corpus_jsonl_path: Path to corpus JSONL file
        client: OpenAI client
        qrel_output_file: Open file handle for qrel output
        log_file: Open file handle for logging
        model: LLM model to use
        temperature: Temperature for LLM
        page2passage_mapping: Optional mapping for efficiency
        qid2wikipedia_cache: Optional cache for QID->Wikipedia mappings

    Returns:
        Statistics dict
    """
    query_id = query_obj['qid']
    total_recall_qids = query_obj['total_recall_qids']
    property_info = query_obj['property']
    property_label = property_info['property_label']
    property_description = property_info['property_description']

    stats = {
        'total_qids': len(total_recall_qids),
        'qids_with_wikipedia': 0,
        'total_passages_checked': 0,
        'relevant_passages': 0
    }

    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Query ID: {query_id}\n")
    log_file.write(f"Query: {query_obj['total_recall_query']}\n")
    log_file.write(f"Property: {property_label} - {property_description}\n")
    log_file.write(f"{'='*80}\n\n")

    # Process each QID
    for qid in total_recall_qids:
        log_file.write(f"\n--- Processing QID: {qid} ---\n")

        # Get Wikipedia page info
        wiki_info = None
        if qid2wikipedia_cache and qid in qid2wikipedia_cache:
            wiki_info = qid2wikipedia_cache[qid]
        else:
            wiki_info = get_wikipedia_info_from_qid(qid)
            print(wiki_info)
            print("-----")
            if qid2wikipedia_cache is not None:
                qid2wikipedia_cache[qid] = wiki_info

        if not wiki_info:
            log_file.write(f"No Wikipedia page found for {qid}\n")
            continue

        stats['qids_with_wikipedia'] += 1
        entity_title = wiki_info['title']
        pageid = wiki_info['pageid']

        log_file.write(f"Wikipedia Title: {entity_title}\n")
        log_file.write(f"Wikipedia Page ID: {pageid}\n")

        # Get passages for this page
        passages = get_passages_for_page(pageid, corpus_jsonl_path, page2passage_mapping)

        if not passages:
            log_file.write(f"No passages found for page {pageid}\n")
            continue

        log_file.write(f"Found {len(passages)} passages\n")
        stats['total_passages_checked'] += len(passages)

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

            log_file.write(f"\nPassage ID: {passage_id}\n")
            log_file.write(f"Content: {passage_content[:200]}...\n")
            log_file.write(f"Judgment: {judgment}\n")

            # Write qrel
            relevance = 1 if judgment == "YES" else 0
            write_trec_qrel(qrel_output_file, query_id, passage_id, relevance)
            qrel_output_file.flush()  # Flush immediately to disk

            if relevance == 1:
                stats['relevant_passages'] += 1

    return stats


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

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Create output directory if needed
    output_dir = Path(args.output_qrel).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache for QID to Wikipedia mappings
    qid2wikipedia_cache = {}

    # Open output files
    with open(args.output_qrel, 'w', encoding='utf-8') as qrel_file, \
         open(args.log_file, 'w', encoding='utf-8') as log_file:

        log_file.write(f"QRel Generation Log\n")
        log_file.write(f"Query file: {args.query_file}\n")
        log_file.write(f"Corpus: {args.corpus_jsonl}\n")
        log_file.write(f"Model: {args.model}\n")
        log_file.write(f"Temperature: {args.temperature}\n")
        log_file.write(f"\n{'='*80}\n")

        # Process queries
        all_stats = []
        for idx, query_obj in tqdm(enumerate(queries), desc="Processing queries"):
            
            if idx == 2:
                break
            
            stats = process_query(
                query_obj=query_obj,
                corpus_jsonl_path=args.corpus_jsonl,
                client=client,
                qrel_output_file=qrel_file,
                log_file=log_file,
                model=args.model,
                temperature=args.temperature,
                page2passage_mapping=page2passage_mapping,
                qid2wikipedia_cache=qid2wikipedia_cache
            )
            all_stats.append(stats)

        # Write summary
        log_file.write(f"\n\n{'='*80}\n")
        log_file.write("SUMMARY\n")
        log_file.write(f"{'='*80}\n")

        total_queries = len(queries)
        total_qids = sum(s['total_qids'] for s in all_stats)
        total_with_wiki = sum(s['qids_with_wikipedia'] for s in all_stats)
        total_passages = sum(s['total_passages_checked'] for s in all_stats)
        total_relevant = sum(s['relevant_passages'] for s in all_stats)

        log_file.write(f"Total queries processed: {total_queries}\n")
        log_file.write(f"Total QIDs: {total_qids}\n")
        log_file.write(f"QIDs with Wikipedia pages: {total_with_wiki}\n")
        log_file.write(f"Total passages checked: {total_passages}\n")
        log_file.write(f"Relevant passages found: {total_relevant}\n")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total queries processed: {total_queries}")
        print(f"Total QIDs: {total_qids}")
        print(f"QIDs with Wikipedia pages: {total_with_wiki}")
        print(f"Total passages checked: {total_passages}")
        print(f"Relevant passages found: {total_relevant}")
        print(f"\nQRels written to: {args.output_qrel}")
        print(f"Log written to: {args.log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TREC-format qrels for Total Recall RAG queries"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="quest",
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
        default="logs",
        help="Directory to write log file (log filename will be auto-generated based on dataset)"
    )

    parser.add_argument(
        "--page2passage_mapping",
        type=str,
        default=None,
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

    args = parser.parse_args()

    # Construct file paths based on dataset name
    dataset_name = args.dataset
    args.query_file = f"corpus_datasets/dataset_creation_heydar/{dataset_name}/test_{dataset_name}_queries.jsonl"
    args.output_qrel = f"corpus_datasets/dataset_creation_heydar/{dataset_name}/qrels_{dataset_name}.txt"

    # Create log file path based on dataset name
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    args.log_file = str(log_dir / f"qrel_generation_{dataset_name}.log")

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
    print("="*80)
    print()

    main(args)
    
    
# python c2_corpus_annotation/qrel_generation.py 
