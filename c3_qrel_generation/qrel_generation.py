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
from typing import List, Dict, Optional, Tuple, Union
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions from c1_1_dataset_creation_mahta
from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file, read_json_from_file
from c1_1_dataset_creation_mahta.query_generation.prompts.LLM_as_relevance_judge.prompt_property_check import (
    PROPERTY_PROMPT,
    PROPERTY_INPUT_TEMPLATE
)

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    Dataset = None


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


def load_corpus_in_memory(corpus_jsonl_path: str) -> Dict[str, Dict]:
    """
    Load entire corpus into memory as a dictionary for fast lookups.

    Args:
        corpus_jsonl_path: Path to corpus JSONL file

    Returns:
        Dictionary mapping passage_id to passage dict
    """
    print(f"Loading corpus into memory from {corpus_jsonl_path}...")
    corpus_dict = {}

    with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            passage = json.loads(line.strip())
            corpus_dict[passage['id']] = passage

    print(f"Loaded {len(corpus_dict)} passages into memory")
    return corpus_dict


def build_page_to_passages_index(corpus_dict: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Build an index from page ID to list of passage IDs.

    Args:
        corpus_dict: Dictionary mapping passage_id to passage dict

    Returns:
        Dictionary mapping page_id to list of passage_ids
    """
    print("Building page-to-passages index...")
    # Use defaultdict to avoid repeated 'if key not in dict' checks
    page_index = defaultdict(list)

    for passage_id in tqdm(corpus_dict.keys(), desc="Building page index"):
        # Passage IDs have format "pageid-0000", "pageid-0001", etc.
        # Extract page_id by finding the last '-' and taking everything before it
        dash_pos = passage_id.rfind('-')
        if dash_pos != -1:  # More efficient than 'if "-" in passage_id'
            page_id = passage_id[:dash_pos]
            page_index[page_id].append(passage_id)

    # Convert defaultdict back to regular dict for cleaner interface
    page_index = dict(page_index)

    print(f"Built index for {len(page_index)} pages")
    return page_index


def get_passages_for_page(
    pageid: str,
    corpus_jsonl_path: Optional[str] = None,
    page2passage_mapping: Optional[Dict] = None,
    corpus_dict: Optional[Dict[str, Dict]] = None,
    page_index: Optional[Dict[str, List[str]]] = None
) -> List[Dict]:
    """
    Get all passages for a given Wikipedia page ID from corpus.

    Args:
        pageid: Wikipedia page ID
        corpus_jsonl_path: Path to corpus JSONL file (for streaming mode)
        page2passage_mapping: Optional pre-built mapping from page ID to passage indices (for streaming)
        corpus_dict: Optional in-memory corpus dictionary (for in-memory mode)
        page_index: Optional page-to-passages index (for in-memory mode)

    Returns:
        List of passage dictionaries with 'id', 'title', 'contents'
    """
    passages = []

    # In-memory mode: use corpus_dict and page_index
    if corpus_dict is not None and page_index is not None:
        if pageid in page_index:
            passage_ids = page_index[pageid]
            for passage_id in passage_ids:
                passages.append(corpus_dict[passage_id])
        return passages

    # Streaming mode with pre-built mapping
    if page2passage_mapping and pageid in page2passage_mapping:
        passage_indices = page2passage_mapping[pageid]
        with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num in passage_indices:
                    passage = json.loads(line.strip())
                    passages.append(passage)
        return passages

    # Streaming mode: scan the corpus file
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
    corpus_jsonl_path: Optional[str],
    client: OpenAI,
    qrel_output_file,
    log_file,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    page2passage_mapping: Optional[Dict] = None,
    qid2wikipedia_cache: Optional[Dict] = None,
    corpus_dict: Optional[Dict[str, Dict]] = None,
    page_index: Optional[Dict[str, List[str]]] = None
) -> Dict:
    """
    Process a single query to generate qrels.

    Args:
        query_obj: Query object from JSONL
        corpus_jsonl_path: Path to corpus JSONL file (for streaming mode)
        client: OpenAI client
        qrel_output_file: Open file handle for qrel output
        log_file: Open file handle for logging
        model: LLM model to use
        temperature: Temperature for LLM
        page2passage_mapping: Optional mapping for efficiency (streaming mode)
        qid2wikipedia_cache: Optional cache for QID->Wikipedia mappings
        corpus_dict: Optional in-memory corpus (in-memory mode)
        page_index: Optional page index (in-memory mode)

    Returns:
        Statistics dict
    """
    query_id = query_obj['qid']

    # Extract entity QIDs and their values from the new nested structure
    property_info = query_obj['property']
    entities_values = property_info.get('entities_values', [])

    # Build list of QIDs and their labels/values
    total_recall_qids = [entity['entity_id'] for entity in entities_values]

    # Extract property metadata from nested property_info structure
    property_metadata = property_info['property_info']
    property_label = property_metadata['label']
    property_description = property_metadata['description']
    property_id = property_metadata['property_id']

    stats = {
        'total_qids': len(total_recall_qids),
        'qids_with_wikipedia': 0,
        'total_passages_checked': 0,
        'relevant_passages': 0,
        'entities_with_relevant_passages': 0  # Track entity coverage
    }

    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Query ID: {query_id}\n")
    log_file.write(f"Query: {query_obj.get('question', query_obj.get('original_query', 'N/A'))}\n")
    log_file.write(f"Property: {property_label} ({property_id}) - {property_description}\n")
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
        passages = get_passages_for_page(
            pageid=pageid,
            corpus_jsonl_path=corpus_jsonl_path,
            page2passage_mapping=page2passage_mapping,
            corpus_dict=corpus_dict,
            page_index=page_index
        )

        if not passages:
            log_file.write(f"No passages found for page {pageid}\n")
            continue

        log_file.write(f"Found {len(passages)} passages\n")
        stats['total_passages_checked'] += len(passages)

        # Track if this entity has at least one relevant passage
        entity_has_relevant_passage = False

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

            # Only write to qrel file if judgment is YES
            if judgment == "YES":
                write_trec_qrel(qrel_output_file, query_id, passage_id, 1)
                qrel_output_file.flush()  # Flush immediately to disk
                stats['relevant_passages'] += 1
                entity_has_relevant_passage = True

        # Update entity coverage count
        if entity_has_relevant_passage:
            stats['entities_with_relevant_passages'] += 1

    return stats


def main(args):
    """Main function to generate qrels."""

    # Load queries
    print(f"Loading queries from {args.query_file}...")
    queries = read_jsonl_from_file(args.query_file)
    print(f"Loaded {len(queries)} queries")

    # Initialize corpus loading based on mode
    corpus_dict = None
    page_index = None
    page2passage_mapping = None

    if args.load_corpus_mode == "memory":
        # Load corpus into memory
        corpus_dict = load_corpus_in_memory(args.corpus_jsonl)

        # In memory mode, we need to build the page index from corpus_dict
        # The page index maps page_id -> list of passage_ids (not line numbers)
        # This is different from page2passage_mapping which maps page_id -> line numbers
        page_index = build_page_to_passages_index(corpus_dict)

        print(f"Corpus loading mode: IN-MEMORY")
        print(f"Memory usage: ~{len(corpus_dict)} passages loaded")
    else:
        # Streaming mode: optionally load page2passage mapping
        if args.page2passage_mapping:
            print(f"Loading page2passage mapping from {args.page2passage_mapping}...")
            page2passage_mapping = read_json_from_file(args.page2passage_mapping)
            print(f"Loaded mapping for {len(page2passage_mapping)} pages")
        print(f"Corpus loading mode: STREAMING")

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
        log_file.write(f"Corpus loading mode: {args.load_corpus_mode}\n")
        log_file.write(f"Model: {args.model}\n")
        log_file.write(f"Temperature: {args.temperature}\n")
        log_file.write(f"\n{'='*80}\n")

        # Process queries
        all_stats = []
        for idx, query_obj in tqdm(enumerate(queries), desc="Processing queries"):

            if idx == 10:
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
                qid2wikipedia_cache=qid2wikipedia_cache,
                corpus_dict=corpus_dict,
                page_index=page_index
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
        total_entities_covered = sum(s['entities_with_relevant_passages'] for s in all_stats)

        # Calculate entity coverage percentage
        entity_coverage_pct = (total_entities_covered / total_with_wiki * 100) if total_with_wiki > 0 else 0

        log_file.write(f"Total queries processed: {total_queries}\n")
        log_file.write(f"Total QIDs: {total_qids}\n")
        log_file.write(f"QIDs with Wikipedia pages: {total_with_wiki}\n")
        log_file.write(f"Total passages checked: {total_passages}\n")
        log_file.write(f"Relevant passages found: {total_relevant}\n")
        log_file.write(f"Entities with at least one relevant passage: {total_entities_covered}\n")
        log_file.write(f"Entity coverage: {entity_coverage_pct:.2f}%\n")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total queries processed: {total_queries}")
        print(f"Total QIDs: {total_qids}")
        print(f"QIDs with Wikipedia pages: {total_with_wiki}")
        print(f"Total passages checked: {total_passages}")
        print(f"Relevant passages found: {total_relevant}")
        print(f"Entities with at least one relevant passage: {total_entities_covered}")
        print(f"Entity coverage: {entity_coverage_pct:.2f}%")
        print(f"\nQRels written to: {args.output_qrel}")
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
        "--load_corpus_mode",
        type=str,
        default="stream",
        choices=["stream", "memory"],
        help="How to load corpus: 'stream' (low memory, slower per-query) or 'memory' (high memory, faster per-query). "
             "Default: 'stream'. Use 'memory' when processing many queries (100s+) and have sufficient RAM."
    )

    args = parser.parse_args()

    # Construct file paths based on dataset name
    dataset_name = args.dataset
    # Map dataset name to directory name (quest -> test_quest)
    dir_name = "test_quest" if dataset_name == "quest" else dataset_name
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
    print(f"Corpus loading mode: {args.load_corpus_mode.upper()}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print("="*80)
    print()

    main(args)


# ============================================================================
# Usage Examples
# ============================================================================
#
# 1. STREAMING MODE (Default - Low memory, good for small query batches):
#    python c2_corpus_annotation/qrel_generation.py --dataset qald10
#
#    OR explicitly:
#    python c2_corpus_annotation/qrel_generation.py --dataset qald10 --load_corpus_mode stream
#
# 2. IN-MEMORY MODE (High memory, faster per-query, good for large batches):
#    python c2_corpus_annotation/qrel_generation.py --dataset qald10 --load_corpus_mode memory
#
# When to use each mode:
# - STREAM: Processing <100 queries, limited RAM, or very large corpus
# - MEMORY: Processing 100s-1000s of queries, have >32GB RAM, corpus fits in memory
#
# ============================================================================ 
