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
import pickle
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, TextIO
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions from c1_1_dataset_creation_mahta
from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file, read_json_from_file
from c1_1_dataset_creation_mahta.query_generation.prompts.LLM_as_relevance_judge.prompt_property_check import (
    PROPERTY_PROMPT,
    PROPERTY_INPUT_TEMPLATE
)
from c1_1_dataset_creation_mahta.query_generation.prompts.LLM_as_relevance_judge.prompt_value_check import (
    VALUE_PROMPT,
    VALUE_INPUT_TEMPLATE
)
# Import analysis functions from qrel_analysis
from c3_qrel_generation.qrel_analysis import calculate_coverage

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    Dataset = None


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


def get_index_cache_path(corpus_jsonl_path: str, index_cache_dir: Optional[str] = None) -> Path:
    """
    Get the cache file path for a corpus index.

    Args:
        corpus_jsonl_path: Path to corpus JSONL file
        index_cache_dir: Optional directory for cache files. If None, uses same dir as corpus

    Returns:
        Path object for the cache file
    """
    corpus_path = Path(corpus_jsonl_path)
    cache_filename = f"{corpus_path.stem}_memory_index.pkl"

    if index_cache_dir:
        cache_dir = Path(index_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / cache_filename
    else:
        return corpus_path.parent / cache_filename


def save_corpus_index(corpus_dict: Dict[str, Dict], page_index: Dict[str, List[str]], cache_path: Path):
    """
    Save corpus_dict and page_index to disk for faster loading next time.

    Args:
        corpus_dict: Dictionary mapping passage_id to passage dict
        page_index: Dictionary mapping page_id to list of passage_ids
        cache_path: Path where to save the index
    """
    print(f"Saving corpus index to {cache_path}...")
    index_data = {
        'corpus_dict': corpus_dict,
        'page_index': page_index
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Get file size for reporting
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Saved index to {cache_path} ({size_mb:.1f} MB)")


def load_corpus_index(cache_path: Path) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
    """
    Load corpus_dict and page_index from disk cache.

    Args:
        cache_path: Path to cached index file

    Returns:
        Tuple of (corpus_dict, page_index)
    """
    print(f"Loading corpus index from cache: {cache_path}...")

    with open(cache_path, 'rb') as f:
        index_data = pickle.load(f)

    corpus_dict = index_data['corpus_dict']
    page_index = index_data['page_index']

    print(f"Loaded {len(corpus_dict)} passages and {len(page_index)} pages from cache")
    return corpus_dict, page_index


def load_corpus_in_memory(corpus_jsonl_path: str, index_cache_dir: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load entire corpus into memory as a dictionary for fast lookups.
    First checks for cached index, otherwise builds from scratch.

    Args:
        corpus_jsonl_path: Path to corpus JSONL file
        index_cache_dir: Optional directory for cache files

    Returns:
        Dictionary mapping passage_id to passage dict
    """
    # Check if cached index exists
    cache_path = get_index_cache_path(corpus_jsonl_path, index_cache_dir)

    if cache_path.exists():
        print(f"Found cached index at {cache_path}")
        try:
            corpus_dict, _ = load_corpus_index(cache_path)
            return corpus_dict
        except Exception as e:
            print(f"Warning: Failed to load cached index ({e}), rebuilding from scratch...")

    print(f"Loading corpus into memory from {corpus_jsonl_path}...")
    corpus_dict = {}

    with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            passage = json.loads(line.strip())
            corpus_dict[passage['id']] = passage

    print(f"Loaded {len(corpus_dict)} passages into memory")
    return corpus_dict


def build_page_to_passages_index(
    corpus_dict: Dict[str, Dict],
    corpus_jsonl_path: str,
    index_cache_dir: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Build an index from page ID to list of passage IDs.
    First checks for cached index, otherwise builds from scratch.

    Args:
        corpus_dict: Dictionary mapping passage_id to passage dict
        corpus_jsonl_path: Path to corpus JSONL (for determining cache path)
        index_cache_dir: Optional directory for cache files

    Returns:
        Dictionary mapping page_id to list of passage_ids
    """
    # Check if cached index exists
    cache_path = get_index_cache_path(corpus_jsonl_path, index_cache_dir)

    if cache_path.exists():
        print(f"Found cached index at {cache_path}")
        try:
            _, page_index = load_corpus_index(cache_path)
            return page_index
        except Exception as e:
            print(f"Warning: Failed to load cached index ({e}), rebuilding from scratch...")

    print("Building page-to-passages index...")
    # Use defaultdict to avoid repeated 'if key not in dict' checks
    page_index = defaultdict(list)

    for passage_id in tqdm(corpus_dict.keys(), desc="Building page index"):
        page_id = extract_page_id_from_passage_id(passage_id)
        if page_id:
            page_index[page_id].append(passage_id)

    # Convert defaultdict back to regular dict for cleaner interface
    page_index = dict(page_index)

    print(f"Built index for {len(page_index)} pages")
    return page_index


def build_page2passage_mapping_from_corpus(corpus_jsonl_path: str) -> Dict[str, List[int]]:
    """
    Build a mapping from page ID to line numbers in the corpus file.
    This is used for efficient passage lookup in streaming mode.

    Args:
        corpus_jsonl_path: Path to corpus JSONL file

    Returns:
        Dictionary mapping page_id to list of line numbers (0-indexed)
    """
    print(f"Scanning corpus to build page-to-passage mapping: {corpus_jsonl_path}")
    page_mapping = defaultdict(list)

    with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Building mapping")):
            passage = json.loads(line.strip())
            passage_id = passage.get('id', '')
            page_id = extract_page_id_from_passage_id(passage_id)
            if page_id:
                page_mapping[page_id].append(line_num)

    # Convert defaultdict to regular dict
    return dict(page_mapping)


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
        passage_indices = set(page2passage_mapping[pageid])  # Convert to set for O(1) lookup
        with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num in passage_indices:
                    passage = json.loads(line.strip())
                    passages.append(passage)
                    # Early exit if we've found all passages
                    if len(passages) == len(passage_indices):
                        break
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
            max_tokens=10,  # Only need "YES" or "NO"
        )

        response = resp.choices[0].message.content.strip()
        if response not in ["YES", "NO"]:
            print(f"Warning: Unexpected response '{response}', defaulting to NO", file=sys.stderr)
            return "NO"
        return response

    except Exception as e:
        print(f"Error in LLM judgment: {e}", file=sys.stderr)
        return "NO"


async def judge_passage_relevance_async(
    passage_content: str,
    entity_label: str,
    property_label: str,
    property_description: str,
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    semaphore: Optional[asyncio.Semaphore] = None
) -> str:
    """
    Async version: Use LLM to judge if passage contains information about the property.

    Args:
        passage_content: The text content of the passage
        entity_label: Name of the entity (e.g., film title)
        property_label: Property name (e.g., "publication date")
        property_description: Description of the property
        client: AsyncOpenAI client
        model: Model to use for judgment
        temperature: Temperature for generation
        semaphore: Optional semaphore for rate limiting

    Returns:
        "YES" or "NO"
    """
    input_instance = PROPERTY_INPUT_TEMPLATE.format(
        entity_name=entity_label,
        property_name=property_label,
        property_description=property_description,
        passage=passage_content
    )

    async def _make_request():
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PROPERTY_PROMPT},
                    {"role": "user", "content": input_instance}
                ],
                temperature=temperature,
                max_tokens=10,  # Only need "YES" or "NO"
            )

            response = resp.choices[0].message.content.strip()
            if response not in ["YES", "NO"]:
                print(f"Warning: Unexpected response '{response}', defaulting to NO", file=sys.stderr)
                return "NO"
            return response

        except Exception as e:
            print(f"Error in LLM judgment: {e}", file=sys.stderr)
            return "NO"

    if semaphore:
        async with semaphore:
            return await _make_request()
    else:
        return await _make_request()


async def judge_passages_batch(
    passages: List[Dict],
    entity_label: str,
    property_label: str,
    property_description: str,
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_concurrent: int = 10
) -> List[Tuple[str, str]]:
    """
    Judge multiple passages in parallel with rate limiting.

    Args:
        passages: List of passage dictionaries with 'id' and 'contents'
        entity_label: Name of the entity
        property_label: Property name
        property_description: Description of the property
        client: AsyncOpenAI client
        model: Model to use for judgment
        temperature: Temperature for generation
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        List of tuples (passage_id, judgment) where judgment is "YES" or "NO"
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all passages
    tasks = []
    passage_ids = []
    for passage in passages:
        task = judge_passage_relevance_async(
            passage_content=passage['contents'],
            entity_label=entity_label,
            property_label=property_label,
            property_description=property_description,
            client=client,
            model=model,
            temperature=temperature,
            semaphore=semaphore
        )
        tasks.append(task)
        passage_ids.append(passage['id'])

    # Execute ALL tasks concurrently using asyncio.gather
    judgments = await asyncio.gather(*tasks)

    # Combine passage IDs with their judgments
    results = list(zip(passage_ids, judgments))

    return results


def write_trec_qrel(
    output_file: TextIO,
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
    qrel_output_file: TextIO,
    log_file: TextIO,
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

    # Handle different dataset structures:
    # QALD10: has nested structure with query_obj['property']['entities_values'] and query_obj['property']['property_info']
    # Quest: has flat structure with query_obj['entities_values'] and query_obj['property_info']
    if 'property' in query_obj and isinstance(query_obj['property'], dict):
        # QALD10 structure
        property_info = query_obj['property']
        entities_values = property_info.get('entities_values', [])
        property_metadata = property_info['property_info']
    else:
        # Quest structure
        entities_values = query_obj.get('entities_values', [])
        property_metadata = query_obj.get('property_info', {})

    # Build list of QIDs and their labels/values
    total_recall_qids = [entity['entity_id'] for entity in entities_values]

    # Extract property metadata (works for both structures)
    property_label = property_metadata.get('label', property_metadata.get('label', 'unknown'))
    property_description = property_metadata.get('description', '')
    property_id = property_metadata.get('property_id', property_metadata.get('id', 'unknown'))

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
        if qid2wikipedia_cache and qid in qid2wikipedia_cache:
            wiki_info = qid2wikipedia_cache[qid]
        else:
            wiki_info = get_wikipedia_info_from_qid(qid)
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


async def process_query_async(
    query_obj: Dict,
    corpus_jsonl_path: Optional[str],
    client: AsyncOpenAI,
    qrel_output_file: TextIO,
    log_file: TextIO,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    page2passage_mapping: Optional[Dict] = None,
    qid2wikipedia_cache: Optional[Dict] = None,
    corpus_dict: Optional[Dict[str, Dict]] = None,
    page_index: Optional[Dict[str, List[str]]] = None,
    max_concurrent: int = 10
) -> Dict:
    """
    Async version: Process a single query to generate qrels with parallel LLM calls.

    Args:
        query_obj: Query object from JSONL
        corpus_jsonl_path: Path to corpus JSONL file (for streaming mode)
        client: AsyncOpenAI client
        qrel_output_file: Open file handle for qrel output
        log_file: Open file handle for logging
        model: LLM model to use
        temperature: Temperature for LLM
        page2passage_mapping: Optional mapping for efficiency (streaming mode)
        qid2wikipedia_cache: Optional cache for QID->Wikipedia mappings
        corpus_dict: Optional in-memory corpus (in-memory mode)
        page_index: Optional page index (in-memory mode)
        max_concurrent: Maximum concurrent LLM API calls

    Returns:
        Statistics dict
    """
    query_id = query_obj['qid']

    # Handle different dataset structures:
    # QALD10: has nested structure with query_obj['property']['entities_values'] and query_obj['property']['property_info']
    # Quest: has flat structure with query_obj['entities_values'] and query_obj['property_info']
    if 'property' in query_obj and isinstance(query_obj['property'], dict):
        # QALD10 structure
        property_info = query_obj['property']
        entities_values = property_info.get('entities_values', [])
        property_metadata = property_info['property_info']
    else:
        # Quest structure
        entities_values = query_obj.get('entities_values', [])
        property_metadata = query_obj.get('property_info', {})

    # Build list of QIDs and their labels/values
    total_recall_qids = [entity['entity_id'] for entity in entities_values]

    # Extract property metadata (works for both structures)
    property_label = property_metadata.get('label', property_metadata.get('label', 'unknown'))
    property_description = property_metadata.get('description', '')
    property_id = property_metadata.get('property_id', property_metadata.get('id', 'unknown'))

    stats = {
        'total_qids': len(total_recall_qids),
        'qids_with_wikipedia': 0,
        'total_passages_checked': 0,
        'relevant_passages': 0,
        'entities_with_relevant_passages': 0,
        'time_llm_calls': 0.0,
        'time_passage_retrieval': 0.0
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
        if qid2wikipedia_cache and qid in qid2wikipedia_cache:
            wiki_info = qid2wikipedia_cache[qid]
        else:
            wiki_info = get_wikipedia_info_from_qid(qid)
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
        t_start = time.time()
        passages = get_passages_for_page(
            pageid=pageid,
            corpus_jsonl_path=corpus_jsonl_path,
            page2passage_mapping=page2passage_mapping,
            corpus_dict=corpus_dict,
            page_index=page_index
        )
        stats['time_passage_retrieval'] += time.time() - t_start

        if not passages:
            log_file.write(f"No passages found for page {pageid}\n")
            continue

        log_file.write(f"Found {len(passages)} passages\n")
        stats['total_passages_checked'] += len(passages)

        # Track if this entity has at least one relevant passage
        entity_has_relevant_passage = False

        # Judge all passages in parallel
        log_file.write(f"Judging {len(passages)} passages in parallel (max_concurrent={max_concurrent})...\n")
        t_start = time.time()

        judgments = await judge_passages_batch(
            passages=passages,
            entity_label=entity_title,
            property_label=property_label,
            property_description=property_description,
            client=client,
            model=model,
            temperature=temperature,
            max_concurrent=max_concurrent
        )

        stats['time_llm_calls'] += time.time() - t_start

        # Process results
        for passage_id, judgment in judgments:
            # Find the passage content for logging
            passage_content = next((p['contents'] for p in passages if p['id'] == passage_id), "")

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


def print_summary_stats(
    all_stats: List[Dict],
    total_queries: int,
    output_file: Optional[TextIO] = None
):
    """
    Print summary statistics to console and optionally to a file.

    Args:
        all_stats: List of statistics dictionaries from processing queries
        total_queries: Total number of queries processed
        output_file: Optional file handle to also write stats to
    """
    total_qids = sum(s['total_qids'] for s in all_stats)
    total_with_wiki = sum(s['qids_with_wikipedia'] for s in all_stats)
    total_passages = sum(s['total_passages_checked'] for s in all_stats)
    total_relevant = sum(s['relevant_passages'] for s in all_stats)
    total_entities_covered = sum(s['entities_with_relevant_passages'] for s in all_stats)

    # Calculate entity coverage percentage
    entity_coverage_pct = (total_entities_covered / total_with_wiki * 100) if total_with_wiki > 0 else 0

    stats_lines = [
        f"Total queries processed: {total_queries}",
        f"Total QIDs: {total_qids}",
        f"QIDs with Wikipedia pages: {total_with_wiki}",
        f"Total passages checked: {total_passages}",
        f"Relevant passages found: {total_relevant}",
        f"Entities with at least one relevant passage: {total_entities_covered}",
        f"Entity coverage: {entity_coverage_pct:.2f}%"
    ]

    # Add timing stats if available
    if all_stats and 'time_llm_calls' in all_stats[0]:
        total_llm_time = sum(s.get('time_llm_calls', 0) for s in all_stats)
        total_retrieval_time = sum(s.get('time_passage_retrieval', 0) for s in all_stats)
        stats_lines.extend([
            "",
            "Performance Metrics:",
            f"  Total LLM call time: {total_llm_time:.2f}s ({total_llm_time/60:.2f}m)",
            f"  Total passage retrieval time: {total_retrieval_time:.2f}s ({total_retrieval_time/60:.2f}m)",
            f"  Avg time per passage (LLM): {total_llm_time/total_passages:.3f}s" if total_passages > 0 else "  Avg time per passage: N/A"
        ])

    # Write to file if provided
    if output_file:
        for line in stats_lines:
            output_file.write(line + "\n")

    # Always print to console
    for line in stats_lines:
        print(line)


def get_all_query_ids_in_qrel(qrel_file_path):
    """
    Extract all query IDs from qrel file in order of appearance.

    Args:
        qrel_file_path: Path to qrel file

    Returns:
        List of query IDs in order of appearance (with duplicates removed, preserving first occurrence)
    """
    query_ids_in_order = []
    seen = set()

    if Path(qrel_file_path).exists():
        with open(qrel_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    qid = parts[0]
                    if qid not in seen:
                        seen.add(qid)
                        query_ids_in_order.append(qid)

    return query_ids_in_order


def count_passages_for_query(qrel_file_path, query_id):
    """
    Count number of passages (qrel entries) for a specific query.

    Args:
        qrel_file_path: Path to qrel file
        query_id: Query ID to count passages for

    Returns:
        Number of passages for the query
    """
    count = 0
    with open(qrel_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == query_id:
                count += 1
    return count


def remove_query_from_qrel(qrel_file_path, query_id_to_remove):
    """
    Remove all entries for a specific query from qrel file.

    Args:
        qrel_file_path: Path to qrel file
        query_id_to_remove: Query ID whose entries should be removed

    Returns:
        Number of entries removed
    """
    temp_file = Path(qrel_file_path).with_suffix('.tmp')

    removed_count = 0
    with open(qrel_file_path, 'r', encoding='utf-8') as infile, \
         open(temp_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == query_id_to_remove:
                removed_count += 1
                continue  # Skip this line
            outfile.write(line)

    # Replace original with cleaned version
    temp_file.replace(qrel_file_path)
    return removed_count


def judge_passage_value(
    passage_content: str,
    entity_label: str,
    property_label: str,
    property_description: str,
    property_value: str,
    property_value_unit: Optional[str],
    property_value_time: Optional[str],
    client: OpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0
) -> str:
    """
    Use LLM to judge if passage contains the correct value for the property.

    Args:
        passage_content: The text content of the passage
        entity_label: Name of the entity (e.g., film title)
        property_label: Property name (e.g., "publication date")
        property_description: Description of the property
        property_value: The expected value of the property
        property_value_unit: Optional unit of the property value
        property_value_time: Optional time of validity for the property value
        client: OpenAI client
        model: Model to use for judgment
        temperature: Temperature for generation

    Returns:
        "YES" or "NO"
    """
    # Build the statement with value, unit, and time
    statement_parts = [f"the [{property_label}] of [{entity_label}] is {property_value}"]
    
    if property_value_unit:
        statement_parts[0] += f" in terms of {property_value_unit}"
    
    statement = statement_parts[0]
    
    if property_value_time:
        statement += f"\nthis statement is valid as of {property_value_time}"
    
    input_instance = VALUE_INPUT_TEMPLATE.format(
        entity_name=entity_label,
        property_name=property_label,
        property_description=property_description,
        property_value=property_value,
        passage=passage_content
    )
    
    # Replace the STATEMENT placeholder with our custom statement
    input_instance = input_instance.replace(
        f"the [{property_label}] of [{entity_label}] is {property_value}",
        statement
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VALUE_PROMPT},
                {"role": "user", "content": input_instance}
            ],
            temperature=temperature,
            max_tokens=10,  # Only need "YES" or "NO"
        )

        response = resp.choices[0].message.content.strip()
        if response not in ["YES", "NO"]:
            print(f"Warning: Unexpected response '{response}', defaulting to NO", file=sys.stderr)
            return "NO"
        return response

    except Exception as e:
        print(f"Error in LLM value judgment: {e}", file=sys.stderr)
        return "NO"


async def judge_passage_value_async(
    passage_content: str,
    entity_label: str,
    property_label: str,
    property_description: str,
    property_value: str,
    property_value_unit: Optional[str],
    property_value_time: Optional[str],
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    semaphore: Optional[asyncio.Semaphore] = None
) -> str:
    """
    Async version: Use LLM to judge if passage contains the correct value for the property.

    Args:
        passage_content: The text content of the passage
        entity_label: Name of the entity
        property_label: Property name
        property_description: Description of the property
        property_value: The expected value of the property
        property_value_unit: Optional unit of the property value
        property_value_time: Optional time of validity for the property value
        client: AsyncOpenAI client
        model: Model to use for judgment
        temperature: Temperature for generation
        semaphore: Optional semaphore for rate limiting

    Returns:
        "YES" or "NO"
    """
    # Build the statement with value, unit, and time
    statement_parts = [f"the [{property_label}] of [{entity_label}] is {property_value}"]
    
    if property_value_unit:
        statement_parts[0] += f" in terms of {property_value_unit}"
    
    statement = statement_parts[0]
    
    if property_value_time:
        statement += f"\nthis statement is valid as of {property_value_time}"
    
    input_instance = VALUE_INPUT_TEMPLATE.format(
        entity_name=entity_label,
        property_name=property_label,
        property_description=property_description,
        property_value=property_value,
        passage=passage_content
    )
    
    # Replace the STATEMENT placeholder with our custom statement
    input_instance = input_instance.replace(
        f"the [{property_label}] of [{entity_label}] is {property_value}",
        statement
    )

    async def _make_request():
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": VALUE_PROMPT},
                    {"role": "user", "content": input_instance}
                ],
                temperature=temperature,
                max_tokens=10,  # Only need "YES" or "NO"
            )

            response = resp.choices[0].message.content.strip()
            if response not in ["YES", "NO"]:
                print(f"Warning: Unexpected response '{response}', defaulting to NO", file=sys.stderr)
                return "NO"
            return response

        except Exception as e:
            print(f"Error in LLM value judgment: {e}", file=sys.stderr)
            return "NO"

    if semaphore:
        async with semaphore:
            return await _make_request()
    else:
        return await _make_request()


async def judge_values_batch(
    passages: List[Dict],
    entity_label: str,
    property_label: str,
    property_description: str,
    property_value: str,
    property_value_unit: Optional[str],
    property_value_time: Optional[str],
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_concurrent: int = 10
) -> List[Tuple[str, str]]:
    """
    Judge multiple passages for value matching in parallel with rate limiting.

    Args:
        passages: List of passage dictionaries with 'id' and 'contents'
        entity_label: Name of the entity
        property_label: Property name
        property_description: Description of the property
        property_value: The expected value of the property
        property_value_unit: Optional unit of the property value
        property_value_time: Optional time of validity for the property value
        client: AsyncOpenAI client
        model: Model to use for judgment
        temperature: Temperature for generation
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        List of tuples (passage_id, judgment) where judgment is "YES" or "NO"
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all passages
    tasks = []
    passage_ids = []
    for passage in passages:
        task = judge_passage_value_async(
            passage_content=passage['contents'],
            entity_label=entity_label,
            property_label=property_label,
            property_description=property_description,
            property_value=property_value,
            property_value_unit=property_value_unit,
            property_value_time=property_value_time,
            client=client,
            model=model,
            temperature=temperature,
            semaphore=semaphore
        )
        tasks.append(task)
        passage_ids.append(passage['id'])

    # Execute ALL tasks concurrently using asyncio.gather
    judgments = await asyncio.gather(*tasks)

    # Combine passage IDs with their judgments
    results = list(zip(passage_ids, judgments))

    return results


def prepare_resume(args, queries):
    """
    Prepare for resume mode: determine which queries to process.

    Args:
        args: Command line arguments
        queries: List of all query objects

    Returns:
        Tuple of (queries_to_process, needs_file_append)
        - queries_to_process: List of query objects to process
        - needs_file_append: Boolean, True if we should append to existing file
    """
    if not args.resume:
        return queries, False

    if not Path(args.output_qrel).exists():
        print("Resume mode enabled but no existing qrel file found. Starting fresh.")
        return queries, False

    # Get all query IDs that appear in the qrel file
    all_in_file = get_all_query_ids_in_qrel(args.output_qrel)

    if not all_in_file:
        print("Resume mode enabled but qrel file is empty. Starting fresh.")
        return queries, False

    # Determine which queries are completed
    completed_query_ids = set()
    query_to_reprocess = None
    last_query_id = all_in_file[-1]

    # Check if we should re-process the last query
    if args.reprocess_last:
        # Always re-process last query
        print(f"⚠️  --reprocess-last flag enabled")
        print(f"   Last query in file: '{last_query_id}'")
        query_to_reprocess = last_query_id
        completed_query_ids = set(all_in_file[:-1])
    elif args.trust_last:
        # Trust that last query is complete
        print(f"✅ --trust-last flag enabled")
        print(f"   Trusting all queries in file (including last: '{last_query_id}')")
        completed_query_ids = set(all_in_file)
    else:
        # Use heuristic: check passage count for last query
        last_query_passage_count = count_passages_for_query(args.output_qrel, last_query_id)

        if last_query_passage_count < args.min_passages_threshold:
            print(f"⚠️  Last query '{last_query_id}' has only {last_query_passage_count} passages")
            print(f"   (threshold: {args.min_passages_threshold})")
            print(f"   Likely incomplete, will re-process this query")
            query_to_reprocess = last_query_id
            completed_query_ids = set(all_in_file[:-1])
        else:
            print(f"✅ Last query '{last_query_id}' has {last_query_passage_count} passages")
            print(f"   (threshold: {args.min_passages_threshold})")
            print(f"   Appears complete, skipping")
            completed_query_ids = set(all_in_file)

    # Clean up qrel file if re-processing a query
    if query_to_reprocess:
        print(f"   Removing existing entries for '{query_to_reprocess}'...")
        removed_count = remove_query_from_qrel(args.output_qrel, query_to_reprocess)
        print(f"   Removed {removed_count} entries")

    # Filter queries to process
    original_count = len(queries)
    queries_to_process = [q for q in queries if q['qid'] not in completed_query_ids]

    # Print summary
    print(f"\n{'='*80}")
    print(f"RESUME MODE SUMMARY")
    print(f"{'='*80}")
    print(f"Total queries in dataset: {original_count}")
    print(f"Already completed (skipping): {len(completed_query_ids)}")
    print(f"Remaining to process: {len(queries_to_process)}")
    if query_to_reprocess:
        print(f"Re-processing (was incomplete): {query_to_reprocess}")
    print(f"{'='*80}\n")

    # We need to append to existing file since we're resuming
    return queries_to_process, True


def check_value(args):
    """
    Perform value check on passages that passed property check.
    Reads the property check qrel file and validates that passages contain the correct values.
    """
    print("\n" + "="*80)
    print("STARTING VALUE CHECK")
    print("="*80)
    
    # Construct input qrel path from property check
    property_qrel_path = args.output_qrel
    
    # Construct output paths for value check
    if "/qrels_" in property_qrel_path:
        value_qrel_path = property_qrel_path.replace("/qrels_", "/qrels_value_")
    else:
        value_qrel_path = property_qrel_path.replace("qrels_", "qrels_value_")
    
    # Construct log file path for value check
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract subset name from query file for log naming
    query_file_path = Path(args.query_file)
    subset_name = query_file_path.stem.replace("_generations", "")
    value_log_file_path = str(log_dir / f"value_check_{subset_name}_{timestamp}.log")
    
    print(f"Input property qrel: {property_qrel_path}")
    print(f"Output value qrel: {value_qrel_path}")
    print(f"Value check log: {value_log_file_path}")
    
    # Load queries to get entity values and property info
    print(f"\nLoading queries from {args.query_file}...")
    all_queries = read_jsonl_from_file(args.query_file)
    print(f"Loaded {len(all_queries)} queries")
    
    # Create a dict mapping query_id to query object for quick lookup
    query_dict = {q['qid']: q for q in all_queries}
    
    # Load property qrel file to get relevant passages
    print(f"\nLoading property qrel from {property_qrel_path}...")
    property_qrels = {}  # query_id -> list of passage_ids
    
    if not Path(property_qrel_path).exists():
        print(f"ERROR: Property qrel file not found: {property_qrel_path}")
        return
    
    with open(property_qrel_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id, _, passage_id, relevance = parts[0], parts[1], parts[2], parts[3]
                if int(relevance) == 1:  # Only keep relevant passages
                    if query_id not in property_qrels:
                        property_qrels[query_id] = []
                    property_qrels[query_id].append(passage_id)
    
    print(f"Loaded qrels for {len(property_qrels)} queries")
    
    # Apply limit if specified (to match property check behavior)
    if args.limit is not None and args.limit > 0:
        # Get first N query IDs and filter property_qrels
        limited_query_ids = list(property_qrels.keys())[:args.limit]
        property_qrels = {qid: property_qrels[qid] for qid in limited_query_ids}
        print(f"Limited to first {len(property_qrels)} queries (--limit {args.limit})")
    
    total_passages_to_check = sum(len(pids) for pids in property_qrels.values())
    print(f"Total passages to check for value: {total_passages_to_check}")
    
    # Initialize corpus loading (same as property_check_main)
    corpus_dict = None
    page_index = None
    
    if args.load_corpus_mode == "memory":
        cache_path = get_index_cache_path(args.corpus_jsonl, args.index_cache_dir)
        index_exists = cache_path.exists()
        
        corpus_dict = load_corpus_in_memory(args.corpus_jsonl, args.index_cache_dir)
        page_index = build_page_to_passages_index(corpus_dict, args.corpus_jsonl, args.index_cache_dir)
        
        if not index_exists:
            save_corpus_index(corpus_dict, page_index, cache_path)
        
        print(f"Corpus loading mode: IN-MEMORY")
        print(f"Memory usage: ~{len(corpus_dict)} passages loaded")
    else:
        print(f"Corpus loading mode: STREAMING (loading from {args.corpus_jsonl})")
        # Build corpus_dict by loading the entire corpus
        corpus_dict = load_corpus_in_memory(args.corpus_jsonl, args.index_cache_dir)
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    if args.use_parallel:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        print(f"Using PARALLEL processing mode with max_concurrent={args.max_concurrent}")
    else:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        print(f"Using SEQUENTIAL processing mode")
    
    # Open output files
    with open(value_qrel_path, 'w', encoding='utf-8') as qrel_file, \
         open(value_log_file_path, 'w', encoding='utf-8') as log_file:
        
        # Write header
        log_file.write(f"Value Check Log\n")
        log_file.write(f"Query file: {args.query_file}\n")
        log_file.write(f"Property qrel: {property_qrel_path}\n")
        log_file.write(f"Corpus: {args.corpus_jsonl}\n")
        log_file.write(f"Model: {args.model}\n")
        log_file.write(f"Temperature: {args.temperature}\n")
        log_file.write(f"\n{'='*80}\n")
        log_file.flush()  # Flush header immediately
        
        # Process each query
        all_stats = []
        
        if args.use_parallel:
            # Async processing
            async def process_all_value_checks():
                stats_list = []
                for query_id in tqdm(property_qrels.keys(), desc="Processing value checks"):
                    if query_id not in query_dict:
                        print(f"Warning: Query {query_id} not found in query file")
                        continue
                    
                    query_obj = query_dict[query_id]
                    passage_ids = property_qrels[query_id]
                    
                    stats = await process_value_check_async(
                        query_obj=query_obj,
                        passage_ids=passage_ids,
                        corpus_dict=corpus_dict,
                        client=client,
                        qrel_output_file=qrel_file,
                        log_file=log_file,
                        model=args.model,
                        temperature=args.temperature,
                        max_concurrent=args.max_concurrent
                    )
                    stats_list.append(stats)
                return stats_list
            
            all_stats = asyncio.run(process_all_value_checks())
        else:
            # Sequential processing
            for query_id in tqdm(property_qrels.keys(), desc="Processing value checks"):
                if query_id not in query_dict:
                    print(f"Warning: Query {query_id} not found in query file")
                    continue
                
                query_obj = query_dict[query_id]
                passage_ids = property_qrels[query_id]
                
                stats = process_value_check(
                    query_obj=query_obj,
                    passage_ids=passage_ids,
                    corpus_dict=corpus_dict,
                    client=client,
                    qrel_output_file=qrel_file,
                    log_file=log_file,
                    model=args.model,
                    temperature=args.temperature
                )
                all_stats.append(stats)
        
        # Write summary
        log_file.write(f"\n\n{'='*80}\n")
        log_file.write("VALUE CHECK SUMMARY\n")
        log_file.write(f"{'='*80}\n")
        
        total_passages_checked = sum(s['total_passages_checked'] for s in all_stats)
        total_value_matched = sum(s['value_matched_passages'] for s in all_stats)
        
        summary_lines = [
            f"Total queries processed: {len(all_stats)}",
            f"Total passages checked: {total_passages_checked}",
            f"Passages with matching values: {total_value_matched}",
            f"Value match rate: {total_value_matched/total_passages_checked*100:.2f}%" if total_passages_checked > 0 else "Value match rate: N/A"
        ]
        
        for line in summary_lines:
            log_file.write(line + "\n")
            print(line)
        
        print(f"\nValue check qrels written to: {value_qrel_path}")
        print(f"Value check log written to: {value_log_file_path}")


def process_value_check(
    query_obj: Dict,
    passage_ids: List[str],
    corpus_dict: Dict[str, Dict],
    client: OpenAI,
    qrel_output_file: TextIO,
    log_file: TextIO,
    model: str = "gpt-4o",
    temperature: float = 0.0
) -> Dict:
    """
    Process value check for a single query.
    
    Args:
        query_obj: Query object from JSONL
        passage_ids: List of passage IDs that passed property check
        corpus_dict: In-memory corpus dictionary
        client: OpenAI client
        qrel_output_file: Open file handle for qrel output
        log_file: Open file handle for logging
        model: LLM model to use
        temperature: Temperature for LLM
    
    Returns:
        Statistics dict
    """
    query_id = query_obj['qid']
    
    # Handle different dataset structures
    if 'property' in query_obj and isinstance(query_obj['property'], dict):
        # QALD10 structure
        property_info = query_obj['property']
        entities_values = property_info.get('entities_values', [])
        property_metadata = property_info['property_info']
    else:
        # Quest structure
        entities_values = query_obj.get('entities_values', [])
        property_metadata = query_obj.get('property_info', {})
    
    property_label = property_metadata.get('label', 'unknown')
    property_description = property_metadata.get('description', '')
    property_id = property_metadata.get('property_id', property_metadata.get('id', 'unknown'))
    
    stats = {
        'total_passages_checked': 0,
        'value_matched_passages': 0
    }
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Query ID: {query_id}\n")
    log_file.write(f"Query: {query_obj.get('question', query_obj.get('original_query', 'N/A'))}\n")
    log_file.write(f"Property: {property_label} ({property_id}) - {property_description}\n")
    log_file.write(f"Passages to check: {len(passage_ids)}\n")
    log_file.write(f"{'='*80}\n\n")
    log_file.flush()  # Flush query header immediately
    
    # Create a mapping from entity QID to entity info
    entity_value_map = {}
    for entity_info in entities_values:
        qid = entity_info['entity_id']
        entity_value_map[qid] = {
            'label': entity_info.get('entity_label', ''),
            'value': entity_info.get('value', ''),
            'unit': entity_info.get('value_unit', None),
            'time': entity_info.get('value_time', None)
        }
    
    # Get passages from corpus and check values
    for passage_id in passage_ids:
        stats['total_passages_checked'] += 1
        
        # Get passage from corpus
        if passage_id not in corpus_dict:
            log_file.write(f"Warning: Passage {passage_id} not found in corpus\n")
            continue
        
        passage = corpus_dict[passage_id]
        passage_content = passage['contents']
        
        # Extract page_id from passage_id to map back to entity
        page_id = extract_page_id_from_passage_id(passage_id)
        
        # Try to find which entity this passage belongs to
        # We need to map page_id back to QID
        # For now, we'll check all entities and use the first match
        matched_entity = None
        for qid, entity_data in entity_value_map.items():
            # We need to get Wikipedia info for this QID
            wiki_info = get_wikipedia_info_from_qid(qid)
            if wiki_info and wiki_info['pageid'] == page_id:
                matched_entity = entity_data
                matched_entity['qid'] = qid
                matched_entity['wiki_title'] = wiki_info['title']
                break
        
        if not matched_entity:
            log_file.write(f"Warning: Could not map passage {passage_id} to entity\n")
            continue
        
        entity_label = matched_entity.get('wiki_title', matched_entity['label'])
        property_value = matched_entity['value']
        property_value_unit = matched_entity.get('unit')
        property_value_time = matched_entity.get('time')
        
        # Judge value match
        judgment = judge_passage_value(
            passage_content=passage_content,
            entity_label=entity_label,
            property_label=property_label,
            property_description=property_description,
            property_value=str(property_value),
            property_value_unit=property_value_unit,
            property_value_time=property_value_time,
            client=client,
            model=model,
            temperature=temperature
        )
        
        log_file.write(f"\nPassage ID: {passage_id}\n")
        log_file.write(f"Entity: {entity_label}\n")
        log_file.write(f"Expected value: {property_value}\n")
        if property_value_unit:
            log_file.write(f"Unit: {property_value_unit}\n")
        if property_value_time:
            log_file.write(f"Time: {property_value_time}\n")
        log_file.write(f"Content: {passage_content[:1000]}...\n")
        log_file.write(f"Value Match Judgment: {judgment}\n")
        log_file.flush()  # Flush log immediately
        
        # Only write to qrel file if judgment is YES
        if judgment == "YES":
            write_trec_qrel(qrel_output_file, query_id, passage_id, 1)
            qrel_output_file.flush()
            stats['value_matched_passages'] += 1
    
    return stats


async def process_value_check_async(
    query_obj: Dict,
    passage_ids: List[str],
    corpus_dict: Dict[str, Dict],
    client: AsyncOpenAI,
    qrel_output_file: TextIO,
    log_file: TextIO,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_concurrent: int = 10
) -> Dict:
    """
    Async version: Process value check for a single query with parallel LLM calls.
    
    Args:
        query_obj: Query object from JSONL
        passage_ids: List of passage IDs that passed property check
        corpus_dict: In-memory corpus dictionary
        client: AsyncOpenAI client
        qrel_output_file: Open file handle for qrel output
        log_file: Open file handle for logging
        model: LLM model to use
        temperature: Temperature for LLM
        max_concurrent: Maximum concurrent LLM API calls
    
    Returns:
        Statistics dict
    """
    query_id = query_obj['qid']
    
    # Handle different dataset structures
    if 'property' in query_obj and isinstance(query_obj['property'], dict):
        # QALD10 structure
        property_info = query_obj['property']
        entities_values = property_info.get('entities_values', [])
        property_metadata = property_info['property_info']
    else:
        # Quest structure
        entities_values = query_obj.get('entities_values', [])
        property_metadata = query_obj.get('property_info', {})
    
    property_label = property_metadata.get('label', 'unknown')
    property_description = property_metadata.get('description', '')
    property_id = property_metadata.get('property_id', property_metadata.get('id', 'unknown'))
    
    stats = {
        'total_passages_checked': 0,
        'value_matched_passages': 0
    }
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Query ID: {query_id}\n")
    log_file.write(f"Query: {query_obj.get('question', query_obj.get('original_query', 'N/A'))}\n")
    log_file.write(f"Property: {property_label} ({property_id}) - {property_description}\n")
    log_file.write(f"Passages to check: {len(passage_ids)}\n")
    log_file.write(f"{'='*80}\n\n")
    log_file.flush()  # Flush query header immediately
    
    # Create a mapping from entity QID to entity info
    entity_value_map = {}
    for entity_info in entities_values:
        qid = entity_info['entity_id']
        entity_value_map[qid] = {
            'label': entity_info.get('entity_label', ''),
            'value': entity_info.get('value', ''),
            'unit': entity_info.get('value_unit', None),
            'time': entity_info.get('value_time', None)
        }
    
    # Prepare passage data for parallel processing
    passage_data_list = []
    for passage_id in passage_ids:
        stats['total_passages_checked'] += 1
        
        if passage_id not in corpus_dict:
            log_file.write(f"Warning: Passage {passage_id} not found in corpus\n")
            continue
        
        passage = corpus_dict[passage_id]
        page_id = extract_page_id_from_passage_id(passage_id)
        
        # Find matching entity
        matched_entity = None
        for qid, entity_data in entity_value_map.items():
            wiki_info = get_wikipedia_info_from_qid(qid)
            if wiki_info and wiki_info['pageid'] == page_id:
                matched_entity = entity_data
                matched_entity['qid'] = qid
                matched_entity['wiki_title'] = wiki_info['title']
                break
        
        if not matched_entity:
            log_file.write(f"Warning: Could not map passage {passage_id} to entity\n")
            continue
        
        passage_data_list.append({
            'id': passage_id,
            'contents': passage['contents'],
            'entity_label': matched_entity.get('wiki_title', matched_entity['label']),
            'property_value': str(matched_entity['value']),
            'property_value_unit': matched_entity.get('unit'),
            'property_value_time': matched_entity.get('time')
        })
    
    if not passage_data_list:
        return stats
    
    # Create parallel tasks for value checking
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    
    for pdata in passage_data_list:
        task = judge_passage_value_async(
            passage_content=pdata['contents'],
            entity_label=pdata['entity_label'],
            property_label=property_label,
            property_description=property_description,
            property_value=pdata['property_value'],
            property_value_unit=pdata['property_value_unit'],
            property_value_time=pdata['property_value_time'],
            client=client,
            model=model,
            temperature=temperature,
            semaphore=semaphore
        )
        tasks.append(task)
    
    # Execute all tasks in parallel
    judgments = await asyncio.gather(*tasks)
    
    # Process results
    for pdata, judgment in zip(passage_data_list, judgments):
        log_file.write(f"\nPassage ID: {pdata['id']}\n")
        log_file.write(f"Entity: {pdata['entity_label']}\n")
        log_file.write(f"Expected value: {pdata['property_value']}\n")
        if pdata['property_value_unit']:
            log_file.write(f"Unit: {pdata['property_value_unit']}\n")
        if pdata['property_value_time']:
            log_file.write(f"Time: {pdata['property_value_time']}\n")
        log_file.write(f"Content: {pdata['contents'][:200]}...\n")
        log_file.write(f"Value Match Judgment: {judgment}\n")
        log_file.flush()  # Flush log immediately
        
        if judgment == "YES":
            write_trec_qrel(qrel_output_file, query_id, pdata['id'], 1)
            qrel_output_file.flush()
            stats['value_matched_passages'] += 1
    
    return stats


def property_check_main(args):
    """Main function to generate qrels using property check."""

    # Load queries
    print(f"Loading queries from {args.query_file}...")
    all_queries = read_jsonl_from_file(args.query_file)
    print(f"Loaded {len(all_queries)} queries")

    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        all_queries = all_queries[:args.limit]
        print(f"Limited to first {len(all_queries)} queries")

    # Handle resume mode
    queries, should_append = prepare_resume(args, all_queries)

    # Check if there are any queries to process
    if len(queries) == 0:
        print("\n" + "="*80)
        print("No queries to process! All queries already completed.")
        print("="*80)
        return

    # Initialize corpus loading based on mode
    corpus_dict = None
    page_index = None
    page2passage_mapping = None

    if args.load_corpus_mode == "memory":
        # Check if index cache exists
        cache_path = get_index_cache_path(args.corpus_jsonl, args.index_cache_dir)
        index_exists = cache_path.exists()

        # Load corpus into memory (will use cache if available)
        corpus_dict = load_corpus_in_memory(args.corpus_jsonl, args.index_cache_dir)

        # In memory mode, we need to build the page index from corpus_dict
        # The page index maps page_id -> list of passage_ids (not line numbers)
        # This is different from page2passage_mapping which maps page_id -> line numbers
        page_index = build_page_to_passages_index(corpus_dict, args.corpus_jsonl, args.index_cache_dir)

        # Save index if it was just built (not loaded from cache)
        if not index_exists:
            save_corpus_index(corpus_dict, page_index, cache_path)

        print(f"Corpus loading mode: IN-MEMORY")
        print(f"Memory usage: ~{len(corpus_dict)} passages loaded")
    else:
        # Streaming mode: optionally load page2passage mapping
        if args.page2passage_mapping:
            mapping_path = Path(args.page2passage_mapping)
            if mapping_path.exists():
                print(f"Loading page2passage mapping from {args.page2passage_mapping}...")
                page2passage_mapping = read_json_from_file(args.page2passage_mapping)
                print(f"Loaded mapping for {len(page2passage_mapping)} pages")
            else:
                print(f"\nWARNING: page2passage mapping file not found: {args.page2passage_mapping}")
                print("Building mapping from corpus (this may take a while)...")
                page2passage_mapping = build_page2passage_mapping_from_corpus(args.corpus_jsonl)
                print(f"Built mapping for {len(page2passage_mapping)} pages")

                # Save the mapping for future use
                print(f"Saving mapping to {args.page2passage_mapping}...")
                mapping_path.parent.mkdir(parents=True, exist_ok=True)
                with open(args.page2passage_mapping, 'w', encoding='utf-8') as f:
                    json.dump(page2passage_mapping, f)
                print(f"Mapping saved successfully")
        else:
            print("\nWARNING: No page2passage mapping file provided!")
            print("This will result in VERY SLOW performance as the entire corpus will be scanned for each query.")
            print("Consider providing --page2passage_mapping argument for better performance.")

        print(f"Corpus loading mode: STREAMING")

    # Initialize OpenAI clients (sync and async)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Use async client if parallel processing is enabled
    if args.use_parallel:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        print(f"Using PARALLEL processing mode with max_concurrent={args.max_concurrent}")
    else:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        print(f"Using SEQUENTIAL processing mode")

    # Create output directory if needed
    output_dir = Path(args.output_qrel).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache for QID to Wikipedia mappings
    qid2wikipedia_cache = {}

    # Determine file opening mode based on resume
    file_mode = 'a' if should_append else 'w'

    # Open output files
    with open(args.output_qrel, file_mode, encoding='utf-8') as qrel_file, \
         open(args.log_file, file_mode, encoding='utf-8') as log_file:

        # Write header (only if not appending)
        if not should_append:
            log_file.write(f"QRel Generation Log\n")
            log_file.write(f"Query file: {args.query_file}\n")
            log_file.write(f"Corpus: {args.corpus_jsonl}\n")
            log_file.write(f"Corpus loading mode: {args.load_corpus_mode}\n")
            log_file.write(f"Parallel processing: {args.use_parallel}\n")
            if args.use_parallel:
                log_file.write(f"Max concurrent LLM calls: {args.max_concurrent}\n")
            log_file.write(f"Model: {args.model}\n")
            log_file.write(f"Temperature: {args.temperature}\n")
            log_file.write(f"\n{'='*80}\n")
        else:
            # Add separator for resumed session
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"\n\n{'='*80}\n")
            log_file.write(f"RESUMED SESSION - {timestamp}\n")
            log_file.write(f"{'='*80}\n")
            log_file.write(f"Resuming with {len(queries)} remaining queries\n")
            log_file.write(f"{'='*80}\n\n")

        # Process queries
        all_stats = []

        if args.use_parallel:
            # Use async processing with parallel LLM calls
            async def process_all_queries():
                stats_list = []
                for i, query_obj in tqdm(enumerate(queries), desc="Processing queries"):
                    # if i == 5:
                    #     break

                    stats = await process_query_async(
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
                        page_index=page_index,
                        max_concurrent=args.max_concurrent
                    )
                    stats_list.append(stats)
                return stats_list

            all_stats = asyncio.run(process_all_queries())
        else:
            # Use sequential processing (original behavior)
            for i, query_obj in tqdm(enumerate(queries), desc="Processing queries"):
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

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        # Print summary statistics to both console and log file
        print_summary_stats(all_stats, len(queries), log_file)

        print(f"\nQRels written to: {args.output_qrel}")
        print(f"Log written to: {args.log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TREC-format qrels for Total Recall RAG queries")
    parser.add_argument("--dataset", type=str, default="quest", choices=["quest", "qald10"], help="Dataset name (e.g., 'quest', 'qald10'). Used to construct file paths.")
    parser.add_argument("--subset", type=str, default=None, help="Subset name (e.g., 'test_quest', 'train_quest'). If not provided, defaults to 'test_quest' for quest dataset.")
    parser.add_argument("--corpus_jsonl", type=str, default="corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl", help="Path to corpus JSONL file with passages")
    parser.add_argument("--log_dir", type=str, default="qrel_logging", help="Directory to write log file (log filename will be auto-generated based on dataset)")
    parser.add_argument("--page2passage_mapping", type=str, default="corpus_datasets/corpus/enwiki_20251001_infoboxconv.index.json", help="Optional: Path to JSON file mapping page IDs to passage indices for faster lookup (used in streaming mode)")
    parser.add_argument("--index_cache_dir", type=str, default="corpus_datasets/corpus", help="Optional: Directory to store/load cached in-memory index. If not specified, cache is stored next to corpus file. Cache file will be named '<corpus_stem>_memory_index.pkl'")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use for relevance judgment (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM generation (default: 0.0 for consistency)")
    parser.add_argument("--load_corpus_mode", type=str, default="stream", choices=["stream", "memory"], help="How to load corpus: 'stream' (low memory, slower per-query) or 'memory' (high memory, faster per-query). Default: 'stream'. Use 'memory' when processing many queries (100s+) and have sufficient RAM.")

    # Parallel processing arguments
    parser.add_argument("--use_parallel", action="store_true", help="Enable parallel LLM API calls for faster processing. This can provide 5-10x speedup depending on max_concurrent setting.")
    parser.add_argument("--max_concurrent", type=int, default=10, help="Maximum number of concurrent LLM API calls when --use_parallel is enabled. Higher values = faster but more API load. Recommended: 10-20. (default: 10)")

    # Limit number of queries
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of queries to process. Useful for testing or debugging. (default: None = process all)")

    # Resume functionality arguments
    parser.add_argument("--resume", action="store_true", help="Resume from existing qrel file by skipping already-completed queries. This allows you to continue generation after interruptions.")
    parser.add_argument("--reprocess-last", action="store_true", help="When resuming, always re-process the last query in the file (safest option, ensures completeness but may waste some LLM calls).")
    parser.add_argument("--trust-last", action="store_true", help="When resuming, trust that the last query in the file is complete (fastest option, but risky if interrupted mid-query).")
    parser.add_argument("--min-passages-threshold", type=int, default=5, help="Minimum number of passages to consider a query complete when resuming. If the last query has fewer passages than this threshold, it will be re-processed. (default: 5)")

    args = parser.parse_args()

    # Validate conflicting arguments
    if args.reprocess_last and args.trust_last:
        parser.error("--reprocess-last and --trust-last are mutually exclusive. Choose one or neither (to use heuristic).")

    # Determine subset and subdirectory
    # User can pass short form (e.g., "test") or full form (e.g., "test_quest")
    # For quest dataset: test -> test/, train -> train/, val -> val/
    # For other datasets (e.g., qald10): no subdirectory
    if args.subset:
        # If subset contains underscore, assume it's full form (e.g., "test_quest")
        if "_" in args.subset:
            subset_name = args.subset
            subdir = args.subset.split("_")[0]  # Extract prefix as subdir
        else:
            # Short form (e.g., "test") - construct full name
            subdir = args.subset
            subset_name = f"{args.subset}_{args.dataset}"  # e.g., "test_quest"
    else:
        # Default mapping: quest -> test_quest, qald10 -> qald10
        if args.dataset == "quest":
            subset_name = "test_quest"
            subdir = "test"
        else:
            subset_name = args.dataset
            subdir = ""

    # Use the generations file which contains full metadata including entity values
    if subdir:
        args.query_file = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/{subdir}/{subset_name}_generations.jsonl"
        args.output_qrel = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/{subdir}/qrels_{subset_name}.txt"
    else:
        args.query_file = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/{subset_name}_generations.jsonl"
        args.output_qrel = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/qrels_{subset_name}.txt"

    # Create log file path based on subset name with timestamp
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_file = str(log_dir / f"qrel_generation_{subset_name}_{timestamp}.log")

    # Print configuration
    print("="*80)
    print("QRel Generation Configuration")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Subset: {subset_name}")
    print(f"Query file: {args.query_file}")
    print(f"Output qrel: {args.output_qrel}")
    print(f"Log file: {args.log_file}")
    print(f"Corpus: {args.corpus_jsonl}")
    print(f"Corpus loading mode: {args.load_corpus_mode.upper()}")
    print(f"Parallel processing: {'ENABLED' if args.use_parallel else 'DISABLED'}")
    if args.use_parallel:
        print(f"Max concurrent LLM calls: {args.max_concurrent}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Query limit: {args.limit if args.limit else 'None (all queries)'}")
    print(f"Resume mode: {'ENABLED' if args.resume else 'DISABLED'}")
    if args.resume:
        if args.reprocess_last:
            print(f"  Strategy: Always re-process last query (safest)")
        elif args.trust_last:
            print(f"  Strategy: Trust last query is complete (fastest)")
        else:
            print(f"  Strategy: Heuristic (threshold: {args.min_passages_threshold} passages)")
    print("="*80)
    print()

    # Run property check first
    # property_check_main(args)

    # Calculate and display entity coverage after qrel generation completes
    # print("\n" + "="*80)
    # print("Calculating entity coverage...")
    # print("="*80 + "\n")

    # coverage_results = calculate_coverage(
    #     dataset=args.dataset,
    #     qrel_file_path=args.output_qrel,
    #     subset=args.subset
    # )
    
    # Now run value check on the property check results
    print("\n" + "="*80)
    print("STARTING VALUE CHECK PHASE")
    print("="*80)
    check_value(args)
    
    # Calculate and display entity coverage for value check results
    print("\n" + "="*80)
    print("Calculating entity coverage for value check results...")
    print("="*80 + "\n")
    
    # Construct value qrel path (same logic as in check_value function)
    property_qrel_path = args.output_qrel
    if "/qrels_" in property_qrel_path:
        value_qrel_path = property_qrel_path.replace("/qrels_", "/qrels_value_")
    else:
        value_qrel_path = property_qrel_path.replace("qrels_", "qrels_value_")
    
    value_coverage_results = calculate_coverage(
        dataset=args.dataset,
        qrel_file_path=value_qrel_path,
        subset=args.subset
    )


# ============================================================================
# Usage Examples
# ============================================================================
#
# python c3_qrel_generation/qrel_generation.py --dataset quest --subset val --use_parallel --load_corpus_mode memory --max_concurrent 20 --limit 10


# 1. PARALLEL PROCESSING MODE (RECOMMENDED - 5-10x faster!):
#    python c3_qrel_generation/qrel_generation.py --dataset qald10 --use_parallel --max_concurrent 20
#
#    With custom concurrency (higher = faster, but more API load):
#    python c3_qrel_generation/qrel_generation.py --dataset quest --use_parallel --max_concurrent 20
#    python c3_qrel_generation/qrel_generation.py --dataset quest --use_parallel --max_concurrent 20 --resume --reprocess-last
#
#    With memory mode for even faster passage retrieval:
#    python c3_qrel_generation/qrel_generation.py --dataset qald10 --use_parallel --load_corpus_mode memory
#
# 2. STREAMING MODE (Default - Low memory, sequential processing):
#    python c3_qrel_generation/qrel_generation.py --dataset qald10
#
#    OR explicitly:
#    python c3_qrel_generation/qrel_generation.py --dataset qald10 --load_corpus_mode stream
#
# 3. IN-MEMORY MODE (High memory, faster passage retrieval):
#    python c3_qrel_generation/qrel_generation.py --dataset qald10 --load_corpus_mode memory
#
#    The first time you run in memory mode, it will:
#    - Load the entire corpus into memory (slow, one-time operation)
#    - Build the page-to-passage index
#    - Save both to a cache file (default: next to corpus as <corpus_name>_memory_index.pkl)
#
#    Subsequent runs will load from the cache file (much faster!)
#
# 4. IN-MEMORY MODE with custom cache directory:
#    python c3_qrel_generation/qrel_generation.py --dataset qald10 --load_corpus_mode memory \
#           --index_cache_dir /path/to/cache/dir
#
# 5. RESUME MODE (Continue from interruption):
#    If your qrel generation was interrupted, you can resume from where you left off:
#
#    Default resume (uses heuristic to detect incomplete last query):
#    python c3_qrel_generation/qrel_generation.py --dataset quest --resume --use_parallel
#
#    Safe resume (always re-processes last query):
#    python c3_qrel_generation/qrel_generation.py --dataset quest --resume --reprocess-last --use_parallel
#
#    Fast resume (trusts last query is complete):
#    python c3_qrel_generation/qrel_generation.py --dataset quest --resume --trust-last --use_parallel
#
#    Custom threshold (re-process if last query has < N passages):
#    python c3_qrel_generation/qrel_generation.py --dataset quest --resume --min-passages-threshold 10 --use_parallel
#
# Performance Comparison:
# ----------------------
# For 700 passages (typical query):
# - Sequential mode: ~17-35 minutes
# - Parallel mode (max_concurrent=10): ~2-4 minutes (10x faster!)
# - Parallel mode (max_concurrent=20): ~1-2 minutes (20x faster!)
#
# When to use each mode:
# - PARALLEL + MEMORY: Best performance, highest throughput (RECOMMENDED)
# - PARALLEL + STREAM: Good performance, lower memory usage
# - SEQUENTIAL + STREAM: Lowest resource usage, slowest (default)
# - SEQUENTIAL + MEMORY: Faster passage retrieval, but still slow LLM calls
#
# Resume Mode Strategies:
# ----------------------
# - Default (heuristic): Re-processes last query if it has < 5 passages (balanced)
# - --reprocess-last: Always re-processes last query (safest, may waste some LLM calls)
# - --trust-last: Never re-processes last query (fastest, risky if interrupted mid-query)
# - --min-passages-threshold N: Custom threshold for heuristic (flexible)
#
# Note: The in-memory index cache can be very large (multi-GB). Make sure you have
# sufficient disk space in the cache directory.
#
# ============================================================================ 
