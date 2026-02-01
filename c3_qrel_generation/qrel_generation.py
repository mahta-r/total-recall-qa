"""
QRel Generation for Total Recall RAG

This script combines:
- Prompt approach from c1_1_dataset_creation_mahta/query_generation/extract_qrels.py

It generates TREC-format qrels by:
1. Reading queries from JSONL file
2. For each query, mapping Wikidata QIDs to Wikipedia pages
3. Retrieving passages from those Wikipedia pages
4. Using LLM as judge with 4-way classification (YES-SAME, YES-DIFFERENT, NO-RELATED, NO-UNRELATED)
5. Using LLM as rewriter for YES-DIFFERENT and NO-RELATED cases
6. Writing qrels in TREC format
"""

import os
import sys
import json
import argparse
import requests
import pickle
import asyncio
import copy
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, TextIO
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions
from c1_1_dataset_creation_mahta.io_utils import read_jsonl_from_file, read_json_from_file, write_json_to_file
from c1_1_dataset_creation_mahta.wikidata.data_utils import format_value, format_time_for_prompt

# Import prompts from c1_1 (extract_qrels approach)
from c1_1_dataset_creation_mahta.query_generation.prompts.LLM_as_relevance_judge.prompt_combined import (
    PROPERTY_CHECK_PROMPT,
    PROPERTY_CHECK_INPUT_TEMPLATE,
    DATATYPE_EXPLANATION
)
from c1_1_dataset_creation_mahta.query_generation.prompts.LLM_as_relevance_judge.prompt_rewrite import (
    REWRITE_PROMPT,
    REWRITE_EXPLANATION
)

# Import analysis functions from qrel_analysis
from c3_qrel_generation.qrel_analysis import calculate_coverage


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
            "User-Agent": "TotalRecallRAG/1.0 (Research Project)"
        }
        
        # Query Wikidata API
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
        
        # Get Wikipedia page ID
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
    """Get the cache file path for a corpus index."""
    corpus_path = Path(corpus_jsonl_path)
    cache_filename = f"{corpus_path.stem}_memory_index.pkl"
    
    if index_cache_dir:
        cache_dir = Path(index_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / cache_filename
    else:
        return corpus_path.parent / cache_filename


def save_corpus_index(corpus_dict: Dict[str, Dict], page_index: Dict[str, List[str]], cache_path: Path):
    """Save corpus_dict and page_index to disk."""
    print(f"Saving corpus index to {cache_path}...")
    index_data = {
        'corpus_dict': corpus_dict,
        'page_index': page_index
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Saved index to {cache_path} ({size_mb:.1f} MB)")


def load_corpus_index(cache_path: Path) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
    """Load corpus_dict and page_index from disk cache."""
    print(f"Loading corpus index from cache: {cache_path}...")
    
    with open(cache_path, 'rb') as f:
        index_data = pickle.load(f)
    
    corpus_dict = index_data['corpus_dict']
    page_index = index_data['page_index']
    
    print(f"Loaded {len(corpus_dict)} passages and {len(page_index)} pages from cache")
    return corpus_dict, page_index


def load_corpus_in_memory(corpus_jsonl_path: str, index_cache_dir: Optional[str] = None) -> Dict[str, Dict]:
    """Load entire corpus into memory as a dictionary for fast lookups."""
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
            try:
                passage = json.loads(line.strip())
                if 'id' not in passage:
                    print(f"Warning: passage missing 'id' field, skipping", file=sys.stderr)
                    continue
                corpus_dict[passage['id']] = passage
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse line, skipping: {e}", file=sys.stderr)
                continue
    
    print(f"Loaded {len(corpus_dict)} passages into memory")
    return corpus_dict


def build_page_to_passages_index(
    corpus_dict: Dict[str, Dict],
    corpus_jsonl_path: str,
    index_cache_dir: Optional[str] = None
) -> Dict[str, List[str]]:
    """Build an index from page ID to list of passage IDs."""
    cache_path = get_index_cache_path(corpus_jsonl_path, index_cache_dir)
    
    if cache_path.exists():
        print(f"Found cached index at {cache_path}")
        try:
            _, page_index = load_corpus_index(cache_path)
            return page_index
        except Exception as e:
            print(f"Warning: Failed to load cached index ({e}), rebuilding from scratch...")
    
    print("Building page-to-passages index...")
    page_index = defaultdict(list)
    
    for passage_id in tqdm(corpus_dict.keys(), desc="Building page index"):
        page_id = extract_page_id_from_passage_id(passage_id)
        if page_id:
            page_index[page_id].append(passage_id)
    
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
                if passage_id in corpus_dict:
                    passages.append(corpus_dict[passage_id])
                else:
                    print(f"Warning: passage_id {passage_id} not found in corpus_dict", file=sys.stderr)
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
    passage: Dict,
    entity_label: str,
    property_label: str,
    property_description: str,
    property_value: str,
    property_datatype: str,
    shared_time: Optional[Dict],
    client: OpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0
) -> str:
    """
    Use LLM to judge passage relevance with 4-way classification.
    
    Returns one of: "YES-SAME", "YES-DIFFERENT", "NO-RELATED", "NO-UNRELATED"
    """
    # Prepare time statement
    if shared_time is not None:
        shared_time_str = format_time_for_prompt(shared_time=shared_time)
        statement_time = f"this statement is valid as of {shared_time_str}."
    else:
        statement_time = ""
    
    # Prepare input
    input_instance = PROPERTY_CHECK_INPUT_TEMPLATE.format(
        entity_name=entity_label,
        property_name=property_label,
        property_description=property_description,
        property_value=property_value,
        time_of_statement=statement_time,
        passage_title=passage.get('title', ''),
        sections=" - ".join(passage.get('section', [])),
        passage=passage.get('contents', '')
    )
    
    # Prepare prompt with datatype explanation
    prompt = PROPERTY_CHECK_PROMPT.format(
        statement_explanation=DATATYPE_EXPLANATION[property_datatype]
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_instance}
            ],
            temperature=temperature,
        )
        
        response = resp.choices[0].message.content.strip()
        
        # Parse response
        if response not in ["YES-SAME", "YES-DIFFERENT", "NO-RELATED", "NO-UNRELATED"]:
            if "YES-SAME".lower() in response.lower():
                response = "YES-SAME"
            elif "YES-DIFFERENT".lower() in response.lower():
                response = "YES-DIFFERENT"
            elif "NO-RELATED".lower() in response.lower():
                response = "NO-RELATED"
            elif "NO-UNRELATED".lower() in response.lower():
                response = "NO-UNRELATED"
            else:
                response = "NO-UNRELATED"
        
        return response
    
    except Exception as e:
        print(f"Error in LLM judgment: {e}", file=sys.stderr)
        return "NO-UNRELATED"


async def judge_passage_relevance_async(
    passage: Dict,
    entity_label: str,
    property_label: str,
    property_description: str,
    property_value: str,
    property_datatype: str,
    shared_time: Optional[Dict],
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    semaphore: Optional[asyncio.Semaphore] = None
) -> str:
    """Async version: Use LLM to judge passage relevance with 4-way classification."""
    # Prepare time statement
    if shared_time is not None:
        shared_time_str = format_time_for_prompt(shared_time=shared_time)
        statement_time = f"this statement is valid as of {shared_time_str}."
    else:
        statement_time = ""
    
    # Prepare input
    input_instance = PROPERTY_CHECK_INPUT_TEMPLATE.format(
        entity_name=entity_label,
        property_name=property_label,
        property_description=property_description,
        property_value=property_value,
        time_of_statement=statement_time,
        passage_title=passage.get('title', ''),
        sections=" - ".join(passage.get('section', [])),
        passage=passage.get('contents', '')
    )
    
    # Prepare prompt with datatype explanation
    prompt = PROPERTY_CHECK_PROMPT.format(
        statement_explanation=DATATYPE_EXPLANATION[property_datatype]
    )
    
    async def _make_request():
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_instance}
                ],
                temperature=temperature,
            )
            
            response = resp.choices[0].message.content.strip()
            
            # Parse response
            if response not in ["YES-SAME", "YES-DIFFERENT", "NO-RELATED", "NO-UNRELATED"]:
                if "YES-SAME".lower() in response.lower():
                    response = "YES-SAME"
                elif "YES-DIFFERENT".lower() in response.lower():
                    response = "YES-DIFFERENT"
                elif "NO-RELATED".lower() in response.lower():
                    response = "NO-RELATED"
                elif "NO-UNRELATED".lower() in response.lower():
                    response = "NO-UNRELATED"
                else:
                    response = "NO-UNRELATED"
            
            return response
        
        except Exception as e:
            print(f"Error in LLM judgment: {e}", file=sys.stderr)
            return "NO-UNRELATED"
    
    if semaphore:
        async with semaphore:
            return await _make_request()
    else:
        return await _make_request()


async def rewrite_passage_async(
    passage: Dict,
    entity_label: str,
    property_label: str,
    property_description: str,
    property_value: str,
    shared_time: Optional[Dict],
    rewrite_type: str,
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    semaphore: Optional[asyncio.Semaphore] = None
) -> str:
    """
    Async version: Use LLM to rewrite passage content.
    
    Args:
        rewrite_type: Either "REPLACE" or "ADD"
    """
    # Prepare time statement
    if shared_time is not None:
        shared_time_str = format_time_for_prompt(shared_time=shared_time)
        statement_time = f"this statement is valid as of {shared_time_str}."
    else:
        statement_time = ""
    
    # Prepare input
    input_instance = PROPERTY_CHECK_INPUT_TEMPLATE.format(
        entity_name=entity_label,
        property_name=property_label,
        property_description=property_description,
        property_value=property_value,
        time_of_statement=statement_time,
        passage_title=passage.get('title', ''),
        sections=" - ".join(passage.get('section', [])),
        passage=passage.get('contents', '')
    )
    
    # Prepare prompt with rewrite explanation
    prompt = REWRITE_PROMPT.format(
        rewrite_explanation=REWRITE_EXPLANATION[rewrite_type]
    )
    
    async def _make_request():
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_instance}
                ],
                temperature=temperature,
            )
            
            response = resp.choices[0].message.content.strip()
            response = re.sub(r'\s+', ' ', response).strip()
            
            return response
        
        except Exception as e:
            print(f"Error in LLM rewriting: {e}", file=sys.stderr)
            return passage.get('contents', '')
    
    if semaphore:
        async with semaphore:
            return await _make_request()
    else:
        return await _make_request()


def rewrite_passage(
    passage: Dict,
    entity_label: str,
    property_label: str,
    property_description: str,
    property_value: str,
    shared_time: Optional[Dict],
    rewrite_type: str,
    client: OpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.7
) -> str:
    """
    Use LLM to rewrite passage content.
    
    Args:
        rewrite_type: Either "REPLACE" or "ADD"
    """
    # Prepare time statement
    if shared_time is not None:
        shared_time_str = format_time_for_prompt(shared_time=shared_time)
        statement_time = f"this statement is valid as of {shared_time_str}."
    else:
        statement_time = ""
    
    # Prepare input (reusing the same template)
    input_instance = PROPERTY_CHECK_INPUT_TEMPLATE.format(
        entity_name=entity_label,
        property_name=property_label,
        property_description=property_description,
        property_value=property_value,
        time_of_statement=statement_time,
        passage_title=passage.get('title', ''),
        sections=" - ".join(passage.get('section', [])),
        passage=passage.get('contents', '')
    )
    
    # Prepare prompt with rewrite explanation
    prompt = REWRITE_PROMPT.format(
        rewrite_explanation=REWRITE_EXPLANATION[rewrite_type]
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_instance}
            ],
            temperature=temperature,
        )
        
        response = resp.choices[0].message.content.strip()
        response = re.sub(r'\s+', ' ', response).strip()
        
        return response
    
    except Exception as e:
        print(f"Error in LLM rewriting: {e}", file=sys.stderr)
        return passage.get('contents', '')


def write_trec_qrel(
    output_file: TextIO,
    query_id: str,
    passage_id: str,
    relevance: int
):
    """Write a single qrel line in TREC format."""
    output_file.write(f"{query_id} 0 {passage_id} {relevance}\n")


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


def get_all_query_ids_in_rewrites(rewrites_file_path):
    """
    Extract all query IDs from rewrites file.
    
    Args:
        rewrites_file_path: Path to rewrites JSONL file
    
    Returns:
        Set of query IDs that have rewrites
    """
    query_ids = set()
    
    if Path(rewrites_file_path).exists():
        with open(rewrites_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    query_id = record.get('query_id')
                    if query_id:
                        query_ids.add(query_id)
                except json.JSONDecodeError:
                    continue
    
    return query_ids


def remove_query_from_rewrites(rewrites_file_path, query_id_to_remove):
    """
    Remove all rewrite entries for a specific query from rewrites file.
    
    Args:
        rewrites_file_path: Path to rewrites JSONL file
        query_id_to_remove: Query ID whose rewrites should be removed
    
    Returns:
        Number of entries removed
    """
    if not Path(rewrites_file_path).exists():
        return 0
    
    temp_file = Path(rewrites_file_path).with_suffix('.tmp')
    
    removed_count = 0
    with open(rewrites_file_path, 'r', encoding='utf-8') as infile, \
         open(temp_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line_stripped = line.strip()
            if not line_stripped:
                outfile.write(line)
                continue
            try:
                record = json.loads(line_stripped)
                if record.get('query_id') == query_id_to_remove:
                    removed_count += 1
                    continue  # Skip this line
                outfile.write(line)
            except json.JSONDecodeError:
                outfile.write(line)  # Keep malformed lines
    
    # Replace original with cleaned version
    temp_file.replace(rewrites_file_path)
    return removed_count


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
        print(f"   Removing existing qrel entries for '{query_to_reprocess}'...")
        removed_qrel_count = remove_query_from_qrel(args.output_qrel, query_to_reprocess)
        print(f"   Removed {removed_qrel_count} qrel entries")
        
        # Also clean up rewrites file if it exists
        if args.passage_rewrites_path and Path(args.passage_rewrites_path).exists():
            print(f"   Removing existing rewrite entries for '{query_to_reprocess}'...")
            removed_rewrite_count = remove_query_from_rewrites(args.passage_rewrites_path, query_to_reprocess)
            print(f"   Removed {removed_rewrite_count} rewrite entries")
    
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


def append_rewrite_to_file(rewrites_path: str, rewrite_record: Dict):
    """
    Append a rewrite record to the rewrites file.
    
    Args:
        rewrites_path: Path to the rewrites JSONL file
        rewrite_record: Dictionary containing rewrite information
    """
    with open(rewrites_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rewrite_record, ensure_ascii=False) + '\n')
        f.flush()  # Flush immediately to disk


async def judge_passages_batch(
    passages: List[Dict],
    entity_label: str,
    property_label: str,
    property_description: str,
    property_value: str,
    property_datatype: str,
    shared_time: Optional[Dict],
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_concurrent: int = 10
) -> List[Tuple[Dict, str]]:
    """
    Judge multiple passages in parallel with rate limiting.
    
    Returns:
        List of tuples (passage_dict, judgment) where judgment is one of the 4 categories
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = []
    for passage in passages:
        task = judge_passage_relevance_async(
            passage=passage,
            entity_label=entity_label,
            property_label=property_label,
            property_description=property_description,
            property_value=property_value,
            property_datatype=property_datatype,
            shared_time=shared_time,
            client=client,
            model=model,
            temperature=temperature,
            semaphore=semaphore
        )
        tasks.append(task)
    
    judgments = await asyncio.gather(*tasks)
    
    results = list(zip(passages, judgments))
    return results


def process_query(
    query_obj: Dict,
    corpus_jsonl_path: Optional[str],
    judge_client: OpenAI,
    rewrite_client: OpenAI,
    qrel_output_file: TextIO,
    log_file: TextIO,
    judge_model: str = "gpt-4o",
    judge_temperature: float = 0.0,
    rewrite_model: str = "gpt-4o",
    rewrite_temperature: float = 0.7,
    page2passage_mapping: Optional[Dict] = None,
    qid2wikipedia_cache: Optional[Dict] = None,
    corpus_dict: Optional[Dict[str, Dict]] = None,
    page_index: Optional[Dict[str, List[str]]] = None,
    rewrite_passages: bool = True,
    passage_rewrites_path: Optional[str] = None
) -> Dict:
    """
    Process a single query to generate qrels using the extract_qrels approach.
    """
    query_id = query_obj['qid']
    
    # Handle different dataset structures
    if 'property' in query_obj and isinstance(query_obj['property'], dict):
        property_info = query_obj['property']
        entities_values = property_info.get('entities_values', [])
        property_metadata = property_info['property_info']
    else:
        entities_values = query_obj.get('entities_values', [])
        property_metadata = query_obj.get('property_info', {})
    
    property_label = property_metadata.get('label', 'unknown')
    property_description = property_metadata.get('description', '')
    property_id = property_metadata.get('property_id', property_metadata.get('id', 'unknown'))
    property_datatype = property_metadata.get('datatype', 'WikibaseItem')
    
    # Get shared_time if it exists
    shared_time = property_metadata.get('shared_time', None)
    
    stats = {
        'total_qids': len(entities_values),
        'qids_with_wikipedia': 0,
        'total_passages_checked': 0,
        'yes_same_passages': 0,
        'yes_different_passages': 0,
        'no_related_passages': 0,
        'no_unrelated_passages': 0,
        'relevant_passages': 0,
        'rewrites_performed': 0,
        'entities_with_relevant_passages': 0
    }
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Query ID: {query_id}\n")
    log_file.write(f"Query: {query_obj.get('question', query_obj.get('original_query', 'N/A'))}\n")
    log_file.write(f"Property: {property_label} ({property_id}) - {property_description}\n")
    log_file.write(f"Property datatype: {property_datatype}\n")
    log_file.write(f"{'='*80}\n\n")
    
    # Process each entity
    for entity_info in entities_values:
        qid = entity_info['entity_id']
        entity_label = entity_info.get('entity_label', '')
        
        # Format the property value (handle both Mahta and Heydar formats)
        value_node = entity_info.get('value_node', entity_info.get('value', ''))
        property_value = format_value_flexible(property_datatype, value_node)
        
        log_file.write(f"\n--- Processing Entity: {entity_label} ({qid}) ---\n")
        log_file.write(f"Property value: {property_value}\n")
        
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
        pageid = wiki_info['pageid']
        
        log_file.write(f"Wikipedia Title: {wiki_info['title']}\n")
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
        
        # Classify each passage
        label_to_passage = defaultdict(list)
        
        for passage in passages:
            passage_id = passage.get('id', 'unknown')
            
            # Judge relevance
            judgment = judge_passage_relevance(
                passage=passage,
                entity_label=entity_label,
                property_label=property_label,
                property_description=property_description,
                property_value=property_value,
                property_datatype=property_datatype,
                shared_time=shared_time,
                client=judge_client,
                model=judge_model,
                temperature=judge_temperature
            )
            
            label_to_passage[judgment].append(passage)
            
            log_file.write(f"\nPassage ID: {passage_id}\n")
            log_file.write(f"Content: {passage.get('contents', '')[:200]}...\n")
            log_file.write(f"Judgment: {judgment}\n")
        
        # Update stats
        stats['yes_same_passages'] += len(label_to_passage['YES-SAME'])
        stats['yes_different_passages'] += len(label_to_passage['YES-DIFFERENT'])
        stats['no_related_passages'] += len(label_to_passage['NO-RELATED'])
        stats['no_unrelated_passages'] += len(label_to_passage['NO-UNRELATED'])
        
        # Handle YES-SAME passages (no rewrite needed)
        for passage in label_to_passage['YES-SAME']:
            write_trec_qrel(qrel_output_file, query_id, passage.get('id', 'unknown'), 1)
            qrel_output_file.flush()
            stats['relevant_passages'] += 1
            entity_has_relevant_passage = True
        
        # Handle YES-DIFFERENT passages (rewrite with REPLACE)
        if rewrite_passages:
            for passage in label_to_passage['YES-DIFFERENT']:
                rewritten_content = rewrite_passage(
                    passage=passage,
                    entity_label=entity_label,
                    property_label=property_label,
                    property_description=property_description,
                    property_value=property_value,
                    shared_time=shared_time,
                    rewrite_type="REPLACE",
                    client=rewrite_client,
                    model=rewrite_model,
                    temperature=rewrite_temperature
                )
                
                log_file.write(f"\n[REWRITE-REPLACE] Passage ID: {passage.get('id', 'unknown')}\n")
                log_file.write(f"Original: {passage.get('contents', '')[:200]}...\n")
                log_file.write(f"Rewritten: {rewritten_content[:200]}...\n")
                
                # Save rewrite to file
                if passage_rewrites_path:
                    rewritten_passage = copy.deepcopy(passage)
                    rewritten_passage['contents'] = rewritten_content
                    rewrite_record = {
                        "query_id": query_id,
                        "passage_id": passage.get('id', 'unknown'),
                        "passage_title": passage.get('title', ''),
                        "entity_label": entity_label,
                        "entity_qid": qid,
                        "property_label": property_label,
                        "property_id": property_id,
                        "property_value": property_value,
                        "shared_time": shared_time,
                        "rewrite_type": "REPLACE",
                        "passage": passage,
                        "rewritten_passage": rewritten_passage
                    }
                    append_rewrite_to_file(passage_rewrites_path, rewrite_record)
                
                write_trec_qrel(qrel_output_file, query_id, passage.get('id', 'unknown'), 1)
                qrel_output_file.flush()
                stats['relevant_passages'] += 1
                stats['rewrites_performed'] += 1
                entity_has_relevant_passage = True
        
        # Handle NO-RELATED passages (if no relevant passages found, rewrite with ADD)
        if rewrite_passages and not entity_has_relevant_passage and len(label_to_passage['NO-RELATED']) > 0:
            # Choose the first NO-RELATED passage
            passage = label_to_passage['NO-RELATED'][0]
            
            rewritten_content = rewrite_passage(
                passage=passage,
                entity_label=entity_label,
                property_label=property_label,
                property_description=property_description,
                property_value=property_value,
                shared_time=shared_time,
                rewrite_type="ADD",
                client=rewrite_client,
                model=rewrite_model,
                temperature=rewrite_temperature
            )
            
            log_file.write(f"\n[REWRITE-ADD] Passage ID: {passage.get('id', 'unknown')}\n")
            log_file.write(f"Original: {passage.get('contents', '')[:200]}...\n")
            log_file.write(f"Rewritten: {rewritten_content[:200]}...\n")
            
            # Save rewrite to file
            if passage_rewrites_path:
                rewritten_passage = copy.deepcopy(passage)
                rewritten_passage['contents'] = rewritten_content
                rewrite_record = {
                    "query_id": query_id,
                    "passage_id": passage.get('id', 'unknown'),
                    "passage_title": passage.get('title', ''),
                    "entity_label": entity_label,
                    "entity_qid": qid,
                    "property_label": property_label,
                    "property_id": property_id,
                    "property_value": property_value,
                    "shared_time": shared_time,
                    "rewrite_type": "ADD",
                    "passage": passage,
                    "rewritten_passage": rewritten_passage
                }
                append_rewrite_to_file(passage_rewrites_path, rewrite_record)
            
            write_trec_qrel(qrel_output_file, query_id, passage.get('id', 'unknown'), 1)
            qrel_output_file.flush()
            stats['relevant_passages'] += 1
            stats['rewrites_performed'] += 1
            entity_has_relevant_passage = True
        
        # Update entity coverage count
        if entity_has_relevant_passage:
            stats['entities_with_relevant_passages'] += 1
        
        log_file.write(f"\nEntity summary: {entity_label}\n")
        log_file.write(f"  YES-SAME: {len(label_to_passage['YES-SAME'])}\n")
        log_file.write(f"  YES-DIFFERENT: {len(label_to_passage['YES-DIFFERENT'])}\n")
        log_file.write(f"  NO-RELATED: {len(label_to_passage['NO-RELATED'])}\n")
        log_file.write(f"  NO-UNRELATED: {len(label_to_passage['NO-UNRELATED'])}\n")
    
    return stats


def format_value_flexible(datatype: str, value_node) -> str:
    """
    Format a property value for display, handling both Mahta and Heydar dataset formats.
    
    Mahta format: value_node is a dict with 'value', 'unit_id', 'precision', etc.
    Heydar format: value_node is a simple scalar (int, float, str) or list
    """
    # Handle simple scalar values (Heydar format)
    if isinstance(value_node, (int, float)):
        return str(value_node)
    elif isinstance(value_node, str):
        return value_node
    elif isinstance(value_node, list):
        # For WikibaseItem lists in Heydar format
        return ", ".join(str(v) for v in value_node)
    
    # Handle dict format (Mahta format)
    # Check if it's a proper Mahta format dict with expected keys
    if isinstance(value_node, dict):
        # For Time datatype, check if it has the expected structure
        if datatype == "Time":
            # Mahta format should have 'value' key with datetime or string
            if 'value' in value_node and isinstance(value_node.get('value'), (str, int, float)):
                # If value is just a string/int (simplified format), convert directly
                return str(value_node['value'])
            elif 'value' in value_node:
                # Full Mahta format with datetime object
                return format_value(datatype, value_node)
            else:
                # Fallback
                return str(value_node)
        
        # For Quantity datatype, check structure
        elif datatype == "Quantity":
            if 'value' in value_node:
                return format_value(datatype, value_node)
            else:
                return str(value_node)
        
        # For other types, use format_value
        else:
            return format_value(datatype, value_node)
    
    # Fallback
    return str(value_node)


async def process_query_async(
    query_obj: Dict,
    corpus_jsonl_path: Optional[str],
    judge_client: AsyncOpenAI,
    rewrite_client: AsyncOpenAI,
    qrel_output_file: TextIO,
    log_file: TextIO,
    judge_model: str = "gpt-4o",
    judge_temperature: float = 0.0,
    rewrite_model: str = "gpt-4o",
    rewrite_temperature: float = 0.7,
    page2passage_mapping: Optional[Dict] = None,
    qid2wikipedia_cache: Optional[Dict] = None,
    corpus_dict: Optional[Dict[str, Dict]] = None,
    page_index: Optional[Dict[str, List[str]]] = None,
    rewrite_passages: bool = True,
    max_concurrent: int = 10,
    passage_rewrites_path: Optional[str] = None
) -> Dict:
    """
    Async version: Process a single query with parallel LLM calls for judging passages.
    """
    query_id = query_obj['qid']
    
    # Handle different dataset structures
    if 'property' in query_obj and isinstance(query_obj['property'], dict):
        property_info = query_obj['property']
        entities_values = property_info.get('entities_values', [])
        property_metadata = property_info['property_info']
    else:
        entities_values = query_obj.get('entities_values', [])
        property_metadata = query_obj.get('property_info', {})
    
    property_label = property_metadata.get('label', 'unknown')
    property_description = property_metadata.get('description', '')
    property_id = property_metadata.get('property_id', property_metadata.get('id', 'unknown'))
    property_datatype = property_metadata.get('datatype', 'WikibaseItem')
    
    # Get shared_time if it exists
    shared_time = property_metadata.get('shared_time', None)
    
    stats = {
        'total_qids': len(entities_values),
        'qids_with_wikipedia': 0,
        'total_passages_checked': 0,
        'yes_same_passages': 0,
        'yes_different_passages': 0,
        'no_related_passages': 0,
        'no_unrelated_passages': 0,
        'relevant_passages': 0,
        'rewrites_performed': 0,
        'entities_with_relevant_passages': 0
    }
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Query ID: {query_id}\n")
    log_file.write(f"Query: {query_obj.get('question', query_obj.get('original_query', 'N/A'))}\n")
    log_file.write(f"Property: {property_label} ({property_id}) - {property_description}\n")
    log_file.write(f"Property datatype: {property_datatype}\n")
    log_file.write(f"{'='*80}\n\n")
    
    # Process each entity
    for entity_info in entities_values:
        qid = entity_info['entity_id']
        entity_label = entity_info.get('entity_label', '')
        
        # Format the property value (handle both Mahta and Heydar formats)
        value_node = entity_info.get('value_node', entity_info.get('value', ''))
        property_value = format_value_flexible(property_datatype, value_node)
        
        log_file.write(f"\n--- Processing Entity: {entity_label} ({qid}) ---\n")
        log_file.write(f"Property value: {property_value}\n")
        
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
        pageid = wiki_info['pageid']
        
        log_file.write(f"Wikipedia Title: {wiki_info['title']}\n")
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
        log_file.write(f"Judging passages in parallel (max_concurrent={max_concurrent})...\n")
        stats['total_passages_checked'] += len(passages)
        
        # Track if this entity has at least one relevant passage
        entity_has_relevant_passage = False
        
        # Judge all passages in parallel
        judgment_results = await judge_passages_batch(
            passages=passages,
            entity_label=entity_label,
            property_label=property_label,
            property_description=property_description,
            property_value=property_value,
            property_datatype=property_datatype,
            shared_time=shared_time,
            client=judge_client,
            model=judge_model,
            temperature=judge_temperature,
            max_concurrent=max_concurrent
        )
        
        # Organize results by judgment
        label_to_passage = defaultdict(list)
        for passage, judgment in judgment_results:
            label_to_passage[judgment].append(passage)
            
            log_file.write(f"\nPassage ID: {passage.get('id', 'unknown')}\n")
            log_file.write(f"Content: {passage.get('contents', '')[:200]}...\n")
            log_file.write(f"Judgment: {judgment}\n")
        
        # Update stats
        stats['yes_same_passages'] += len(label_to_passage['YES-SAME'])
        stats['yes_different_passages'] += len(label_to_passage['YES-DIFFERENT'])
        stats['no_related_passages'] += len(label_to_passage['NO-RELATED'])
        stats['no_unrelated_passages'] += len(label_to_passage['NO-UNRELATED'])
        
        # Handle YES-SAME passages (no rewrite needed)
        for passage in label_to_passage['YES-SAME']:
            write_trec_qrel(qrel_output_file, query_id, passage.get('id', 'unknown'), 1)
            qrel_output_file.flush()
            stats['relevant_passages'] += 1
            entity_has_relevant_passage = True
        
        # Handle YES-DIFFERENT passages (rewrite with REPLACE)
        if rewrite_passages:
            # Rewrite YES-DIFFERENT passages in parallel
            rewrite_tasks = []
            semaphore = asyncio.Semaphore(max_concurrent)
            
            for passage in label_to_passage['YES-DIFFERENT']:
                task = rewrite_passage_async(
                    passage=passage,
                    entity_label=entity_label,
                    property_label=property_label,
                    property_description=property_description,
                    property_value=property_value,
                    shared_time=shared_time,
                    rewrite_type="REPLACE",
                    client=rewrite_client,
                    model=rewrite_model,
                    temperature=rewrite_temperature,
                    semaphore=semaphore
                )
                rewrite_tasks.append((passage, task))
            
            if rewrite_tasks:
                rewrite_results = await asyncio.gather(*[task for _, task in rewrite_tasks])
                
                for (passage, _), rewritten_content in zip(rewrite_tasks, rewrite_results):
                    log_file.write(f"\n[REWRITE-REPLACE] Passage ID: {passage.get('id', 'unknown')}\n")
                    log_file.write(f"Original: {passage.get('contents', '')[:200]}...\n")
                    log_file.write(f"Rewritten: {rewritten_content[:200]}...\n")
                    
                    # Save rewrite to file
                    if passage_rewrites_path:
                        rewritten_passage = copy.deepcopy(passage)
                        rewritten_passage['contents'] = rewritten_content
                        rewrite_record = {
                            "query_id": query_id,
                            "passage_id": passage.get('id', 'unknown'),
                            "passage_title": passage.get('title', ''),
                            "entity_label": entity_label,
                            "entity_qid": qid,
                            "property_label": property_label,
                            "property_id": property_id,
                            "property_value": property_value,
                            "shared_time": shared_time,
                            "rewrite_type": "REPLACE",
                            "passage": passage,
                            "rewritten_passage": rewritten_passage
                        }
                        append_rewrite_to_file(passage_rewrites_path, rewrite_record)
                    
                    write_trec_qrel(qrel_output_file, query_id, passage.get('id', 'unknown'), 1)
                    qrel_output_file.flush()
                    stats['relevant_passages'] += 1
                    stats['rewrites_performed'] += 1
                    entity_has_relevant_passage = True
        
        # Handle NO-RELATED passages (if no relevant passages found, rewrite with ADD)
        if rewrite_passages and not entity_has_relevant_passage and len(label_to_passage['NO-RELATED']) > 0:
            # Choose the first NO-RELATED passage
            passage = label_to_passage['NO-RELATED'][0]
            
            rewritten_content = await rewrite_passage_async(
                passage=passage,
                entity_label=entity_label,
                property_label=property_label,
                property_description=property_description,
                property_value=property_value,
                shared_time=shared_time,
                rewrite_type="ADD",
                client=rewrite_client,
                model=rewrite_model,
                temperature=rewrite_temperature
            )
            
            log_file.write(f"\n[REWRITE-ADD] Passage ID: {passage.get('id', 'unknown')}\n")
            log_file.write(f"Original: {passage.get('contents', '')[:200]}...\n")
            log_file.write(f"Rewritten: {rewritten_content[:200]}...\n")
            
            # Save rewrite to file
            if passage_rewrites_path:
                rewritten_passage = copy.deepcopy(passage)
                rewritten_passage['contents'] = rewritten_content
                rewrite_record = {
                    "query_id": query_id,
                    "passage_id": passage.get('id', 'unknown'),
                    "passage_title": passage.get('title', ''),
                    "entity_label": entity_label,
                    "entity_qid": qid,
                    "property_label": property_label,
                    "property_id": property_id,
                    "property_value": property_value,
                    "shared_time": shared_time,
                    "rewrite_type": "ADD",
                    "passage": passage,
                    "rewritten_passage": rewritten_passage
                }
                append_rewrite_to_file(passage_rewrites_path, rewrite_record)
            
            write_trec_qrel(qrel_output_file, query_id, passage.get('id', 'unknown'), 1)
            qrel_output_file.flush()
            stats['relevant_passages'] += 1
            stats['rewrites_performed'] += 1
            entity_has_relevant_passage = True
        
        # Update entity coverage count
        if entity_has_relevant_passage:
            stats['entities_with_relevant_passages'] += 1
        
        log_file.write(f"\nEntity summary: {entity_label}\n")
        log_file.write(f"  YES-SAME: {len(label_to_passage['YES-SAME'])}\n")
        log_file.write(f"  YES-DIFFERENT: {len(label_to_passage['YES-DIFFERENT'])}\n")
        log_file.write(f"  NO-RELATED: {len(label_to_passage['NO-RELATED'])}\n")
        log_file.write(f"  NO-UNRELATED: {len(label_to_passage['NO-UNRELATED'])}\n")
    
    return stats


def print_summary_stats(
    all_stats: List[Dict],
    total_queries: int,
    output_file: Optional[TextIO] = None
):
    """Print summary statistics to console and optionally to a file."""
    total_qids = sum(s['total_qids'] for s in all_stats)
    total_with_wiki = sum(s['qids_with_wikipedia'] for s in all_stats)
    total_passages = sum(s['total_passages_checked'] for s in all_stats)
    total_yes_same = sum(s['yes_same_passages'] for s in all_stats)
    total_yes_different = sum(s['yes_different_passages'] for s in all_stats)
    total_no_related = sum(s['no_related_passages'] for s in all_stats)
    total_no_unrelated = sum(s['no_unrelated_passages'] for s in all_stats)
    total_relevant = sum(s['relevant_passages'] for s in all_stats)
    total_rewrites = sum(s['rewrites_performed'] for s in all_stats)
    total_entities_covered = sum(s['entities_with_relevant_passages'] for s in all_stats)
    
    entity_coverage_pct = (total_entities_covered / total_with_wiki * 100) if total_with_wiki > 0 else 0
    
    stats_lines = [
        f"Total queries processed: {total_queries}",
        f"Total entities (QIDs): {total_qids}",
        f"Entities with Wikipedia pages: {total_with_wiki}",
        f"Total passages checked: {total_passages}",
        f"",
        "LLM Judge Classification:",
        f"  YES-SAME passages: {total_yes_same}",
        f"  YES-DIFFERENT passages: {total_yes_different}",
        f"  NO-RELATED passages: {total_no_related}",
        f"  NO-UNRELATED passages: {total_no_unrelated}",
        f"",
        f"Relevant passages found (in qrels): {total_relevant}",
        f"Passage rewrites performed: {total_rewrites}",
        f"Entities with at least one relevant passage: {total_entities_covered}",
        f"Entity coverage: {entity_coverage_pct:.2f}%"
    ]
    
    if output_file:
        for line in stats_lines:
            output_file.write(line + "\n")
    
    for line in stats_lines:
        print(line)


def main(args):
    """Main function to generate qrels using extract_qrels approach."""
    
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
        # In-memory mode
        print(f"Corpus loading mode: IN-MEMORY")
        
        # Check if index cache exists
        cache_path = get_index_cache_path(args.corpus_jsonl, args.index_cache_dir)
        index_exists = cache_path.exists()
        
        # Load corpus into memory (will use cache if available)
        corpus_dict = load_corpus_in_memory(args.corpus_jsonl, args.index_cache_dir)
        
        # Build page index from corpus_dict
        page_index = build_page_to_passages_index(corpus_dict, args.corpus_jsonl, args.index_cache_dir)
        
        # Save index if it was just built (not loaded from cache)
        if not index_exists:
            save_corpus_index(corpus_dict, page_index, cache_path)
        
        print(f"Memory usage: ~{len(corpus_dict)} passages loaded")
    else:
        # Streaming mode
        print(f"Corpus loading mode: STREAMING")
        
        # Optionally load page2passage mapping
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
    
    # Initialize OpenAI clients
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Use async client if parallel processing is enabled
    if args.use_parallel:
        judge_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        rewrite_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        print(f"Using PARALLEL processing mode with max_concurrent={args.max_concurrent}")
    else:
        judge_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        rewrite_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
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
            log_file.write(f"Judge model: {args.judge_model}\n")
            log_file.write(f"Judge temperature: {args.judge_temperature}\n")
            log_file.write(f"Rewrite model: {args.rewrite_model}\n")
            log_file.write(f"Rewrite temperature: {args.rewrite_temperature}\n")
            log_file.write(f"Rewrite passages: {args.rewrite_passages}\n")
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
                for query_obj in tqdm(queries, desc="Processing queries"):
                    stats = await process_query_async(
                        query_obj=query_obj,
                        corpus_jsonl_path=args.corpus_jsonl,
                        judge_client=judge_client,
                        rewrite_client=rewrite_client,
                        qrel_output_file=qrel_file,
                        log_file=log_file,
                        judge_model=args.judge_model,
                        judge_temperature=args.judge_temperature,
                        rewrite_model=args.rewrite_model,
                        rewrite_temperature=args.rewrite_temperature,
                        page2passage_mapping=page2passage_mapping,
                        qid2wikipedia_cache=qid2wikipedia_cache,
                        corpus_dict=corpus_dict,
                        page_index=page_index,
                        rewrite_passages=args.rewrite_passages,
                        max_concurrent=args.max_concurrent,
                        passage_rewrites_path=args.passage_rewrites_path
                    )
                    stats_list.append(stats)
                return stats_list
            
            all_stats = asyncio.run(process_all_queries())
        else:
            # Use sequential processing (original behavior)
            for query_obj in tqdm(queries, desc="Processing queries"):
                stats = process_query(
                    query_obj=query_obj,
                    corpus_jsonl_path=args.corpus_jsonl,
                    judge_client=judge_client,
                    rewrite_client=rewrite_client,
                    qrel_output_file=qrel_file,
                    log_file=log_file,
                    judge_model=args.judge_model,
                    judge_temperature=args.judge_temperature,
                    rewrite_model=args.rewrite_model,
                    rewrite_temperature=args.rewrite_temperature,
                    page2passage_mapping=page2passage_mapping,
                    qid2wikipedia_cache=qid2wikipedia_cache,
                    corpus_dict=corpus_dict,
                    page_index=page_index,
                    rewrite_passages=args.rewrite_passages,
                    passage_rewrites_path=args.passage_rewrites_path
                )
                all_stats.append(stats)
        
        # Write summary
        log_file.write(f"\n\n{'='*80}\n")
        log_file.write("SUMMARY\n")
        log_file.write(f"{'='*80}\n")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        print_summary_stats(all_stats, len(queries), log_file)
        
        print(f"\nQRels written to: {args.output_qrel}")
        print(f"Passage rewrites written to: {args.passage_rewrites_path}")
        print(f"Log written to: {args.log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TREC-format qrels using extract_qrels approach (4-way classification + rewriting)")
    parser.add_argument("--dataset", type=str, default="quest", choices=["quest", "qald10"], help="Dataset name")
    parser.add_argument("--subset", type=str, default=None, help="Subset name (e.g., 'test_quest', 'train_quest')")
    parser.add_argument("--corpus_jsonl", type=str, default="corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl", help="Path to corpus JSONL file")
    parser.add_argument("--log_dir", type=str, default="qrel_logging", help="Directory to write log file")
    parser.add_argument("--passage_rewrites_path", type=str, default=None, help="Path to save passage rewrites (JSONL format). If not provided, rewrites will only be logged.")
    parser.add_argument("--page2passage_mapping", type=str, default="corpus_datasets/corpus/enwiki_20251001_infoboxconv.index.json", help="Path to JSON file mapping page IDs to passage indices (used in streaming mode)")
    parser.add_argument("--index_cache_dir", type=str, default="corpus_datasets/corpus", help="Directory to store/load cached index (used in memory mode)")
    parser.add_argument("--load_corpus_mode", type=str, default="memory", choices=["stream", "memory"], help="Corpus loading mode: 'stream' (low memory, slower) or 'memory' (high memory, faster)")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini", help="Model to use for relevance judgment")
    parser.add_argument("--judge_temperature", type=float, default=0.0, help="Temperature for judge model")
    parser.add_argument("--rewrite_model", type=str, default="gpt-4o-mini", help="Model to use for rewriting")
    parser.add_argument("--rewrite_temperature", type=float, default=0.7, help="Temperature for rewrite model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to process")
    parser.add_argument("--rewrite_passages", action="store_true", default=True, help="Enable passage rewriting (default: True)")
    parser.add_argument("--no_rewrite", dest='rewrite_passages', action="store_false", help="Disable passage rewriting")
    parser.add_argument("--use_parallel", action="store_true", help="Enable parallel LLM API calls for faster processing")
    parser.add_argument("--max_concurrent", type=int, default=10, help="Maximum number of concurrent LLM API calls when --use_parallel is enabled (default: 10)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing qrel file by skipping already-completed queries")
    parser.add_argument("--reprocess_last", action="store_true", help="When resuming, always re-process the last query (safest option)")
    parser.add_argument("--trust_last", action="store_true", help="When resuming, trust that the last query is complete (fastest option)")
    parser.add_argument("--min_passages_threshold", type=int, default=5, help="Minimum passages to consider a query complete when resuming (default: 5)")
    
    args = parser.parse_args()
    
    # Validate conflicting arguments
    if args.reprocess_last and args.trust_last:
        parser.error("--reprocess_last and --trust_last are mutually exclusive. Choose one or neither (to use heuristic).")
    
    # Determine subset and subdirectory
    if args.subset:
        if "_" in args.subset:
            subset_name = args.subset
            subdir = args.subset.split("_")[0]
        else:
            subdir = args.subset
            subset_name = f"{args.subset}_{args.dataset}"
    else:
        if args.dataset == "quest":
            subset_name = "test_quest"
            subdir = "test"
        else:
            subset_name = args.dataset
            subdir = ""
    
    # Construct file paths
    if subdir:
        args.query_file = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/{subdir}/{subset_name}_generations.jsonl"
        args.output_qrel = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/{subdir}/qrels_{subset_name}.txt"
        if args.passage_rewrites_path is None:
            args.passage_rewrites_path = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/{subdir}/passage_rewrites_{subset_name}.jsonl"
    else:
        args.query_file = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/{subset_name}_generations.jsonl"
        args.output_qrel = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/qrels_{subset_name}.txt"
        if args.passage_rewrites_path is None:
            args.passage_rewrites_path = f"corpus_datasets/dataset_creation_heydar/{args.dataset}/passage_rewrites_{subset_name}.jsonl"
    
    # Create log file path
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
    print(f"Passage rewrites: {args.passage_rewrites_path}")
    print(f"Log file: {args.log_file}")
    print(f"Corpus: {args.corpus_jsonl}")
    print(f"Corpus loading mode: {args.load_corpus_mode.upper()}")
    if args.load_corpus_mode == "stream":
        print(f"Page2passage mapping: {args.page2passage_mapping}")
    else:
        print(f"Index cache dir: {args.index_cache_dir}")
    print(f"Parallel processing: {'ENABLED' if args.use_parallel else 'DISABLED'}")
    if args.use_parallel:
        print(f"Max concurrent LLM calls: {args.max_concurrent}")
    print(f"Judge model: {args.judge_model} (temp={args.judge_temperature})")
    print(f"Rewrite model: {args.rewrite_model} (temp={args.rewrite_temperature})")
    print(f"Rewrite passages: {args.rewrite_passages}")
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
    
    main(args)
    
    # Calculate and display entity coverage after qrel generation completes
    print("\n" + "="*80)
    print("Calculating entity coverage...")
    print("="*80 + "\n")
    
    coverage_results = calculate_coverage(
        dataset=args.dataset,
        qrel_file_path=args.output_qrel,
        subset=args.subset
    )


# ============================================================================
# Usage Examples
# ============================================================================
#
# 1. PARALLEL + IN-MEMORY MODE (RECOMMENDED - Fastest, 5-10x speedup):
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test \
#        --use_parallel --load_corpus_mode memory --max_concurrent 20
#
# 2. PARALLEL + STREAMING MODE (Good performance, lower memory):
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test \
#        --use_parallel --load_corpus_mode stream --max_concurrent 20
#
# 3. SEQUENTIAL + IN-MEMORY MODE (Default - No parallel processing):
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test
#
# 4. SEQUENTIAL + STREAMING MODE (Lowest memory usage, slowest):
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test --load_corpus_mode stream
#
# 5. With specific models:
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test --use_parallel \
#        --judge_model gpt-4o-mini --rewrite_model gpt-4o-mini
#
# 6. Limit queries for testing:
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset val --use_parallel \
#        --load_corpus_mode memory --max_concurrent 20 --limit 10
#
# 7. Disable rewriting (judge only):
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test --no_rewrite
#
# 8. Resume mode (continue from interruption):
#    Default resume (uses heuristic to detect incomplete last query):
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test --use_parallel --resume
#
#    Safe resume (always re-processes last query):
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test --use_parallel --resume --reprocess_last
#
#    Fast resume (trusts last query is complete):
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test --use_parallel --resume --trust_last
#
#    Custom threshold (re-process if last query has < N passages):
#    python c3_qrel_generation/qrel_generation.py \
#        --dataset quest --subset test --use_parallel --resume --min_passages_threshold 10
#
# Performance Notes:
# - PARALLEL mode: 5-10x faster than sequential, use for production runs
# - max_concurrent: Higher values (20-30) = faster but more API load
# - MEMORY mode: Best for processing many queries (100s+), requires sufficient RAM
# - STREAM mode: Better for single/few queries or limited RAM environments
# - First run in MEMORY mode will cache the index for faster subsequent runs
# - First run in STREAM mode will build page2passage mapping if not provided
#
# Resume Mode Strategies:
# - Default (heuristic): Re-processes last query if it has < 5 passages (balanced)
# - --reprocess_last: Always re-processes last query (safest, may waste some LLM calls)
# - --trust_last: Never re-processes last query (fastest, risky if interrupted mid-query)
# - --min_passages_threshold N: Custom threshold for heuristic (flexible)
# - Resume cleans up both qrel file AND rewrites file for re-processed queries
#
# ============================================================================
