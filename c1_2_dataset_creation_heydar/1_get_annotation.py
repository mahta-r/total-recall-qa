#!/usr/bin/env python3
"""
Unified Dataset Annotation Processing

This script processes different datasets (QALD10, Quest) to generate annotations with QIDs and properties.
The dataset type is specified as an input parameter, and the appropriate annotation function is called.

Supported datasets:
- qald10: QALD10 dataset with SPARQL queries
- quest: Quest dataset with Wikipedia document titles

Usage:
    python c1_2_dataset_creation_heydar/unified/1_get_annotation.py --dataset qald10 --model openai/gpt-4o
    python c1_2_dataset_creation_heydar/unified/1_get_annotation.py --dataset quest --input test.jsonl
"""

import os
import sys
import json
import argparse
import requests
import time
import random
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Any
from collections import Counter
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from importlib import import_module
    get_properties_module = import_module('2_get_properties')
    get_all_aggregatable_properties = get_properties_module.get_all_aggregatable_properties
    NO_AGGREGATION_PROPS = get_properties_module.NO_AGGREGATION_PROPS
    INTERNAL_WIKI_PROPS = get_properties_module.INTERNAL_WIKI_PROPS
    initialize_internal_props = get_properties_module.initialize_internal_props
except ImportError:
    print("Warning: Could not import 2_get_properties module. Will try alternative import.")
    try:
        get_properties_module = import_module('2_get_properties')
        get_all_aggregatable_properties = get_properties_module.get_all_aggregatable_properties
        NO_AGGREGATION_PROPS = get_properties_module.NO_AGGREGATION_PROPS
        INTERNAL_WIKI_PROPS = get_properties_module.INTERNAL_WIKI_PROPS
        initialize_internal_props = get_properties_module.initialize_internal_props
    except ImportError:
        print("Warning: Could not import 2_get_properties module.")

try:
    from c3_task_evaluation.src.prompt_templetes import SYSTEM_PROMPT_SPARQL_LIST
except ImportError:
    SYSTEM_PROMPT_SPARQL_LIST = ""  # Will be loaded if needed


# ========================================================================
# Common utility functions (shared by both datasets)
# ========================================================================

def get_qid_from_wikipedia_title(title: str, lang: str = "en") -> Optional[str]:
    """
    Convert a Wikipedia title to a Wikidata QID.

    Args:
        title: Wikipedia article title
        lang: Wikipedia language code (default: en)

    Returns:
        Wikidata QID (e.g., "Q42") or None if not found
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "format": "json",
    }
    headers = {"User-Agent": "Quest-Pipeline/1.0 (research; heydar.soudani@ru.nl)"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id != "-1":  # -1 means page not found
                pageprops = page_data.get("pageprops", {})
                qid = pageprops.get("wikibase_item")
                if qid:
                    return qid

        return None
    except Exception as e:
        print(f"Error fetching QID for '{title}': {e}")
        return None


def get_qids_batch(titles: List[str], lang: str = "en", batch_size: int = 50) -> Dict[str, Optional[str]]:
    """
    Convert multiple Wikipedia titles to Wikidata QIDs in batches.

    Args:
        titles: List of Wikipedia article titles
        lang: Wikipedia language code (default: en)
        batch_size: Number of titles to process per API call

    Returns:
        Dictionary mapping title -> QID (or None if not found)
    """
    results = {}

    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        titles_str = "|".join(batch)

        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": titles_str,
            "prop": "pageprops",
            "format": "json",
        }
        headers = {"User-Agent": "Quest-Pipeline/1.0 (research; heydar.soudani@ru.nl)"}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                title = page_data.get("title")
                if title:
                    pageprops = page_data.get("pageprops", {})
                    qid = pageprops.get("wikibase_item")
                    results[title] = qid

            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"Error fetching QIDs for batch: {e}")
            # Fallback to individual queries for this batch
            for title in batch:
                results[title] = get_qid_from_wikipedia_title(title, lang)

    return results


def get_wikipedia_title_from_qid(qid: str, lang: str = "en") -> Optional[str]:
    """
    Convert a Wikidata QID to Wikipedia title.

    Args:
        qid: Wikidata QID (e.g., "Q42")
        lang: Wikipedia language code (default: en)

    Returns:
        Wikipedia article title or None if not found
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "format": "json",
        "props": "sitelinks",
        "sitefilter": f"{lang}wiki",
    }
    headers = {"User-Agent": "HeydarSoudani-ResearchBot/1.0 (https://example.com; heydar.soudani@ru.nl)"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        entity = data.get("entities", {}).get(qid, {})
        sitelinks = entity.get("sitelinks", {})
        wiki_key = f"{lang}wiki"

        if wiki_key in sitelinks:
            return sitelinks[wiki_key]["title"]
        else:
            return None

    except Exception as e:
        print(f"Error fetching {qid}: {e}")
        return None


# ========================================================================
# QALD10-specific annotation functions
# ========================================================================

def extract_qid(text: str) -> Optional[str]:
    """Extract QID from Wikidata entity URL."""
    match = re.search(r'entity/(Q\d+)', text)
    if match:
        return match.group(1)
    return None


def get_most_frequent_instance_of(results_by_item: Dict[str, List[Dict]]) -> Optional[Dict]:
    """
    Get the most frequent instance type across all items.

    Args:
        results_by_item: Dictionary mapping QID -> list of instance types

    Returns:
        Dictionary with most common instance info
    """
    counter = Counter()
    id_to_label = {}

    for qid, instances in results_by_item.items():
        for inst in instances:
            inst_id = inst["id"]
            counter[inst_id] += 1
            # store label (first one is fine — labels are consistent on Wikidata)
            if inst_id not in id_to_label:
                id_to_label[inst_id] = inst["label"]

    # Get the ID with the highest frequency
    if not counter:
        return None

    most_common_id, freq = counter.most_common(1)[0]
    return {
        "id": most_common_id,
        "label": id_to_label[most_common_id],
        "count": freq
    }


def get_instances_of(qids: List[str], batch_size: int = 50) -> Optional[Dict]:
    """
    Get the most common instance type for a list of QIDs.

    Args:
        qids: List of Wikidata QIDs
        batch_size: Batch size for SPARQL queries

    Returns:
        Dictionary with most common instance info
    """
    WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

    clean_qids = [qid for qid in qids if isinstance(qid, str) and re.fullmatch(r"Q\d+", qid)]
    if not clean_qids:
        return {}

    # Will accumulate instance info for ALL QIDs across batches
    all_results_by_item = {qid: [] for qid in clean_qids}

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "MyResearchBot/1.0 (your_email@example.com)",
    }

    # Process in batches to keep each query reasonable
    for start in range(0, len(clean_qids), batch_size):
        batch = clean_qids[start:start + batch_size]
        values_block = " ".join(f"wd:{qid}" for qid in batch)

        query = f"""
        SELECT ?item ?itemLabel ?instance ?instanceLabel WHERE {{
          VALUES ?item {{ {values_block} }}
          ?item wdt:P31 ?instance .
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .
          }}
        }}
        """

        response = requests.post(
            WIKIDATA_SPARQL_URL,
            data={"query": query},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Collect results for this batch
        for row in data["results"]["bindings"]:
            item_uri = row["item"]["value"]          # e.g. "http://www.wikidata.org/entity/Q42"
            instance_uri = row["instance"]["value"]  # e.g. "http://www.wikidata.org/entity/Q5"
            item_qid = item_uri.rsplit("/", 1)[-1]
            instance_qid = instance_uri.rsplit("/", 1)[-1]

            instance_label = row.get("instanceLabel", {}).get("value", "")

            if item_qid in all_results_by_item:
                entry = {"id": instance_qid, "label": instance_label}
                if entry not in all_results_by_item[item_qid]:
                    all_results_by_item[item_qid].append(entry)

    most_frequent_instance = get_most_frequent_instance_of(all_results_by_item)
    return most_frequent_instance


def process_qald10_annotations(input_file: str, output_file: str, entity_type_output_file: str = None,
                                model_name: str = "openai/gpt-4o", retries: int = 2,
                                timeout_seconds: int = 30) -> int:
    """
    Process QALD10 dataset to generate annotations.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        entity_type_output_file: Optional path to entity type mapping file
        model_name: Model name for LLM calls
        retries: Number of retries for SPARQL queries
        timeout_seconds: Timeout for SPARQL queries

    Returns:
        0 on success, 1 on error
    """
    WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

    print("=" * 70)
    print("QALD10 Dataset Annotation Processing")
    print("=" * 70)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()

    # Check if input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Initialize OpenAI client and session
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"))
    user_prompt_template = "Here is my input query:\n{sparql_query}"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "heyda-sparql-runner/1.0 (research; contact: youremail@example.com)",
        "Accept": "application/sparql-results+json"
    })

    def choose_endpoint(sparql_text: str) -> str:
        text = (sparql_text or "").lower()
        if "wikidata.org" in text or "wd:" in text or "wdt:" in text:
            return "https://query.wikidata.org/sparql"
        if "dbpedia.org" in text or "dbo:" in text or "dbr:" in text:
            return "https://dbpedia.org/sparql"
        return "https://dbpedia.org/sparql"

    def run_query(ep: str, sparql_text: str) -> Dict[str, Any]:
        params = {"query": sparql_text}
        last_error = None
        for attempt in range(retries + 1):
            try:
                resp = session.get(ep, params=params, timeout=timeout_seconds)
                if resp.status_code == 200:
                    return resp.json()
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                last_error = str(e)
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
        raise RuntimeError(f"SPARQL request failed after {retries+1} attempts: {last_error}")

    def format_results(json_obj: Dict[str, Any]) -> List[Dict[str, str]]:
        if "boolean" in json_obj:
            return [{"ASK": "true" if json_obj["boolean"] else "false"}]

        head = json_obj.get("head", {})
        vars_ = head.get("vars", []) or []
        bindings = json_obj.get("results", {}).get("bindings", []) or []

        rows = []
        for b in bindings:
            row = {}
            for v in vars_:
                cell = b.get(v, {})
                row[v] = str(cell.get("value", ""))
            rows.append(row)
        return rows

    def extract_converted_query(response_text: str) -> str:
        # Try code fence
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, flags=re.S)
        if fence_match:
            payload = fence_match.group(1)
            data = json.loads(payload)
            if "converted_query" in data:
                return data["converted_query"].strip()

        # Try JSON object
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = response_text[start:end+1]
            try:
                data = json.loads(candidate)
                if "converted_query" in data:
                    return data["converted_query"].strip()
            except json.JSONDecodeError:
                pass

        # Try regex
        m = re.search(r'"converted_query"\s*:\s*"((?:\\.|[^"\\])*)"', response_text, flags=re.S)
        if m:
            return bytes(m.group(1), "utf-8").decode("unicode_escape").strip()

        raise ValueError("Could not find a valid 'converted_query' in the response.")

    # Main loop
    instances_of_index = {}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(output_file, "w", encoding="utf-8") as out_f:

        for idx, line in enumerate(tqdm(in_f)):
            if not line.strip():
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSONL line: {line[:120]}...")
                continue

            file_id = rec.get("file_id")
            qid = rec.get("qid")
            query = rec.get("query")
            sparql_text = rec.get("sparql") or ""
            answer_value_ = rec.get("answers", {}).get('value', [])
            answer_value = answer_value_[0] if len(answer_value_) > 0 else []

            if not sparql_text.strip():
                continue

            # Get the updated answer
            ep = choose_endpoint(sparql_text)
            try:
                rows = format_results(run_query(ep, sparql_text))
                if not rows:
                    updated_answer = ""
                else:
                    _, updated_answer = next(iter(rows[0].items()))
            except Exception as e:
                print(f"file_id: {file_id} | qid: {qid}")
                print("ERROR:", e)
                updated_answer = ""

            # Get main entity & property
            main_qids = sorted(set(re.findall(r'\bwd:(Q[1-9]\d*)\b', sparql_text)))
            main_properties = sorted(set(re.findall(r'\bP[1-9]\d*\b', sparql_text)))

            # Get the new sparql for intermediate answers
            parts = re.split(r'(XMLSchema#>)', sparql_text, maxsplit=1)
            if len(parts) > 1:
                prefixes = parts[0] + parts[1]
                sparql_query_body = parts[2].strip()
            else:
                print("Expected exactly one 'XMLSchema#>' split point in the SPARQL string.")
                continue

            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_SPARQL_LIST},
                    {"role": "user", "content": user_prompt_template.format(sparql_query=sparql_query_body)}
                ]
            )
            sparql_query_list = extract_converted_query(completion.choices[0].message.content).replace('\n', '')
            new_sparql_query = prefixes + sparql_query_list

            # Get the intermediate answers
            intermidate_list = []
            ep = choose_endpoint(new_sparql_query)
            try:
                result_json = run_query(ep, new_sparql_query)
                rows = format_results(result_json)
                if rows:
                    for i, row in enumerate(rows):
                        _, first_value = next(iter(row.items()))
                        intermidate_list.append(extract_qid(first_value))
            except Exception as e:
                print(f"file_id: {file_id} | qid: {qid}")
                print("ERROR:", e)

            # Get the instance of intermediate answers
            intermidate_list_instances_of = get_instances_of(intermidate_list)

            # Update global index of unique instances_of
            if intermidate_list_instances_of and intermidate_list_instances_of['label'] not in instances_of_index:
                instances_of_index[intermidate_list_instances_of['label']] = {"wikidata_id": intermidate_list_instances_of['id']}

            # Edit on final answer
            updated_answer_ = "yes" if updated_answer == "true" else "no" if updated_answer == "false" else updated_answer
            match = re.match(r"^https?://www\.wikidata\.org/entity/(Q\d+)$", updated_answer)
            if match:
                updated_answer_ = get_wikipedia_title_from_qid(match.group(1))

            # Write in file - unified format
            item = {
                "qid": qid,
                "query": query,
                "intermediate_qids": intermidate_list,
                "answer": updated_answer_,
                "extra": {
                    "file_id": file_id,
                    "dataset_answer": answer_value,
                    "is_changed": not (str(answer_value) == str(updated_answer)),
                    "main_entities": main_qids,
                    "main_properties": main_properties,
                    "intermediate_qids_instances_of": intermidate_list_instances_of
                }
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write entity type mapping file
    if entity_type_output_file:
        os.makedirs(os.path.dirname(entity_type_output_file), exist_ok=True)
        with open(entity_type_output_file, "w", encoding="utf-8") as f_types:
            json.dump(instances_of_index, f_types, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("QALD10 ANNOTATION PROCESSING COMPLETED")
    print("=" * 70)
    print(f"Output: {output_file}")

    return 0


# ========================================================================
# Quest-specific annotation functions
# ========================================================================

def process_quest_entry(entry: Dict, entry_idx: int) -> Optional[Dict]:
    """
    Process a single Quest dataset entry.

    Args:
        entry: Quest dataset entry with 'docs' field
        entry_idx: Index of the entry (for generating QID)

    Returns:
        Processed entry with QIDs in unified format, or None if processing failed
    """
    docs = entry.get("docs", [])
    if not docs:
        return None

    # Step 1: Map document titles to QIDs (these are intermediate_qids)
    title_to_qid = get_qids_batch(docs)

    # Check if ALL documents have QIDs - skip if any document is missing a QID
    missing_qids = [title for title, qid in title_to_qid.items() if qid is None]
    if missing_qids:
        print(f"\nSkipping query (missing QIDs for {len(missing_qids)} docs): {entry.get('query', 'N/A')}")
        print(f"  Missing QIDs for: {missing_qids[:3]}" + (" ..." if len(missing_qids) > 3 else ""))
        return None

    intermediate_qids = [qid for qid in title_to_qid.values() if qid is not None]

    if len(intermediate_qids) < 2:  # Need at least 2 entities for meaningful properties
        return None

    # Step 2: Get instances_of for intermediate QIDs
    intermediate_qids_instances_of = get_instances_of(intermediate_qids)

    # Step 3: Generate a unique QID for this Quest entry (quest_<index>)
    qid = f"quest_{entry_idx}"

    # Step 4: Create result entry in unified format
    result = {
        "qid": qid,
        "query": entry.get("query", ""),
        "intermediate_qids": intermediate_qids,
        "answer": None,  # Quest doesn't have answers
        "extra": {
            "docs": docs,
            "intermediate_qids_instances_of": intermediate_qids_instances_of
        }
    }

    return result


def process_quest_annotations(input_file: str, output_file: str, subsample: float = 200,
                               limit: int = None) -> int:
    """
    Process the Quest dataset and generate annotations.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        subsample: Number of samples to process (-1 for all, 0-1 for percentage, >1 for absolute number)
        limit: Optional limit to override subsample

    Returns:
        0 on success, 1 on error
    """
    print("=" * 70)
    print("Quest Dataset Annotation Processing")
    print("=" * 70)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return 1

    # Load all dataset entries
    print("Loading dataset...")
    all_entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    all_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total_lines = len(all_entries)
    print(f"Loaded {total_lines} entries from dataset")

    # Calculate number of samples to process based on subsample parameter
    if subsample == -1:
        num_samples = total_lines
        print(f"Subsample: Processing ALL {total_lines} samples")
    elif 0 < subsample < 1:
        num_samples = int(total_lines * subsample)
        print(f"Subsample: Processing {subsample*100:.1f}% = {num_samples} samples (out of {total_lines})")
    else:
        num_samples = int(subsample)
        print(f"Subsample: Processing {num_samples} samples (out of {total_lines})")

    # Override with --limit if specified
    if limit:
        num_samples = min(num_samples, limit)
        print(f"Limit override: Processing {num_samples} samples")

    # Randomly subsample entries
    if num_samples < total_lines:
        print(f"Randomly subsampling {num_samples} entries...")
        sampled_entries = random.sample(all_entries, num_samples)
    else:
        sampled_entries = all_entries

    print()

    # Process Quest entries
    print("=" * 70)
    print("Processing Quest Entries")
    print("=" * 70)

    processed_count = 0
    skipped_count = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, entry in enumerate(tqdm(sampled_entries, desc="Processing entries")):
            # Process the entry with index for QID generation
            result = process_quest_entry(entry, idx)

            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed_count += 1
            else:
                skipped_count += 1

    # Summary
    print()
    print("=" * 70)
    print("QUEST ANNOTATION PROCESSING COMPLETED")
    print("=" * 70)
    print(f"Total lines in dataset: {total_lines}")
    print(f"Attempted to process: {num_samples} entries")
    print(f"Successfully processed: {processed_count} entries")
    print(f"Skipped: {skipped_count} entries")
    print(f"Output: {output_file}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset to generate annotations with QIDs and properties')
    parser.add_argument('--dataset', type=str, required=True, choices=['qald10', 'quest'],
                        help='Dataset type to process (qald10 or quest)')
    parser.add_argument('--model', type=str, default='openai/gpt-4o',
                        help='Model to use for LLM steps (QALD10 only, default: openai/gpt-4o)')

    # QALD10-specific arguments
    parser.add_argument('--qald10_input', type=str,
                        default='corpus_datasets/qald_aggregation_samples/wikidata_aggregation.jsonl',
                        help='Input file for QALD10 dataset')
    parser.add_argument('--qald10_output', type=str,
                        default='corpus_datasets/dataset_creation_heydar/unified/qald10_annotations.jsonl',
                        help='Output file for QALD10 annotations')
    parser.add_argument('--qald10_entity_types', type=str,
                        default='corpus_datasets/dataset_creation_heydar/unified/qald10_entity_types.json',
                        help='Entity type mapping file for QALD10')

    # Quest-specific arguments
    parser.add_argument('--quest_input', type=str,
                        help='Input file for Quest dataset (e.g., train.jsonl)')
    parser.add_argument('--quest_output', type=str,
                        help='Output file for Quest annotations (auto-generated if not specified)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of entries to process (Quest only, for testing)')
    parser.add_argument('--subsample', type=float, default=200,
                        help='Number of samples to process (Quest only): -1 for all, 0-1 for percentage, >1 for absolute number (default: 200)')

    args = parser.parse_args()

    # Route to appropriate dataset handler
    if args.dataset == 'qald10':
        sys.exit(process_qald10_annotations(
            input_file=args.qald10_input,
            output_file=args.qald10_output,
            entity_type_output_file=args.qald10_entity_types,
            model_name=args.model
        ))
    elif args.dataset == 'quest':
        # Setup Quest paths
        base_dir = Path(__file__).parent.parent.parent

        if not args.quest_input:
            print("Error: --quest_input is required for Quest dataset")
            sys.exit(1)

        input_file = base_dir / "corpus_datasets" / "quest_dataset" / args.quest_input

        if args.quest_output:
            output_file = Path(args.quest_output)
        else:
            output_dir = base_dir / "corpus_datasets" / "dataset_creation_heydar" / "unified"
            output_file = output_dir / f"{Path(args.quest_input).stem}_quest_annotations.jsonl"

        sys.exit(process_quest_annotations(
            input_file=str(input_file),
            output_file=str(output_file),
            subsample=args.subsample,
            limit=args.limit
        ))
