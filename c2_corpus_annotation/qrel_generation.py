import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import time
import json
import requests
import argparse
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


from src.prompt_templetes import SUBQUERY_GENERATOR_PROMPT, RELEVANCE_JUDGMENT_PROMPT
from utils.general_utils import set_seed

os.environ["OPENAI_API_KEY"] = ''
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
WIKIPEDIA_API_URL   = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "YourAppName/1.0 (contact@example.com)"


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def get_wikipedia_from_wikidata(qid: str, lang: str = "en", timeout: float = 8.0):
    # Basic sanity check on QID format
    if not re.fullmatch(r"Q\d+", qid or ""):
        return {"wikidata_id": qid, "error": "Invalid QID format (expected like 'Q42')"}

    s = _session()

    # --- Fetch from Wikidata ---
    url = WIKIDATA_ENTITY_URL.format(qid=qid)
    try:
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            # Not JSON; capture a short preview for debugging
            preview = r.text[:200].replace("\n", " ")
            return {"wikidata_id": qid, "error": f"Wikidata returned non-JSON: {preview} ..."}
    except requests.RequestException as e:
        return {"wikidata_id": qid, "error": f"Wikidata request failed: {e}"}

    entity = data.get("entities", {}).get(qid)
    if not entity:
        return {"wikidata_id": qid, "error": "QID not found in Wikidata response"}

    sitelinks = entity.get("sitelinks", {})
    wiki_key = f"{lang}wiki"
    if wiki_key not in sitelinks:
        return {"wikidata_id": qid, "error": f"No {lang} Wikipedia page found"}

    title = sitelinks[wiki_key].get("title")
    if not title:
        return {"wikidata_id": qid, "error": "Wikipedia title missing in sitelink"}

    # --- Resolve page on Wikipedia ---
    wp_url = WIKIPEDIA_API_URL.format(lang=lang)
    params = {
        "action": "query",
        "titles": title,
        "prop": "info",
        "inprop": "pageid",
        "redirects": "1",   # follow redirects
        "format": "json",
        "formatversion": "2"
    }
    try:
        wp_r = s.get(wp_url, params=params, timeout=timeout)
        wp_r.raise_for_status()
        try:
            wp_data = wp_r.json()
        except ValueError:
            preview = wp_r.text[:200].replace("\n", " ")
            return {"wikidata_id": qid, "error": f"Wikipedia returned non-JSON: {preview} ..."}
    except requests.RequestException as e:
        return {"wikidata_id": qid, "error": f"Wikipedia request failed: {e}"}

    pages = wp_data.get("query", {}).get("pages", [])
    if not pages:
        return {"wikidata_id": qid, "wikipedia_title": title, "error": "No pages in Wikipedia response"}

    page = pages[0]
    if "missing" in page:
        return {"wikidata_id": qid, "wikipedia_title": title, "error": "Wikipedia page missing"}

    return {
        "wikidata_id": qid,
        "wikipedia_title": page.get("title", title),
        "wikipedia_id": page.get("pageid")
    }

def norm_key(x):
    return str(x).strip().replace(" ", "_") if x is not None else ""

def load_corpus_by_prefix(file_path):
    """
    Loads the corpus into memory, grouped by the first part of 'id'.
    Returns a dict: {wiki_id_prefix: [entries]}.
    """
    data = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                entry = json.loads(line)
                wiki_id = entry["id"].split("-", 1)[0]
                data[norm_key(wiki_id)].append(entry)
            except json.JSONDecodeError:
                continue
    return data

def qrel_generation(args):
    
    class SubqueryGeneratorAPI:
        def __init__(self, generation_model, prompt=SUBQUERY_GENERATOR_PROMPT):
            self.generator_model = generation_model
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"))
            self.prompt = prompt
            
        def generate(self, main_query, entity_title, temperature=0.0):
            messages = [{
                "role": "user", "content": self.prompt.format(main_query=main_query, entity=entity_title)
            }]
            completion = self.client.chat.completions.create(
                model=self.generator_model,
                messages=messages,
                # temperature=temperature
            )
            output_text = completion.choices[0].message.content
            
            # -- Get answer
            print(output_text)
            
            return output_text
    
    class RelevanceJudgementAPI:
        def __init__(self, generation_model, prompt=RELEVANCE_JUDGMENT_PROMPT):
            self.generator_model = generation_model
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"))
            self.prompt = prompt
            
        def generate(self, query, passage, temperature=0.0, max_retries=5, retry_delay=0.5):
            for attempt in range(1, max_retries + 1):
                try:
                    messages = [{
                        "role": "user",
                        "content": self.prompt.format(query=query, passage=passage)
                    }]
                    completion = self.client.chat.completions.create(
                        model=self.generator_model,
                        messages=messages,
                        # temperature=temperature,
                    )
                    output_text = completion.choices[0].message.content
                    match = re.search(r'final score:\s*([0-9]+)', output_text, re.IGNORECASE)
                    if match:
                        score = str(match.group(1))
                        return score
                    else:
                        print(f"[Attempt {attempt}] No score found in output: {output_text.replace('\n', ' ')}")
                except Exception as e:
                    print(f"[Attempt {attempt}] Error: {e}")

                if attempt < max_retries:
                    time.sleep(retry_delay)

            print(f"⚠️ Failed to extract score after {max_retries} attempts.")
            return None
    
    # Load existing qids from the qrel file
    existing_qids = set()
    if os.path.exists(args.qrel_file):
        with open(args.qrel_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    qid = parts[0]
                    existing_qids.add(qid)
        print(f"Found {len(existing_qids)} existing qids in qrel file.")
    
    dataset = {}
    if os.path.exists(args.dataset_file):
        with open(args.dataset_file, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                sample = json.loads(line)
                qid = str(sample["qid"])  # adjust if your key differs
                dataset[qid] = sample
    print(f"Total dataset size: {len(dataset)}")
    
    filtered_dataset = {qid: data for qid, data in dataset.items() if qid not in existing_qids}
    print(f"Filtered dataset size (remaining to process): {len(filtered_dataset)}")
    
    subquery_generator_model = SubqueryGeneratorAPI(args.model_name_or_path)
    relevance_judgement_model= RelevanceJudgementAPI(args.model_name_or_path)
    corpus_index = load_corpus_by_prefix(args.corpus_path)
    
    with open(args.qrel_file, "a") as out_f:
        for idx, (qid, sample) in enumerate(filtered_dataset.items()):
            if idx == 1:
                break
            
            if 'qid' in sample:
                qid, main_query = sample['qid'], sample['query']
                main_entities, intermidate_list  = sample['main_entities'], sample['intermidate_list']
                
    
                start_time = time.time()
                print(f"\n=== Sample {qid} is being processed ...")
                main_docs = []
                for main_entity in main_entities:
                    wiki_info = get_wikipedia_from_wikidata(main_entity)
                    wikipedia_id = wiki_info.get('wikipedia_id', '')
                    corpus_entries = corpus_index.get(norm_key(wikipedia_id), [])
                    main_docs.extend(corpus_entries)
                print(f"-- # main docs: {len(main_docs)}")
                
                print(f"-- # Intermidate entities: {len(intermidate_list)}")
                for int_entity_idx, entity_qid in enumerate(intermidate_list):
                    # if int_entity_idx == 2:
                    #     break
                    
                    print(f"- Generating relevance Judgement for {entity_qid} ...")
                    wiki_info = get_wikipedia_from_wikidata(entity_qid)
                    wikipedia_id = wiki_info.get('wikipedia_id', '')
                    entity_title = wiki_info.get('wikipedia_title', '')
                    corpus_entries = corpus_index.get(norm_key(wikipedia_id), [])
                    intermidate_query = subquery_generator_model.generate(main_query, entity_title)
                    
                    for doc_id, doc in enumerate(tqdm(main_docs + corpus_entries)):
                        # if doc_id == 10:
                        #     break
                        docid = doc.get('id', '')
                        passage = doc.get('contents', '')
                        relevance = relevance_judgement_model.generate(intermidate_query, passage)
                        
                        # <query_id> 0 <doc_id> <relevance> <wikidata_id>
                        out_f.write(f"{qid} 0 {docid} {relevance} {entity_qid}\n")

                end_time = time.time()
                elapsed = end_time - start_time
                print(f"=== Sample {qid} has been processed in {elapsed:.2f} seconds!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='openai/gpt-4o', choices=[
        "openai/gpt-4o", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
    ])
    parser.add_argument('--corpus_path', type=str, default='corpus_datasets/enwiki_20251001.jsonl')
    parser.add_argument('--dataset_file', type=str, default="corpus_datasets/qald_aggregation_samples/wikidata_totallist.jsonl")
    parser.add_argument('--qrel_file', type=str, default="corpus_datasets/qald_aggregation_samples/qrels.txt")
    
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    # === Run ========================
    set_seed(args.seed)
    qrel_generation(args)
    
    # python c1_2_corpus_dataset_preparation/qrel_generation.py

