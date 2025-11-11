import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import argparse
import requests
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from string import Template
from typing import Optional, Dict, Any, List

from src.prompt_templetes import SYSTEM_PROMPT_SPARQL_LIST

os.environ["OPENAI_API_KEY"] = ''


def get_annotations(input_file, output_file, endpoint=None, retries: int = 2, timeout_seconds: int = 30):
    
    ### --- Input data -------------------
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    ### --- Initial & Models -------------
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"))
    user_prompt_template = "Here is my input query:\n{sparql_query}"
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "heyda-sparql-runner/1.0 (research; contact: youremail@example.com)",
        "Accept": "application/sparql-results+json"
    })

    ### --- Functions -------------------
    def choose_endpoint(sparql_text: str) -> str:
        if endpoint:
            return endpoint
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
                # Some endpoints use 200 even for errors, try to parse JSON either way
                if resp.status_code == 200:
                    return resp.json()
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                last_error = str(e)
            # brief backoff
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
        raise RuntimeError(f"SPARQL request failed after {retries+1} attempts: {last_error}")

    def format_results(json_obj: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert SPARQL JSON result to a list of simple dict rows (var -> value).
        Works for SELECT. For ASK, returns [{"ASK": "true/false"}].
        """
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
                # Value may be absent; default to empty string
                row[v] = str(cell.get("value", ""))
            rows.append(row)
        return rows

    def extract_converted_query(response_text: str) -> str:
        """
        Extracts {"converted_query": "..."} from a model's text output.
        Handles code fences and stray text. Raises ValueError if not found.
        """
        # 1) If the model wrapped JSON in ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, flags=re.S)
        if fence_match:
            payload = fence_match.group(1)
            data = json.loads(payload)
            if "converted_query" in data:
                return data["converted_query"].strip()

        # 2) Otherwise, try to isolate the first JSON object in the text
        #    (find the first '{' and last '}' and parse that slice).
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

        # 3) As a last resort, regex specifically for the "converted_query" field value
        #    (handles cases where the model printed only that field).
        m = re.search(r'"converted_query"\s*:\s*"((?:\\.|[^"\\])*)"', response_text, flags=re.S)
        if m:
            # Unescape JSON string escapes
            return bytes(m.group(1), "utf-8").decode("unicode_escape").strip()

        raise ValueError("Could not find a valid 'converted_query' in the response.")

    def extract_qid(text):
        match = re.search(r'entity/(Q\d+)', text)
        if match:
            return match.group(1)
        return None

    def get_wikipedia_title_from_qid(qid, lang="en"):
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


    ### -- main loop --------------------    
    with open(input_path, "r", encoding="utf-8") as in_f, open(output_file, "w", encoding="utf-8") as out_f:
        for idx, line in enumerate(tqdm(in_f)):
            # if idx == 6:
            #     break
            
            # --- Read the sample ----------------
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
            answer_value_ = rec.get("answers").get('value')
            answer_value = answer_value_[0] if len(answer_value_) > 0 else []
            
            if not sparql_text.strip():
                continue
            
            # --- Get the updated answer ------------------------------
            ep = choose_endpoint(sparql_text)
            try:
                result_json = run_query(ep, sparql_text)
                rows = format_results(result_json)
                if not rows:
                    print("(no results)")
                else:
                    _, updated_answer = next(iter(rows[0].items()))
                        
            except Exception as e:
                print()
                print("=" * 80)
                print(f"file_id: {file_id} | qid: {qid}")
                print("-" * 80)
                print("ERROR:", e)
                updated_answer = ""
            
            # --- Get main entity -------------------------------------
            main_qids = sorted(set(re.findall(r'\bwd:(Q[1-9]\d*)\b', sparql_text)))

            # --- Get main property -----------------------------------
            properties = sorted(set(re.findall(r'\bP[1-9]\d*\b', sparql_text)))
            
            # --- get the new sparql for intermediate answers ---------
            parts = re.split(r'(XMLSchema#>)', sparql_text, maxsplit=1)
            if len(parts) > 1:
                prefixes = parts[0] + parts[1]
                sparql_query_body = parts[2].strip()
            else:
                raise ValueError("Expected exactly one 'XMLSchema#>' split point in the SPARQL string.")

            completion = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_SPARQL_LIST},
                    {"role": "user", "content": user_prompt_template.format(sparql_query=sparql_query_body)}
                ]
            )
            sparql_query_list = extract_converted_query(completion.choices[0].message.content).replace('\n', '')
            new_sparql_query = prefixes + sparql_query_list
            
            # --- Get the intermediate answers ---------------------------
            intermidate_list = []
            ep = choose_endpoint(new_sparql_query)
            try:
                result_json = run_query(ep, new_sparql_query)
                rows = format_results(result_json)
                if not rows:
                    print("(no results)")
                else:
                    for i, row in enumerate(rows):
                        _, first_value = next(iter(row.items()))
                        intermidate_list.append(extract_qid(first_value))
                        # print(new_sparql_query)
                        # print(f"[{i+1}] {extract_qid(first_value)}")
                        
            except Exception as e:
                print()
                print("=" * 80)
                print(f"file_id: {file_id} | qid: {qid}")
                print("ERROR:", e)
                print("-" * 80)
                print()

            # --- Edit on final answer ----------------------------------- 
            updated_answer_ = "yes" if updated_answer == "true" else "no" if updated_answer == "false" else updated_answer
            match = re.match(r"^https?://www\.wikidata\.org/entity/(Q\d+)$", updated_answer)
            if match:
                updated_answer_ = get_wikipedia_title_from_qid(match.group(1))

            # --- Write in file ------------------------------------------
            item = {
                "file_id": file_id,
                "qid": qid,
                "query": query,
                "dataset_answer": answer_value,
                "updated_answer": updated_answer_,
                "is_changed": not (str(answer_value) == str(updated_answer)),
                "main_entities": main_qids,
                "properties": properties,
                "intermidate_list": intermidate_list
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='openai/gpt-4o')
    args = parser.parse_args()
    
    # === Files ====================
    input_file = "corpus_datasets/qald_aggregation_samples/wikidata_aggregation.jsonl"
    output_file = "corpus_datasets/qald_aggregation_samples/wikidata_totallist_1.jsonl"
    
    get_annotations(input_file, output_file)


# python c1_corpus_dataset_preparation/get_intermediate_annotation.py
