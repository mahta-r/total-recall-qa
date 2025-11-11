"""
WDQS ↔︎ Wikipedia overlap checker for subclass completeness.

Given:
  - a Wikidata class QID (e.g., Q901 for "Nobel Prize in Physics laureate")
  - optional enwiki category title(s) (e.g., "Category:Nobel laureates in Physics")
  - optional enwiki list page title(s) (e.g., "List of Nobel laureates in Physics")

This script:
  1) Pulls all enwiki sitelinks for items that are instance of the class (wdt:P31 wd:<QID>)
  2) Pulls all enwiki pages from the given category/categories and/or list page(s)
  3) Maps Wikipedia pages to Wikidata items via pageprops=wikibase_item (where available)
  4) Computes overlap statistics to estimate completeness

Usage:
  python wdqs_wikipedia_overlap.py --class Q901 \
    --category "Category:Nobel laureates in Physics" \
    --list "List of Nobel laureates in Physics"

Notes:
  - Requires: requests
  - Respects continuation for MediaWiki API, with basic retries
  - Resolves redirects to canonical titles
  - Outputs a summary JSON and prints human-readable stats
"""

import argparse
import json
import time
import requests
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

from wikidata.sparql_utils import safe_query


WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
MW_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TotalRecallRAG/0.1 (contact: mrafiee@umass.edu)"


def _req_get(url: str, params: Dict[str, str], max_retries: int = 5, backoff: float = 1.0):
    headers = {"User-Agent": USER_AGENT}
    for i in range(max_retries):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(backoff * (2 ** i))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def normalize_title(title: str) -> str:
    t = title.replace(" ", "_")
    if not t:
        return t
    return t[0].upper() + t[1:]


def extract_title_from_url(url: str) -> Optional[str]:
    try:
        if "/wiki/" in url:
            title = url.split("/wiki/", 1)[1]
            title = unquote(title)
            return normalize_title(title)
    except Exception:
        return None
    return None


from SPARQLWrapper import SPARQLWrapper, JSON


def wdqs_items_with_enwiki_sitelinks(qid, limit=100, endpoint="https://query.wikidata.org/sparql"):
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)

    query = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>

    SELECT ?item ?article WHERE {{
      ?item wdt:P31 wd:{qid} .
      ?article schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:inLanguage "en" .
    }}
    LIMIT {limit}
    """
    sparql.setQuery(query)
    results = safe_query(sparql)

    titles = set()
    item_for_title = {}

    for row in results["results"]["bindings"]:
        qid_row = row["item"]["value"].split("/")[-1]
        article_url = row["article"]["value"]
        # Extract enwiki title from URL
        title = article_url.split("/wiki/", 1)[-1]
        title = title.replace("_", " ")
        titles.add(title)
        item_for_title[title] = qid_row

    return titles, item_for_title



def mw_resolve_redirects(titles: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    BATCH = 50
    for i in range(0, len(titles), BATCH):
        batch = titles[i:i+BATCH]
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(batch),
            "redirects": 1,
            "converttitles": 1,
        }
        r = _req_get(MW_API, params)
        j = r.json()

        norm = {normalize_title(n["from"]): normalize_title(n["to"]) for n in j.get("query", {}).get("normalized", [])}
        redirs = {normalize_title(rd["from"]): normalize_title(rd["to"]) for rd in j.get("query", {}).get("redirects", [])}
        pages = j.get("query", {}).get("pages", {})
        for _, page in pages.items():
            t = page.get("title")
            if t:
                canon = normalize_title(t)
                mapping[canon] = canon
        for src, dst in norm.items():
            mapping[src] = dst
        for src, dst in redirs.items():
            mapping[src] = dst

        time.sleep(0.05)
    return mapping


def mw_category_members(category: str) -> Set[str]:
    titles: Set[str] = set()
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "page",
            "cmnamespace": 0,
            "cmlimit": "500",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        r = _req_get(MW_API, params)
        j = r.json()
        for p in j.get("query", {}).get("categorymembers", []):
            titles.add(normalize_title(p["title"]))
        cmcontinue = j.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(0.05)

    if titles:
        mapping = mw_resolve_redirects(sorted(titles))
        titles = {mapping.get(t, t) for t in titles}
    return titles


def mw_list_links(list_title: str) -> Set[str]:
    titles: Set[str] = set()
    plcontinue = None
    while True:
        params = {
            "action": "query",
            "format": "json",
            "prop": "links",
            "titles": list_title,
            "plnamespace": 0,
            "pllimit": "500",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue
        r = _req_get(MW_API, params)
        j = r.json()
        pages = j.get("query", {}).get("pages", {})
        for _, page in pages.items():
            for link in page.get("links", []):
                titles.add(normalize_title(link["title"]))
        plcontinue = j.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break
        time.sleep(0.05)

    if titles:
        mapping = mw_resolve_redirects(sorted(titles))
        titles = {mapping.get(t, t) for t in titles}
    return titles


def mw_titles_to_qids(titles: List[str]) -> Dict[str, Optional[str]]:
    title2qid: Dict[str, Optional[str]] = {}
    BATCH = 50
    for i in range(0, len(titles), BATCH):
        batch = titles[i:i+BATCH]
        params = {
            "action": "query",
            "format": "json",
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "redirects": 1,
            "converttitles": 1,
            "titles": "|".join(batch),
        }
        r = _req_get(MW_API, params)
        j = r.json()
        pages = j.get("query", {}).get("pages", {})
        for _, page in pages.items():
            title = normalize_title(page.get("title", ""))
            qid = None
            if "pageprops" in page and "wikibase_item" in page["pageprops"]:
                qid = page["pageprops"]["wikibase_item"]
            if title:
                title2qid[title] = qid
        time.sleep(0.05)
    return title2qid


def compute_overlap(class_qid: str, categories: List[str], lists: List[str], wd_limit: Optional[int] = None) -> Dict:
    wd_titles, wd_title2qid = wdqs_items_with_enwiki_sitelinks(class_qid, limit=100)

    wiki_titles: Set[str] = set()
    for cat in categories:
        wiki_titles |= mw_category_members(cat)
    for lst in lists:
        wiki_titles |= mw_list_links(lst)

    wiki_title2qid = mw_titles_to_qids(sorted(wiki_titles)) if wiki_titles else {}

    A = wiki_titles
    B = wd_titles

    inter = A & B
    only_wiki = A - B
    only_wd = B - A

    coverage = (len(inter) / len(A)) if A else 1.0
    jaccard = (len(inter) / len(A | B)) if (A or B) else 1.0

    only_wiki_with_qids = {t: wiki_title2qid.get(t) for t in sorted(only_wiki)}
    only_wd_with_qids = {t: wd_title2qid.get(t) for t in sorted(only_wd)}

    return {
        "class_qid": class_qid,
        "counts": {
            "wikipedia_A": len(A),
            "wikidata_B": len(B),
            "intersection": len(inter),
            "only_wikipedia": len(only_wiki),
            "only_wikidata": len(only_wd),
        },
        "metrics": {
            "coverage_A_in_B": coverage,
            "jaccard": jaccard,
            "abs_delta": abs(len(A) - len(B)),
        },
        "samples": {
            "only_wikipedia_titles": sorted(list(only_wiki))[:50],
            "only_wikidata_titles": sorted(list(only_wd))[:50],
        },
        "gaps": {
            "only_wikipedia_title_to_qid": only_wiki_with_qids,
            "only_wikidata_title_to_qid": only_wd_with_qids,
        },
    }

def get_enwiki_category_for_class(qid):
    """Look up the main Wikipedia category for a class QID via P910."""
    query = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>
    SELECT ?enwiki_cat WHERE {{
      wd:{qid} wdt:P910 ?category .
      ?enwiki_cat schema:about ?category ;
                  schema:isPartOf <https://en.wikipedia.org/> ;
                  schema:inLanguage "en" .
    }}
    """
    r = _req_get(WDQS_ENDPOINT, {"query": query, "format": "json"})
    rows = r.json().get("results", {}).get("bindings", [])
    cats = []
    for row in rows:
        url = row["enwiki_cat"]["value"]
        title = extract_title_from_url(url)
        if title and title.startswith("Category:"):
            cats.append(title)
    return cats


def get_enwiki_lists_for_class(qid):
    """Look up enwiki 'List of ...' pages for a class QID via P360 or sitelink title match."""
    query = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>
    SELECT ?enwiki_list WHERE {{
      {{
        wd:{qid} wdt:P360 ?listitem .
        ?enwiki_list schema:about ?listitem ;
                     schema:isPartOf <https://en.wikipedia.org/> ;
                     schema:inLanguage "en" .
      }}
      UNION
      {{
        wd:{qid} ?p ?dummy .
        ?enwiki_list schema:about wd:{qid} ;
                     schema:isPartOf <https://en.wikipedia.org/> ;
                     schema:inLanguage "en" .
      }}
    }}
    """
    r = _req_get(WDQS_ENDPOINT, {"query": query, "format": "json"})
    rows = r.json().get("results", {}).get("bindings", [])
    lists = []
    for row in rows:
        url = row["enwiki_list"]["value"]
        title = extract_title_from_url(url)
        if title and title.lower().startswith("list_of"):
            lists.append(title)
    return lists


def compute_overlap_auto(qid, wd_limit=None):
    """Automatically look up enwiki category/list pages for a class QID and run overlap check."""
    cats = get_enwiki_category_for_class(qid)
    lists = get_enwiki_lists_for_class(qid)

    print(f"[Auto lookup] Found categories: {cats}")
    print(f"[Auto lookup] Found lists: {lists}")

    return compute_overlap(qid, cats, lists, wd_limit=wd_limit)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="class_qid", required=True, help="Wikidata class QID (e.g., Q5)")
    ap.add_argument("--category", action="append", default=[], help='enwiki Category title')
    ap.add_argument("--list", dest="lists", action="append", default=[], help='enwiki List page title')
    ap.add_argument("--wd-limit", type=int, default=None, help="Optional cap for WDQS paging (for testing)")
    ap.add_argument("--out", default=None, help="Write detailed JSON to this path")
    args = ap.parse_args()

    result = compute_overlap(args.class_qid, args.category, args.lists, wd_limit=args.wd_limit)

    print("=== Overlap Summary ===")
    print(f"Class: {result['class_qid']}")
    c = result["counts"]
    print(f"  Wikipedia A: {c['wikipedia_A']}")
    print(f"  Wikidata  B: {c['wikidata_B']}")
    print(f"  Intersection: {c['intersection']}")
    print(f"  Only Wikipedia: {c['only_wikipedia']}")
    print(f"  Only Wikidata: {c['only_wikidata']}")

    m = result["metrics"]
    print("=== Metrics ===")
    print(f"  Coverage (A in B): {m['coverage_A_in_B']:.4f}")
    print(f"  Jaccard: {m['jaccard']:.4f}")
    print(f"  |A - B|: {m['abs_delta']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nWrote detailed JSON to: {args.out}")


if __name__ == "__main__":
    # main()
    # titles, mapping = wdqs_items_with_enwiki_sitelinks("Q174844", limit=100)
    # print(titles)
    # print(mapping)
    result = compute_overlap_auto("Q174844")

    print("=== Overlap Summary ===")
    print(f"Class: {result['class_qid']}")
    c = result["counts"]
    print(f"  Wikipedia A: {c['wikipedia_A']}")
    print(f"  Wikidata  B: {c['wikidata_B']}")
    print(f"  Intersection: {c['intersection']}")
    print(f"  Only Wikipedia: {c['only_wikipedia']}")
    print(f"  Only Wikidata: {c['only_wikidata']}")

    m = result["metrics"]
    print("=== Metrics ===")
    print(f"  Coverage (A in B): {m['coverage_A_in_B']:.4f}")
    print(f"  Jaccard: {m['jaccard']:.4f}")
    print(f"  |A - B|: {m['abs_delta']}")
