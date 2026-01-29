import requests
import time
from urllib.parse import unquote

from io_utils import read_jsonl_from_file, read_json_from_file, write_json_to_file



MW_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TotalRecallRAG/0.1 (contact: email@example.edu)"



def safe_get(url, params, headers, session=None, max_retries=5):
    backoff = 2

    if session:
        session.headers.update(headers)

    for attempt in range(max_retries):
        if session:
            r = session.get(url, params=params, headers=headers, timeout=20)
        else:
            r = requests.get(url, params=params, headers=headers, timeout=20)

        if r.status_code == 200:
            if "application/json" in r.headers.get("Content-Type", ""):
                return r
            else:
                raise ValueError("Non-JSON response")

        if r.status_code in (400, 403, 404):
            raise RuntimeError(f"Fatal error {r.status_code}")

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", backoff))
            print(f"Rate limited. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            backoff *= 2
            continue

        if r.status_code in (500, 502, 503, 504, 409):
            print(f"Rate limited. Retrying after {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2
            continue

        raise RuntimeError(f"Unhandled status code {r.status_code}")

    raise RuntimeError("Max retries exceeded")



def fetch_pageid(wikipedia_url, session):
    title = unquote(wikipedia_url.split("/wiki/")[-1])
    headers={
        "User-Agent": USER_AGENT
    }
    params={
        "action": "query",
        "titles": title,
        "redirects": 1,
        "converttitles": 1,
        "prop": "info",
        "format": "json",
        "formatversion": 2,
        "origin": "*", 
    }

    resp = safe_get(MW_API, headers=headers, params=params, session=session)

    if not resp.text.strip():
        raise RuntimeError("Empty response body")

    result = resp.json()

    pages = result.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        raise RuntimeError("Page not found")

    pageid = pages[0]["pageid"]
    assert pageid is not None, f"Page not found for {wikipedia_url}"
    assert str(pageid).isdigit(), f"Invalid pageid for {wikipedia_url}: {pageid}"
    assert int(pageid) > 0, f"Page not found for {wikipedia_url}"
    
    return pageid




def get_category_members(category, visited_categories=None, seen_pages=None):
    if visited_categories is None:
        print("Initializing visited_categories set")
        visited_categories = set()
    if seen_pages is None:
        seen_pages = set()
    if category in visited_categories:
        return
    visited_categories.add(category)
    
    # print(f"\r{len(visited_categories)} categories, {len(seen_pages)} pages visited so far..." , end="", flush=True)
    # if len(seen_pages) % 50 == 0:
    #     print(visited_categories)

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "max",
        "format": "json",
        "redirects": 1,
        # "prop": "pageprops",
        # "ppprop": "wikibase_item"
    }
    headers = {"User-Agent": USER_AGENT}
    while True:
        r = safe_get(MW_API, headers=headers, params=params)
        result = r.json()
        for cat in [item['title'] for item in result["query"]["categorymembers"] if item["ns"] == 14]:
            print(cat)
        print(result["continue"] if "continue" in result else "No continue")
        print("--------------------------------")
        input()
        for item in result["query"]["categorymembers"]:
            title = item["title"]
            if item["ns"] == 0:  # 0 = article namespace
                if title not in seen_pages:
                    seen_pages.add(title)
                    yield item
            if item["ns"] == 14:  # 14 = category namespace
                # Recurse into subcategories
                yield from get_category_members(title, visited_categories, seen_pages)

        if "continue" not in result:
            break

        params = {
            **params,
            **result["continue"]
        }


def get_direct_category_pages(category):
    """
    Yield only pages (ns=0) that are directly in `category`.
    Does NOT recurse into subcategories.
    """

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "max",
        "format": "json",
        "redirects": 1,
        "prop": "pageprops",
        "ppprop": "wikibase_item",
    }

    headers = {"User-Agent": USER_AGENT}

    while True:
        r = safe_get(MW_API, headers=headers, params=params)
        result = r.json()

        for item in result["query"]["categorymembers"]:
            if item["ns"] == 0:  # article namespace only
                # yield {
                #     "pageid": item["pageid"],
                #     "title": item["title"],
                #     "qid": item.get("pageprops", {}).get("wikibase_item"),
                # }
                yield item

        if "continue" not in result:
            break

        params = {**params, **result["continue"]}



def get_direct_category_pages_with_qids(category):
    """
    Yield only direct article pages (ns=0) in a category,
    including their Wikidata QIDs if they exist.
    """

    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmnamespace": 0,      # only articles
        "gcmlimit": "max",
        "prop": "pageprops",
        "ppprop": "wikibase_item",
        "format": "json",
        "redirects": 1,
    }

    headers = {"User-Agent": USER_AGENT}

    while True:
        r = safe_get(MW_API, headers=headers, params=params)
        data = r.json()

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            yield {
                "pageid": page["pageid"],
                "title": page["title"],
                "qid": page.get("pageprops", {}).get("wikibase_item"),
            }

        if "continue" not in data:
            break

        params = {**params, **data["continue"]}






# def get_category_members(category):
#     params = {
#         "action": "query",
#         "list": "categorymembers",
#         "cmtitle": category,
#         "cmlimit": "max",
#         "format": "json",
#         "redirects": 1
#     }
#     headers = {"User-Agent": USER_AGENT}
#     while True:
#         r = safe_get(MW_API, headers=headers, params=params)
#         result = r.json()
#         for item in result["query"]["categorymembers"]:
#             if item["ns"] == 0:  # 0 = article namespace
#                 yield item
#             # Recurse into subcategories
#             elif item["ns"] == 14:  # 14 = category namespace
#                 yield from some_random_name(item["title"])
#             else:
#                 print(f"Skipping item in ns {item['ns']}: {item['title']}")
#         if "continue" not in result:
#             break
#         params.update(result["continue"])