import requests
import time



MW_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TotalRecallRAG/0.1 (contact: email@example.edu)"



def safe_get(url, params, headers, max_retries=5):
    backoff = 2
    for attempt in range(max_retries):
        r = requests.get(url, params=params, headers=headers)

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




def get_category_members(category, visited_categories=None, seen_pages=None):
    if visited_categories is None:
        print("Initializing visited_categories set")
        visited_categories = set()
    if seen_pages is None:
        seen_pages = set()
    if category in visited_categories:
        return
    visited_categories.add(category)
    
    print(f"\r{len(visited_categories)} categories visited so far..." , end="", flush=True)
    print(f"\r{len(seen_pages)} pages seen so far..." , end="", flush=True)

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