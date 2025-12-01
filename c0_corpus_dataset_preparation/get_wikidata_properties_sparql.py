import requests
import json
import time

ENDPOINT_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/sparql+json",
    "User-Agent": "total-recall-rag/0.3 (your-email@example.org)"
}

def get_properties_batch(limit=5000, offset=0, language="en", mode="all"):
    """
    mode:
      - "all"      -> all properties
      - "numeric"  -> only numeric-like properties (Quantity)
      - "itemlist" -> only item-valued properties (WikibaseItem)
    """
    if mode == "numeric":
        ptype_filter = "VALUES ?ptype { wikibase:Quantity }"
    elif mode == "itemlist":
        ptype_filter = "VALUES ?ptype { wikibase:WikibaseItem }"
    else:  # all
        ptype_filter = "OPTIONAL { ?property wikibase:propertyType ?ptype . }"

    query = f"""
    SELECT ?property ?propertyLabel ?propertyDescription ?ptype WHERE {{
      ?property a wikibase:Property .
      OPTIONAL {{ ?property wikibase:propertyType ?ptype . }}

      {ptype_filter}

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "{language}" .
      }}
    }}
    ORDER BY ?property
    LIMIT {limit}
    OFFSET {offset}
    """

    response = requests.post(
        ENDPOINT_URL,
        data={"query": query, "format": "json"},
        headers=HEADERS,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def uri_to_pid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]

def uri_to_datatype(uri: str | None) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


def fetch_properties(mode="numeric"):
    all_props = []
    limit = 5000
    offset = 0

    while True:
        print(f"Fetching batch offset={offset}, mode={mode}")
        data = get_properties_batch(limit=limit, offset=offset, mode=mode)

        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            break

        for row in bindings:
            prop_uri = row["property"]["value"]
            pid = uri_to_pid(prop_uri)
            label = row.get("propertyLabel", {}).get("value", "")
            desc = row.get("propertyDescription", {}).get("value", "")
            ptype_uri = row.get("ptype", {}).get("value", None)
            datatype = uri_to_datatype(ptype_uri)

            all_props.append({
                "id": pid,
                "uri": prop_uri,
                "label": label,
                "description": desc,
                "datatype": datatype,
            })

        if len(bindings) < limit:
            break
        offset += limit
        time.sleep(0.5)

    return all_props


if __name__ == "__main__":
    # numeric-only properties
    numeric_props = fetch_properties(mode="numeric")
    with open("corpus_datasets/wikidata_numeric_properties.json", "w", encoding="utf-8") as f:
        json.dump(numeric_props, f, ensure_ascii=False, indent=2)

    # item-valued “list” properties
    list_props = fetch_properties(mode="itemlist")
    with open("corpus_datasets/wikidata_itemlist_properties.json", "w", encoding="utf-8") as f:
        json.dump(list_props, f, ensure_ascii=False, indent=2)


# python c0_corpus_dataset_preparation/get_wikidata_properties_sparql.py








# import requests
# import json
# import time

# ENDPOINT_URL = "https://query.wikidata.org/sparql"
# HEADERS = {
#     "Accept": "application/sparql+json",
#     "User-Agent": "total-recall-rag/0.1 (heydar.soudani@yourdomain.example)"  # <- put real email/affiliation
# }

# def get_properties_batch(limit=10000, offset=0, language="en"):
#     query = f"""
#     SELECT ?property ?propertyLabel ?propertyDescription WHERE {{
#       ?property a wikibase:Property .
#       SERVICE wikibase:label {{
#         bd:serviceParam wikibase:language "{language}" .
#       }}
#     }}
#     ORDER BY ?property
#     LIMIT {limit}
#     OFFSET {offset}
#     """

#     # Use POST (recommended for big queries)
#     response = requests.post(
#         ENDPOINT_URL,
#         data={"query": query, "format": "json"},
#         headers=HEADERS,
#         timeout=120,
#     )

#     # Debug if not OK
#     if not response.ok:
#         print("HTTP error from endpoint:")
#         print("status:", response.status_code)
#         print("headers:", response.headers)
#         print("body (first 500 chars):")
#         print(response.text[:500])
#         response.raise_for_status()

#     # Sometimes you still get HTML with status 200 if rate-limited
#     ctype = response.headers.get("Content-Type", "")
#     if "json" not in ctype:
#         print("Unexpected content type:", ctype)
#         print("Body (first 500 chars):")
#         print(response.text[:500])
#         raise RuntimeError("Endpoint did not return JSON. Probably rate-limited or error page.")

#     try:
#         return response.json()
#     except json.JSONDecodeError:
#         print("Failed to decode JSON. Raw body (first 500 chars):")
#         print(response.text[:500])
#         raise


# def uri_to_pid(uri: str) -> str:
#     return uri.rsplit("/", 1)[-1]


# def main():
#     all_props = []
#     limit = 5000   # lower than 10000 to be nicer
#     offset = 0

#     while True:
#         print(f"Fetching batch: offset={offset}, limit={limit}")
#         data = get_properties_batch(limit=limit, offset=offset)

#         bindings = data.get("results", {}).get("bindings", [])
#         if not bindings:
#             break

#         for row in bindings:
#             prop_uri = row["property"]["value"]
#             pid = uri_to_pid(prop_uri)
#             label = row.get("propertyLabel", {}).get("value", "")
#             description = row.get("propertyDescription", {}).get("value", "")

#             all_props.append({
#                 "id": pid,
#                 "uri": prop_uri,
#                 "label": label,
#                 "description": description,
#             })

#         if len(bindings) < limit:
#             break

#         offset += limit
#         # Be polite: sleep a bit between batches
#         time.sleep(0.5)

#     print(f"Total properties fetched: {len(all_props)}")

#     out_file = "corpus_datasets/wikidata_properties.json"
#     with open(out_file, "w", encoding="utf-8") as f:
#         json.dump(all_props, f, ensure_ascii=False, indent=2)

#     print(f"Saved to {out_file}")


# if __name__ == "__main__":
#     main()

   
    
    



