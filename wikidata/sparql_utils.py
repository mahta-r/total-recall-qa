import time
import random
import tqdm
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote
from urllib.error import HTTPError
from SPARQLWrapper import SPARQLWrapper, JSON

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
MW_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TotalRecallRAG/0.1 (contact: mrafiee@umass.edu)"


def safe_query(sparql, max_retries=5):
    for attempt in range(max_retries):
        try:
            return sparql.query().convert()
        except HTTPError as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)
        except Exception as e:
            raise
    raise Exception("Max retries exceeded")



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


def is_wikidata_id(value):
    return isinstance(value, str) and value.startswith("Q") and value[1:].isdigit()


def get_subclasses_of_class(qid, endpoint=WDQS_ENDPOINT):
    sparql = SPARQLWrapper(endpoint)
    sparql.addCustomHttpHeader('User-Agent', 'vscode (mrafiee@umass.edu)')
    query = f"""
    SELECT ?subclass ?subclassLabel WHERE {{
      ?subclass wdt:P279* wd:{qid}.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = safe_query(sparql)
    subclasses = []
    for result in results["results"]["bindings"]:
        subclasses.append({
            "id": result["subclass"]["value"].split("/")[-1],
            "label": result["subclassLabel"]["value"]
        })
    return subclasses


def count_instances_in_batches(subclass_ids, endpoint=WDQS_ENDPOINT):
    union_blocks = "\nUNION\n".join([
        f"""
        {{
          BIND(wd:{qid} AS ?subclass)
          ?entity wdt:P31 wd:{qid}.
        }}
        """ for qid in subclass_ids
    ])
    query = f"""
    SELECT ?subclass (COUNT(?entity) AS ?instanceCount) WHERE {{
      {union_blocks}
    }}
    GROUP BY ?subclass
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.addCustomHttpHeader('User-Agent', 'vscode (mrafiee@umass.edu)')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = safe_query(sparql)
    counts = {}
    for result in results["results"]["bindings"]:
        qid = result["subclass"]["value"].split("/")[-1]
        counts[qid] = int(result["instanceCount"]["value"])
    # Fill in zero for any QIDs not returned
    for qid in subclass_ids:
        counts.setdefault(qid, 0)
    return counts


def get_subclasses_with_instance_count(qid, endpoint=WDQS_ENDPOINT, batch_size=20, sleep=1):
    subclasses = get_subclasses_of_class(qid, endpoint)
    subclasses_with_counts = add_instance_counts(subclasses[:1]) # Skip the first subclass (the class itself)
    return subclasses_with_counts


def add_instance_counts(subclasses, endpoint=WDQS_ENDPOINT, batch_size=20, sleep=1):
    # Split subclasses into batches
    for i in tqdm.tqdm(range(0, len(subclasses), batch_size)):
        batch = subclasses[i:min(i+batch_size, len(subclasses))]
        batch_ids = [subclass["id"] for subclass in batch]
        try:
            counts = count_instances_in_batches(batch_ids, endpoint)
            for subclass in batch:
                count = counts.get(subclass["id"], 0)
                subclass["instance_count"] = count
        except Exception as e:
            print(f"Error for batch {batch_ids}: {e}")
        time.sleep(sleep)  # Be polite to the endpoint
    return subclasses


def get_instances_of_class(qid, endpoint=WDQS_ENDPOINT, limit=100):
    sparql = SPARQLWrapper(endpoint)
    sparql.addCustomHttpHeader('User-Agent', 'vscode (mrafiee@umass.edu)')
    query = f"""
    SELECT ?entity ?entityLabel ?wikipedia WHERE {{
      ?entity wdt:P31 wd:{qid}.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
      OPTIONAL {{
        ?wikipedia schema:about ?entity ;
                   schema:isPartOf <https://en.wikipedia.org/> ;
                   schema:inLanguage "en" .
      }}
    }}
    LIMIT {limit}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = safe_query(sparql)
    instances = []
    for result in results["results"]["bindings"]:
        instances.append({
            "id": result["entity"]["value"].split("/")[-1],
            "label": result["entityLabel"]["value"],
            "wikipedia": result.get("wikipedia", {}).get("value", None)
        })
    return instances


def get_properties_of_subclass(subclass_qid, limit=100, endpoint=WDQS_ENDPOINT):
    query = f"""
    SELECT ?property ?propertyLabel ?propertyDescription (COUNT(*) AS ?count) WHERE {{
      ?entity wdt:P31 wd:{subclass_qid}.
      ?entity ?prop ?value.
      ?statementProperty wikibase:directClaim ?prop.
      FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
      BIND(IRI(STR(?statementProperty)) AS ?property)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
    }}
    GROUP BY ?property ?propertyLabel ?propertyDescription
    ORDER BY DESC(?count)
    LIMIT {limit}
    """

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader('User-Agent', 'vscode (mrafiee@umass.edu)')
    results = safe_query(sparql)

    properties = []
    for result in results["results"]["bindings"]:
        prop_uri = result["property"]["value"]
        prop_id = prop_uri.split("/")[-1]
        properties.append({
            "property_id": prop_id,
            "label": result.get("propertyLabel", {}).get("value", ""),
            "description": result.get("propertyDescription", {}).get("value", ""),
            "count": int(result["count"]["value"]),
        })
        
    return properties


def get_enwiki_category_for_class(qid):
    """Look up the main Wikipedia category for a class QID via P910, returning QID, label, and enwiki sitelink if present."""

    sparql = SPARQLWrapper(WDQS_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)

    query = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>
    SELECT ?category ?categoryLabel ?enwiki_cat WHERE {{
      wd:{qid} wdt:P910 ?category .
      OPTIONAL {{
        ?enwiki_cat schema:about ?category ;
                    schema:isPartOf <https://en.wikipedia.org/> ;
                    schema:inLanguage "en" .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
    }}
    """
    sparql.setQuery(query)
    results = safe_query(sparql)

    cats = []
    for row in results["results"]["bindings"]:
        cat_qid = row["category"]["value"].split("/")[-1]
        cat_label = row.get("categoryLabel", {}).get("value", "")
        enwiki_url = row.get("enwiki_cat", {}).get("value")
        enwiki_title = extract_title_from_url(enwiki_url) if enwiki_url else None
        cats.append({
            "qid": cat_qid,
            "label": cat_label,
            "enwiki_url": enwiki_url,
            "enwiki_title": enwiki_title,
        })
    return cats


def get_enwiki_lists_for_class(qid):
    """Look up enwiki 'List of ...' pages for a class QID via P360 or sitelink title match."""
    from SPARQLWrapper import SPARQLWrapper, JSON

    sparql = SPARQLWrapper(WDQS_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)

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
        wd:{qid} wdt:P31 wd:Q13406463 .
        ?enwiki_list schema:about wd:{qid} ;
                     schema:isPartOf <https://en.wikipedia.org/> ;
                     schema:inLanguage "en" .
      }}
    }}
    """
    sparql.setQuery(query)
    results = safe_query(sparql)

    lists = []
    for row in results["results"]["bindings"]:
        enwiki_url = row["enwiki_list"]["value"]
        enwiki_title = extract_title_from_url(enwiki_url)
        if enwiki_title and enwiki_title.lower().startswith("list_of"):
            lists.append({
                "enwiki_url": enwiki_url,
                "enwiki_title": enwiki_title,
        })
    return lists


def get_quantity_property(qid: str, endpoint=WDQS_ENDPOINT) -> Optional[int]:
    """
    Returns the value of the quantity property (P1114) for a Wikidata item (subclass QID).
    If not present, returns None.
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    query = f"""
    SELECT ?quantity WHERE {{
      wd:{qid} wdt:P1114 ?quantity .
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    results = safe_query(sparql)
    bindings = results["results"]["bindings"]
    if bindings:
        value = bindings[0]["quantity"]["value"]
        try:
            return int(value)
        except ValueError:
            return None
    return None


# def get_classes_with_quantity_property(endpoint=WDQS_ENDPOINT, limit=1000):
#     sparql = SPARQLWrapper(endpoint)
#     sparql.setReturnFormat(JSON)
#     sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
#     query = f"""
#     SELECT ?class ?classLabel ?quantity WHERE {{
#       ?class wdt:P1114 ?quantity .
#       SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
#     }}
#     ORDER BY DESC(?quantity)
#     """
#     # LIMIT {limit}
#     sparql.setQuery(query)
#     results = safe_query(sparql)
#     classes = []
#     for result in results["results"]["bindings"]:
#         qid = result["class"]["value"].split("/")[-1]
#         label = result.get("classLabel", {}).get("value", "")
#         quantity = result["quantity"]["value"]
#         classes.append({"id": qid, "label": label, "quantity": quantity})
#     return classes


def get_classes_with_single_quantity(endpoint=WDQS_ENDPOINT):
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    query = f"""
    SELECT ?class ?classLabel ?classDescription (SAMPLE(?quantity) AS ?uniqueQuantity) WHERE {{
    ?class wdt:P1114 ?quantity .
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
    }}
    GROUP BY ?class ?classLabel ?classDescription
    HAVING (COUNT(DISTINCT ?quantity) = 1)
    ORDER BY DESC(?uniqueQuantity)
    """
    sparql.setQuery(query)
    results = safe_query(sparql)
    classes = []
    for result in results["results"]["bindings"]:
        qid = result["class"]["value"].split("/")[-1]
        label = result.get("classLabel", {}).get("value", "")
        description = result.get("classDescription", {}).get("value", "")
        quantity = result["uniqueQuantity"]["value"]
        classes.append({"id": qid, "label": label, "description": description ,"quantity": quantity})
    return classes


def get_structural_properties(type_qid, endpoint=WDQS_ENDPOINT):
    query = f"""
    SELECT ?prop WHERE {{
      ?prop wdt:P31/wdt:P279* wd:{type_qid}.
    }}
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader('User-Agent', USER_AGENT)
    
    results = safe_query(sparql)
    return set(result["prop"]["value"].split("/")[-1] for result in results["results"]["bindings"])


def is_property_fully_populated(subclass_qid, property_id, endpoint=WDQS_ENDPOINT):
    query = f"""
    ASK {{
      ?entity wdt:P31 wd:{subclass_qid}.
      MINUS {{
        ?entity wdt:{property_id} ?value.
      }}
    }}
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader('User-Agent', USER_AGENT)
    result = safe_query(sparql)
    return not result["boolean"]  # False = not fully populated, so we return True if boolean is False


def get_entity_property_values(subclass_qid, property_id, endpoint=WDQS_ENDPOINT):
    query = f"""
    SELECT ?entity ?entityLabel ?value WHERE {{
      ?entity wdt:P31 wd:{subclass_qid}.
      ?entity wdt:{property_id} ?value.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
    }}
    """

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader('User-Agent', 'vscode (mrafiee@umass.edu)')
    results = safe_query(sparql)

    rows = []
    for r in results["results"]["bindings"]:
        rows.append({
            "entity_id": r["entity"]["value"].split("/")[-1],
            "entity_label": r.get("entityLabel", {}).get("value", ""),
            "value": r["value"]["value"]
        })

    return rows


def get_label_and_description(entity_id: str, endpoint=WDQS_ENDPOINT) -> Optional[Dict]:
    """
    Given a Wikidata entity QID, returns its label and description.
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    query = f"""
    SELECT ?entityLabel ?entityDescription WHERE {{
      VALUES ?entity {{ wd:{entity_id} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
    }}
    """
    sparql.setQuery(query)
    results = safe_query(sparql)
    bindings = results["results"]["bindings"]
    if bindings:
        return {
            "id": entity_id,
            "label": bindings[0].get("entityLabel", {}).get("value", ""),
            "description": bindings[0].get("entityDescription", {}).get("value", "")
        }
    return None


