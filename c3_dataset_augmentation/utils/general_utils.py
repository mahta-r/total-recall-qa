import time
import random
from urllib.parse import unquote
from urllib.error import HTTPError
from SPARQLWrapper import SPARQLWrapper, JSON

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"

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


def build_values_block(var, ids):
    prefixed = " ".join(f"wd:{i}" for i in ids)
    return f"VALUES {var} {{ {prefixed} }}"

def get_entity_info(entity_qids, endpoint=WDQS_ENDPOINT, user_agent="your-app (you@example.com)"):
    if not entity_qids:
        return []

    entities_block = build_values_block("?entity", entity_qids)

    query = f"""
    SELECT ?entity ?entityLabel ?wikipedia WHERE {{
      {entities_block}
      OPTIONAL {{
        ?wikipedia schema:about ?entity ;
                   schema:isPartOf <https://en.wikipedia.org/> .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", user_agent)
    results = sparql.query().convert()

    entity_info = {qid: {"id": qid, "label": "", "wikipedia": ""} for qid in entity_qids}

    for r in results["results"]["bindings"]:
        qid = r["entity"]["value"].rsplit("/", 1)[-1]
        label = r.get("entityLabel", {}).get("value", "")
        wiki = r.get("wikipedia", {}).get("value", "")
        entity_info[qid]["label"] = label
        entity_info[qid]["wikipedia"] = wiki

    # return list in same order as input
    return [entity_info[qid] for qid in entity_qids]


def get_properties_of_item(item_qid, limit=100, endpoint=WDQS_ENDPOINT):
    query = f"""
    SELECT ?property ?propertyLabel ?propertyDescription (COUNT(*) AS ?count) WHERE {{
      VALUES ?entity {{ wd:{item_qid} }}
      ?entity ?prop ?value .
      # This triple ensures ?prop is a wdt:… IRI, so no extra filter is needed
      ?property wikibase:directClaim ?prop .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
    }}
    GROUP BY ?property ?propertyLabel ?propertyDescription
    ORDER BY DESC(?count)
    """

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader('User-Agent', 'vscode (heydar.soudani@ru.nl)')
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

def build_values_block(var, ids):
    prefixed = " ".join(f"wd:{i}" for i in ids)
    return f"VALUES {var} {{ {prefixed} }}"

def get_property_values(entity_qids, property_ids, endpoint=WDQS_ENDPOINT, user_agent="your-app (you@example.com)"):
    """
    Returns:
    {
      "property_values": {
         "P856": [
            {"entity_id": "Q855", "entity_label": "...", "value": "..."},
            {"entity_id": "Q35314", "entity_label": "...", "value": "..."},
            ...
         ],
         "P31": [...],
         ...
      }
    }
    """
    if not entity_qids or not property_ids:
        return {"property_values": {}}

    entities_block = build_values_block("?entity", entity_qids)
    properties_block = build_values_block("?property", property_ids)

    query = f"""
    SELECT ?property ?entity ?entityLabel ?value ?valueLabel WHERE {{
      {entities_block}
      {properties_block}
      ?property wikibase:directClaim ?prop .
      ?entity ?prop ?value .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
    }}
    """

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", user_agent)

    results = sparql.query().convert()

    out = {"property_values": {}}
    # Keep a stable ordering of entities based on input list
    entity_order = {q:i for i,q in enumerate(entity_qids)}

    # Temporarily collect per property as dict {entity_id: list_of_values}
    tmp = {}

    for r in results["results"]["bindings"]:
        prop_id = r["property"]["value"].rsplit("/", 1)[-1]  # Pxxx
        ent_id = r["entity"]["value"].rsplit("/", 1)[-1]     # Qxxx
        ent_label = r.get("entityLabel", {}).get("value", ent_id)

        # Prefer human-readable label for IRI values; else use the literal string
        value_node = r["value"]
        if value_node["type"] == "uri":
            # If it's a wikidata entity (wd:Q…), use its label when we have it
            val_label = r.get("valueLabel", {}).get("value")
            # Still keep a readable fallback
            value_str = val_label if val_label else value_node["value"]
        else:
            # Literal (string/number/date/url)
            value_str = value_node["value"]

        tmp.setdefault(prop_id, {}).setdefault(ent_id, []).append(value_str)

    # Now format to the requested list-of-objects per property,
    # preserving the input entity order and joining multiple values
    for prop_id, ent_map in tmp.items():
        rows = []
        for ent in sorted(ent_map.keys(), key=lambda q: entity_order.get(q, 10**9)):
            # If an entity has multiple values for the same property, join them by " | "
            joined_value = " | ".join(ent_map[ent])
            rows.append({
                "entity_id": ent,
                "entity_label": next(  # find any label from our earlier pass
                    (r.get("entityLabel", {}).get("value", ent)
                     for r in results["results"]["bindings"]
                     if r["entity"]["value"].rsplit("/",1)[-1] == ent),
                    ent
                ),
                "value": joined_value
            })
        out["property_values"][prop_id] = rows

    # Ensure all requested properties are present, even if no matches (shouldn't happen if "shared")
    for pid in property_ids:
        out["property_values"].setdefault(pid, [])

    return out



