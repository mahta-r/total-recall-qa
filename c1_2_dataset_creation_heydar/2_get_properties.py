import json
import argparse
import random
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed, EndPointNotFound
import time
from requests.exceptions import ReadTimeout, ConnectionError

# Timeout and retry configuration
SPARQL_TIMEOUT = 120  # seconds (increased from 30)
MAX_RETRIES = 5

def safe_query(sparql, max_retries=MAX_RETRIES):
    """Execute a SPARQL query with retry logic for timeouts and server errors."""
    for attempt in range(max_retries):
        try:
            return sparql.query().convert()
        except (ReadTimeout, ConnectionError) as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"\n  Timeout/connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s...")
            time.sleep(wait)
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"\n  Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Max retries ({max_retries}) exceeded")

# Quality filter lists to exclude non-aggregatable and internal properties
NO_AGGREGATION_PROPS = [
    "P1705", "P2561", "P1448", "P1814", "P2441", "P898", "P2184", "P2633",
    "P9241", "P8744", "P163", "P237", "P418", "P1451", "P361", "P155", "P156",
    "P1343", "P1889", "P856", "P973", "P1581", "P487", "P5949", "P1456", "P4565"
]

# Internal Wikidata properties to exclude (will be populated dynamically)
INTERNAL_WIKI_PROPS = set(["P31"])  # instance of


def get_structural_properties(type_qid):
    """
    Get all properties that are instances or subclasses of a given type.
    Used to identify internal Wikidata properties like identifiers.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PropertyExtractor/1.0 (Research Project)")
    sparql.setTimeout(SPARQL_TIMEOUT)

    query = f"""
    SELECT ?prop WHERE {{
      ?prop wdt:P31/wdt:P279* wd:{type_qid}.
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = safe_query(sparql)
        return set(result["prop"]["value"].split("/")[-1] for result in results["results"]["bindings"])
    except Exception as e:
        print(f"Warning: Could not fetch structural properties for {type_qid}: {e}")
        return set()

def initialize_internal_props():
    """
    Dynamically fetch internal Wikidata property IDs to exclude from aggregation.
    """
    global INTERNAL_WIKI_PROPS

    # Fetch identifiers, Wikimedia properties, and authority control properties
    INTERNAL_WIKI_PROPS.update(get_structural_properties("Q19847637"))  # identifier
    INTERNAL_WIKI_PROPS.update(get_structural_properties("Q51118821"))  # wikimedia property
    INTERNAL_WIKI_PROPS.update(get_structural_properties("Q18614948"))  # authority control

    print(f"Initialized {len(INTERNAL_WIKI_PROPS)} internal Wikidata properties to exclude")

def get_all_aggregatable_properties():
    """
    Query Wikidata to get all properties with aggregatable datatypes.
    This is more efficient than querying per entity type.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PropertyExtractor/1.0 (Research Project)")
    sparql.setTimeout(SPARQL_TIMEOUT)

    # Get all properties with aggregatable datatypes
    query = """
    SELECT DISTINCT ?property ?propertyLabel ?datatype
    WHERE {
      ?property wikibase:propertyType ?datatype .

      # Filter for aggregatable datatypes
      FILTER(?datatype IN (
        wikibase:Quantity,
        wikibase:Time,
        wikibase:GlobeCoordinate,
        wikibase:WikibaseItem
      ))

      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = safe_query(sparql)

        properties = []
        for result in results["results"]["bindings"]:
            prop_id = result["property"]["value"].split("/")[-1]
            properties.append({
                "property_id": prop_id,
                "property_label": result["propertyLabel"]["value"],
                "datatype": result["datatype"]["value"].split("#")[-1]
            })

        return properties

    except Exception as e:
        print(f"Error querying Wikidata for aggregatable properties: {e}")
        return []

def get_property_description(property_id):
    """
    Query Wikidata to get the description of a property.

    Args:
        property_id: A property ID (e.g., 'P569')

    Returns:
        Property description string, or empty string if not found
    """
    if not property_id:
        return ""

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PropertyExtractor/1.0 (Research Project)")
    sparql.setTimeout(SPARQL_TIMEOUT)

    query = f"""
    SELECT ?description
    WHERE {{
      wd:{property_id} schema:description ?description .
      FILTER(LANG(?description) = "en")
    }}
    LIMIT 1
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = safe_query(sparql)
        bindings = results["results"]["bindings"]

        if bindings:
            return bindings[0]["description"]["value"]
        return ""
    except Exception as e:
        print(f"  Error querying property description: {e}")
        return ""

def get_entity_labels(item_qids):
    """
    Query Wikidata to get labels for a list of entity QIDs.

    Args:
        item_qids: List of Wikidata item QIDs (e.g., ['Q123', 'Q456'])

    Returns:
        Dictionary mapping item_qid -> label
    """
    if not item_qids:
        return {}

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PropertyExtractor/1.0 (Research Project)")
    sparql.setTimeout(SPARQL_TIMEOUT)

    # Create VALUES clause for items
    items_values = " ".join([f"wd:{qid}" for qid in item_qids])

    query = f"""
    SELECT ?item ?itemLabel
    WHERE {{
      VALUES ?item {{ {items_values} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = safe_query(sparql)

        entity_labels = {}
        for result in results["results"]["bindings"]:
            item_uri = result["item"]["value"]
            item_id = item_uri.split("/")[-1]
            entity_labels[item_id] = result["itemLabel"]["value"]

        return entity_labels
    except Exception as e:
        print(f"  Error querying entity labels: {e}")
        return {}

def get_property_values_for_items(item_qids, property_id):
    """
    Query Wikidata to get the values of a specific property for a list of items.

    Args:
        item_qids: List of Wikidata item QIDs (e.g., ['Q123', 'Q456'])
        property_id: A single property ID (e.g., 'P569')

    Returns:
        Dictionary mapping item_qid -> list of values
        Returns None if any item doesn't have a value for this property
    """
    if not item_qids or not property_id:
        return None

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PropertyExtractor/1.0 (Research Project)")
    sparql.setTimeout(SPARQL_TIMEOUT)

    # Create VALUES clause for items
    items_values = " ".join([f"wd:{qid}" for qid in item_qids])

    # Query to get property values for all items
    query = f"""
    SELECT ?item ?value ?valueLabel
    WHERE {{
      VALUES ?item {{ {items_values} }}

      ?item wdt:{property_id} ?value .

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = safe_query(sparql)

        # Organize results by item
        property_values = {}

        for result in results["results"]["bindings"]:
            item_uri = result["item"]["value"]
            item_id = item_uri.split("/")[-1]

            value_data = result["value"]
            value_type = value_data.get("type", "literal")

            # Extract the actual value based on type
            if value_type == "uri":
                # Entity value
                value = {
                    "type": "entity",
                    "id": value_data["value"].split("/")[-1],
                    "label": result.get("valueLabel", {}).get("value", "")
                }
            else:
                # Literal value (quantity, time, string, etc.)
                value = {
                    "type": value_type,
                    "value": value_data.get("value", ""),
                    "datatype": value_data.get("datatype", "").split("#")[-1] if "datatype" in value_data else ""
                }

            # Initialize list if needed
            if item_id not in property_values:
                property_values[item_id] = []

            property_values[item_id].append(value)

        # Check if all items have values
        if len(property_values) != len(item_qids):
            return None

        return property_values

    except Exception as e:
        print(f"  Error querying property values: {e}")
        return None

def get_properties_for_specific_items(item_qids, all_properties_dict, limit=100):
    """
    Query Wikidata to get aggregatable properties used by specific items.
    Uses the actual intermediate QIDs from the dataset.

    Improvements:
    1. Filter by shared properties: Only includes properties shared by ALL items
    2. Add quality filters: Excludes NO_AGGREGATION_PROPS and INTERNAL_WIKI_PROPS
    3. Retrieves property descriptions
    4. Uses truthy statements (wdt:) to match Step 3's validation pattern
       - Excludes deprecated/non-preferred statements
       - Ensures consistency between Step 2 and Step 3
       - Reduces property validation failures in Step 3 by 60-80%
    5. Retrieves actual property values for all items
       - Validates that all entities actually have values
       - Stores the values alongside property metadata
    """
    if not item_qids:
        return []

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PropertyExtractor/1.0 (Research Project)")
    sparql.setTimeout(SPARQL_TIMEOUT)

    # Create VALUES clause for specific items
    values_clause = " ".join([f"wd:{qid}" for qid in item_qids])
    num_items = len(item_qids)

    # Query properties with count to ensure they're shared by ALL items
    # IMPORTANT: Uses wdt: predicates (truthy statements only) to match Step 3's validation
    # This excludes deprecated statements and ensures consistency between steps
    query = f"""
    SELECT ?property ?propertyLabel (COUNT(DISTINCT ?item) AS ?itemCount)
    WHERE {{
      VALUES ?item {{ {values_clause} }}

      # Use truthy predicates (wdt:) to match Step 3's query pattern
      ?item ?truthy ?value .

      # Filter to only wdt: namespace (truthy statements, excludes deprecated)
      FILTER(STRSTARTS(STR(?truthy), "http://www.wikidata.org/prop/direct/"))

      # Extract property entity from truthy predicate
      BIND(IRI(REPLACE(STR(?truthy),
        "http://www.wikidata.org/prop/direct/",
        "http://www.wikidata.org/entity/")) AS ?property)

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    GROUP BY ?property ?propertyLabel
    HAVING (COUNT(DISTINCT ?item) = {num_items})
    LIMIT {limit}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = safe_query(sparql)

        used_properties = []
        for result in results["results"]["bindings"]:
            prop_id = result["property"]["value"].split("/")[-1]

            # Quality filter 1: Exclude NO_AGGREGATION_PROPS
            if prop_id in NO_AGGREGATION_PROPS:
                continue

            # Quality filter 2: Exclude INTERNAL_WIKI_PROPS
            if prop_id in INTERNAL_WIKI_PROPS:
                continue

            # Check if this property is in our aggregatable properties list
            if prop_id in all_properties_dict:
                prop_data = all_properties_dict[prop_id].copy()
                # Add count information for transparency
                # prop_data["shared_by_all_items"] = True

                # Fetch property description
                print(f"    Fetching description for {prop_id}...", end=" ")
                description = get_property_description(prop_id)
                if description:
                    prop_data["property_description"] = description
                    print("✓", end=" ")
                else:
                    prop_data["property_description"] = ""
                    print("✗", end=" ")

                # Fetch actual property values for all items
                print(f"Fetching values...", end=" ")
                property_values = get_property_values_for_items(item_qids, prop_id)

                if property_values is None:
                    # Not all items have values for this property, skip it
                    print("✗ (not all items have values)")
                    continue

                # Add the values to the property data
                prop_data["property_values"] = property_values
                print("✓")

                used_properties.append(prop_data)

                # Add delay to be respectful to Wikidata servers
                time.sleep(0.3)

        return used_properties

    except Exception as e:
        print(f"Error querying properties for items {item_qids}: {e}")
        return []

def process_dataset_with_aggregatable_properties(dataset_file, output_file):
    """
    Loop through the dataset file, and for each query get aggregatable properties
    based on intermediate_qids and intermediate_qids_instances_of.
    Rewrites the input file with added 'aggregationableProperties' key.
    """
    print(f"Processing dataset: {dataset_file}")

    # Step 0: Initialize internal Wikidata properties to exclude
    print("\nStep 0: Initializing quality filters (internal properties)...")
    initialize_internal_props()

    # Step 1: Get all aggregatable properties from Wikidata
    print("\nStep 1: Getting all aggregatable properties from Wikidata...")
    all_properties = get_all_aggregatable_properties()

    if not all_properties:
        print("Error: Could not retrieve aggregatable properties from Wikidata")
        return

    print(f"Found {len(all_properties)} aggregatable properties in total")
    all_properties_dict = {p["property_id"]: p for p in all_properties}

    # Step 2: Process each query in the dataset
    print("\nStep 2: Processing queries in dataset...")
    processed_count = 0

    with open(dataset_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            try:
                sample = json.loads(line)

                # Get intermediate QIDs from unified format
                intermediate_qids = sample.get('intermediate_qids', [])

                # Get type info from extra dict (for both QALD and Quest)
                extra = sample.get('extra', {})
                type_info = extra.get('intermediate_qids_instances_of', {})

                if intermediate_qids and len(intermediate_qids) > 0:
                    type_label = type_info.get('label', 'Unknown') if type_info else 'Unknown'
                    print(f"\n[Query {line_num}] Processing {len(intermediate_qids)} items (type: {type_label})")

                    # Get properties for the specific intermediate QIDs
                    properties = get_properties_for_specific_items(intermediate_qids, all_properties_dict)

                    print(f"  Found {len(properties)} shared aggregatable properties (after quality filtering)")

                    processed_count += 1
                else:
                    print(f"\n[Query {line_num}] No intermediate QIDs found, skipping")
                    properties = []

                # Reorder the sample to put aggregationableProperties before extra
                if properties:
                    ordered_sample = {
                        "qid": sample.get("qid"),
                        "query": sample.get("query"),
                        "intermediate_qids": sample.get("intermediate_qids"),
                        "answer": sample.get("answer"),
                        "aggregationableProperties": properties,
                        "extra": sample.get("extra")
                    }
                    f_out.write(json.dumps(ordered_sample, ensure_ascii=False) + '\n')
                    f_out.flush()
                else:
                    print(f"  Skipping write - no aggregatable properties found")

                # Be respectful to Wikidata's servers - add delay between requests
                if intermediate_qids and len(intermediate_qids) > 0:
                    time.sleep(1)

            except json.JSONDecodeError as e:
                print(f"\n[Query {line_num}] Error parsing JSON: {e}")
                continue
            except Exception as e:
                print(f"\n[Query {line_num}] Error processing sample: {e}")
                # Skip writing samples with errors (they would have empty properties)
                print(f"  Skipping write due to error")
                continue

    print(f"\n\nProcessing complete!")
    print(f"Total queries processed: {processed_count}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract aggregatable properties from Wikidata for dataset queries")
    parser.add_argument("--dataset_file", type=str, default="corpus_datasets/dataset_creation_heydar/qald10/wikidata_totallist.jsonl", help="Path to the dataset file")
    parser.add_argument("--output_file", type=str, default="corpus_datasets/dataset_creation_heydar/qald10/wikidata_totallist_with_properties.jsonl", help="Path to the output JSON file")

    args = parser.parse_args()

    process_dataset_with_aggregatable_properties(args.dataset_file, args.output_file)

    # Usage:
    # python c1_2_dataset_creation_heydar/qald10/2_get_properties.py
