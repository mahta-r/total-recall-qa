import json
import argparse
from SPARQLWrapper import SPARQLWrapper, JSON
import time

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
    sparql.setTimeout(30)

    query = f"""
    SELECT ?prop WHERE {{
      ?prop wdt:P31/wdt:P279* wd:{type_qid}.
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
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
    sparql.setTimeout(60)

    # Get all properties with aggregatable datatypes
    query = """
    SELECT DISTINCT ?property ?propertyLabel ?datatype
    WHERE {
      ?property wikibase:propertyType ?datatype .

      # Filter for aggregatable datatypes
      FILTER(?datatype IN (
        wikibase:Quantity,
        wikibase:Time,
        wikibase:GlobeCoordinate
      ))

      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()

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

def get_properties_for_specific_items(item_qids, all_properties_dict, limit=100):
    """
    Query Wikidata to get aggregatable properties used by specific items.
    Uses the actual intermediate QIDs from the dataset.

    Improvements:
    1. Filter by shared properties: Only includes properties shared by ALL items
    2. Add quality filters: Excludes NO_AGGREGATION_PROPS and INTERNAL_WIKI_PROPS
    """
    if not item_qids:
        return []

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PropertyExtractor/1.0 (Research Project)")
    sparql.setTimeout(30)

    # Create VALUES clause for specific items
    values_clause = " ".join([f"wd:{qid}" for qid in item_qids])
    num_items = len(item_qids)

    # Query properties with count to ensure they're shared by ALL items
    query = f"""
    SELECT ?property ?propertyLabel (COUNT(DISTINCT ?item) AS ?itemCount)
    WHERE {{
      VALUES ?item {{ {values_clause} }}

      ?item ?p ?value .
      ?property wikibase:directClaim ?p .

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    GROUP BY ?property ?propertyLabel
    HAVING (COUNT(DISTINCT ?item) = {num_items})
    LIMIT {limit}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()

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
                prop_data["shared_by_all_items"] = True
                used_properties.append(prop_data)

        return used_properties

    except Exception as e:
        print(f"Error querying properties for items {item_qids}: {e}")
        return []

def process_dataset_with_aggregatable_properties(dataset_file, output_file):
    """
    Loop through the dataset file, and for each query get aggregatable properties
    based on intermidate_qids and intermidate_qids_instances_of.
    Output file has same structure as input with new key 'aggregationableProperties'.
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
                # if line_num == 10:
                #     break
                
                sample = json.loads(line)

                # Get intermediate QIDs
                intermediate_qids = sample.get('intermidate_qids', [])
                type_info = sample.get('intermidate_qids_instances_of', {})

                if intermediate_qids and len(intermediate_qids) > 0:
                    type_label = type_info.get('label', 'Unknown') if type_info else 'Unknown'
                    print(f"\n[Query {line_num}] Processing {len(intermediate_qids)} items (type: {type_label})")

                    # Get properties for the specific intermediate QIDs
                    properties = get_properties_for_specific_items(intermediate_qids, all_properties_dict)

                    # Add aggregatable properties to the sample
                    sample['aggregationableProperties'] = properties

                    print(f"  Found {len(properties)} shared aggregatable properties (after quality filtering)")

                    processed_count += 1
                else:
                    print(f"\n[Query {line_num}] No intermediate QIDs found, skipping")
                    sample['aggregationableProperties'] = []

                # Write the updated sample to output file only if aggregationableProperties is not empty
                if sample['aggregationableProperties']:
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
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
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="corpus_datasets/dataset_creation_heydar/qald10/wikidata_totallist.jsonl",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="corpus_datasets/dataset_creation_heydar/qald10/wikidata_totallist_with_properties.jsonl",
        help="Path to the output JSON file"
    )

    args = parser.parse_args()

    process_dataset_with_aggregatable_properties(args.dataset_file, args.output_file)

    # Usage:
    # python c1_2_dataset_creation_heydar/qald10/2_get_properties.py
