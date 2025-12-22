import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from openai import OpenAI

try:
    from c1_2_qald_dataset_augmentation.utils.io_utils import read_text_from_file
except:
    from utils.io_utils import read_text_from_file

from datetime import datetime
from statistics import mean


def calculate_answer(property_values, aggregation_function, datatype):
    """
    Calculate the answer based on property values and aggregation function.

    Args:
        property_values: Dictionary mapping item_qid -> list of values
        aggregation_function: One of COUNT, SUM, AVG, MAX, MIN, EARLIEST, LATEST
        datatype: Property datatype (Time, Quantity, etc.)

    Returns:
        Calculated answer as string or number
    """
    if not property_values:
        return None

    # Extract all values
    all_values = []
    for item_values in property_values.values():
        for val in item_values:
            if val['type'] == 'literal':
                all_values.append(val['value'])
            elif val['type'] == 'entity':
                # For entity values, we can count them
                all_values.append(val['id'])

    if not all_values:
        return None

    agg_func = aggregation_function.upper()

    try:
        # COUNT aggregation
        if agg_func == 'COUNT':
            return len(all_values)

        # For Time datatype - always return years, not full dates
        if datatype == 'Time':
            # Parse dates and extract years
            years = []
            for val in all_values:
                try:
                    # Parse ISO format datetime
                    if isinstance(val, str) and 'T' in val:
                        dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                        years.append(dt.year)
                except:
                    continue

            if not years:
                return None

            if agg_func == 'EARLIEST' or agg_func == 'MIN':
                return min(years)
            elif agg_func == 'LATEST' or agg_func == 'MAX':
                return max(years)
            elif agg_func == 'AVG':
                return round(mean(years), 2)
            else:
                # Default to earliest for Time
                return min(years)

        # For Quantity datatype
        elif datatype == 'Quantity':
            # Extract numeric values
            numeric_values = []
            for val in all_values:
                try:
                    if isinstance(val, (int, float)):
                        numeric_values.append(float(val))
                    elif isinstance(val, str):
                        # Try to extract number from string
                        num_str = val.split()[0]  # Take first token
                        numeric_values.append(float(num_str))
                except:
                    continue

            if not numeric_values:
                return None

            if agg_func == 'SUM':
                return round(sum(numeric_values), 2)
            elif agg_func == 'AVG':
                return round(mean(numeric_values), 2)
            elif agg_func == 'MAX':
                return round(max(numeric_values), 2)
            elif agg_func == 'MIN':
                return round(min(numeric_values), 2)

        # Default to count for other cases
        return len(all_values)

    except Exception as e:
        print(f"      Error calculating answer: {e}")
        return None


def generate_total_recall_query(client, prompt_template, original_query, property_label, property_id, datatype, entity_type, num_entities, model_name):
    """
    Generate a new Total Recall query using LLM.

    Args:
        client: OpenAI client instance
        prompt_template: Prompt template string
        original_query: The original query
        property_label: Label of the property
        property_id: ID of the property (e.g., P569)
        datatype: Datatype of the property
        entity_type: Type of entities (e.g., "human")
        num_entities: Number of entities
        model_name: Name of the model to use

    Returns:
        Generated query string
    """
    # Fill in the prompt template
    prompt = prompt_template.format(
        original_query=original_query,
        property_label=property_label,
        property_id=property_id,
        datatype=datatype,
        entity_type=entity_type,
        num_entities=num_entities
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        response_text = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            # Sometimes LLMs wrap JSON in markdown code blocks
            if response_text.startswith('```'):
                # Extract JSON from code block
                lines = response_text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or (not line.startswith('```') and '{' in line):
                        json_lines.append(line)
                response_text = '\n'.join(json_lines).strip()

            response_json = json.loads(response_text)
            generated_query = response_json.get('question', '').strip()
            aggregation_function = response_json.get('aggregation', '').strip().upper()

            if not generated_query or not aggregation_function:
                print(f"    Invalid response format - missing fields. Response: {response_text}")
                return None

            return {
                'question': generated_query,
                'aggregation': aggregation_function
            }
        except json.JSONDecodeError as e:
            # Fallback: try to extract from text manually
            print(f"    JSON parse error: {e}")
            print(f"    Attempting manual extraction...")

            # Try to find JSON pattern in the text
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                try:
                    response_json = json.loads(json_match.group(0))
                    generated_query = response_json.get('question', '').strip()
                    aggregation_function = response_json.get('aggregation', '').strip().upper()

                    if generated_query and aggregation_function:
                        return {
                            'question': generated_query,
                            'aggregation': aggregation_function
                        }
                except:
                    pass

            print(f"    Could not extract valid JSON from response: {response_text}")
            return None

    except Exception as e:
        print(f"    Error generating query with LLM: {e}")
        return None


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
    sparql.setTimeout(30)

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
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]

        if bindings:
            return bindings[0]["description"]["value"]
        return ""
    except Exception as e:
        print(f"  Error querying property description: {e}")
        return ""


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
    sparql.setTimeout(30)

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
        results = sparql.query().convert()

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


def get_last_processed_qid(output_file):
    """
    Read the output file and find the last processed QID.

    Args:
        output_file: Path to the output file

    Returns:
        The last processed QID, or None if file doesn't exist or is empty
    """
    if not os.path.exists(output_file):
        return None

    try:
        last_qid = None
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    last_qid = entry.get('qid')
                except:
                    continue
        return last_qid
    except:
        return None


def process_dataset_for_valid_pairs(dataset_file, output_file, prompt_template_path, model_name, resume=True):
    """
    Loop through the dataset file, and for each row find valid pairs of
    (intermediate_qids, property) where all items have a value for that property.

    For each valid pair, generate a Total Recall query using LLM.
    Each valid pair is written as a separate entry to the output file.

    Args:
        dataset_file: Path to input dataset
        output_file: Path to output file
        prompt_template_path: Path to prompt template
        model_name: Model name to use
        resume: If True, resume from last processed QID
    """
    print(f"Processing dataset: {dataset_file}")
    print(f"Using model: {model_name}")

    # Check if we should resume from a previous run
    last_processed_qid = None
    if resume:
        last_processed_qid = get_last_processed_qid(output_file)
        if last_processed_qid:
            print(f"Resuming from last processed QID: {last_processed_qid}")
        else:
            print("Starting fresh (no previous progress found)")

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    prompt_template = read_text_from_file(prompt_template_path)

    valid_pairs_count = 0
    total_rows = 0
    skipped_rows = 0
    found_last_qid = False if last_processed_qid else True

    # Open output file in append mode if resuming, otherwise write mode
    file_mode = 'a' if (resume and last_processed_qid) else 'w'

    with open(dataset_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, file_mode, encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            try:
                sample = json.loads(line)
                current_qid = sample.get('qid')

                # Skip rows until we find the last processed QID
                if not found_last_qid:
                    if current_qid == last_processed_qid:
                        found_last_qid = True
                        print(f"Found last processed QID at line {line_num}, resuming from next...")
                    skipped_rows += 1
                    continue

                total_rows += 1

                # Get intermediate QIDs and aggregatable properties from unified format
                intermediate_qids = sample.get('intermediate_qids', [])
                aggregatable_properties = sample.get('aggregationableProperties', [])

                # Filter out None values from intermediate_qids
                intermediate_qids = [qid for qid in intermediate_qids if qid is not None]

                if not intermediate_qids or not aggregatable_properties:
                    print(f"\n[Row {line_num}] No intermediate QIDs or properties, skipping")
                    continue

                print(f"\n[Row {line_num}] Processing {len(intermediate_qids)} items with {len(aggregatable_properties)} properties")

                # Get extra info from unified format
                extra = sample.get('extra', {})

                # Try each property to find valid pairs
                for prop in aggregatable_properties:
                    property_id = prop.get('property_id')
                    property_label = prop.get('property_label')

                    if not property_id:
                        continue

                    print(f"  Checking property {property_id} ({property_label})...", end=" ")

                    # Get property values for all intermediate items
                    property_values = get_property_values_for_items(intermediate_qids, property_id)

                    if property_values is not None:
                        # All items have values for this property - valid pair!
                        print("✓ Valid pair found!")

                        # Get property description from SPARQL
                        print(f"    Fetching property description...", end=" ")
                        property_description = get_property_description(property_id)
                        if property_description:
                            print("✓")
                        else:
                            print("✗ (empty)")

                        # Get entity type for the prompt from extra dict
                        entity_type_info = extra.get('intermediate_qids_instances_of', {})
                        entity_type = entity_type_info.get('label', 'entity') if entity_type_info else 'entity'

                        # Generate Total Recall query using LLM
                        print(f"    Generating Total Recall query...", end=" ")
                        generation_result = generate_total_recall_query(
                            client=client,
                            prompt_template=prompt_template,
                            original_query=sample.get('query', ''),
                            property_label=property_label,
                            property_id=property_id,
                            datatype=prop.get('datatype', ''),
                            entity_type=entity_type,
                            num_entities=len(intermediate_qids),
                            model_name=model_name
                        )

                        if generation_result and isinstance(generation_result, dict):
                            print("✓")
                            generated_query = generation_result['question']
                            aggregation_function = generation_result['aggregation']
                        else:
                            print("✗ Failed to generate query")
                            continue

                        # Calculate the answer based on aggregation function
                        print(f"    Calculating answer using {aggregation_function}...", end=" ")
                        calculated_answer = calculate_answer(
                            property_values=property_values,
                            aggregation_function=aggregation_function,
                            datatype=prop.get('datatype', '')
                        )

                        if calculated_answer is not None:
                            print("✓")
                        else:
                            print("✗ Failed to calculate")
                            # Continue anyway, but mark as None
                            calculated_answer = None

                        # Create the new property object with updated format
                        property_obj = {
                            "property_id": property_id,
                            "property_label": property_label,
                            "property_description": property_description
                        }

                        entry = {
                            "qid": f"{sample.get('qid')}_{property_id.lower()}",
                            "original_query": sample.get('query'),
                            "total_recall_query": generated_query,
                            "total_recall_answer": calculated_answer,
                            "aggregation_function": aggregation_function,
                            "property": property_obj,
                            "total_recall_qids": intermediate_qids
                        }

                        # Write entry to output file
                        f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        f_out.flush()  # Ensure it's written immediately

                        valid_pairs_count += 1
                    else:
                        print("✗ Not all items have values")

                    # Be respectful to Wikidata's servers - add delay between requests
                    time.sleep(0.5)

            except json.JSONDecodeError as e:
                print(f"\n[Row {line_num}] Error parsing JSON: {e}")
                continue
            except Exception as e:
                print(f"\n[Row {line_num}] Error processing sample: {e}")
                continue

    print(f"\n\nProcessing complete!")
    if skipped_rows > 0:
        print(f"Skipped rows (already processed): {skipped_rows}")
    print(f"Total rows processed: {total_rows}")
    print(f"Total valid pairs found: {valid_pairs_count}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Total Recall queries from valid item-property pairs")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="corpus_datasets/dataset_creation_heydar/qald10/wikidata_totallist_with_properties.jsonl",
        help="Path to the input dataset file with properties"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="corpus_datasets/dataset_creation_heydar/qald10/wikidata_total_recall_queries.jsonl",
        help="Path to the output file for generated queries"
    )
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        default="c1_2_dataset_creation_heydar/qald10/prompts/query_generation_v1.txt",
        help="Path to the prompt template file"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/gpt-4o",
        help="Model name to use for query generation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last processed QID (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start fresh, overwrite output file"
    )

    args = parser.parse_args()

    process_dataset_for_valid_pairs(
        args.dataset_file,
        args.output_file,
        args.prompt_template_path,
        args.model_name_or_path,
        args.resume
    )

    # Usage:
    # python c1_2_dataset_creation_heydar/qald10/3_query_generation.py
    # python c1_2_dataset_creation_heydar/qald10/3_query_generation.py --model_name_or_path anthropic/claude-3.5-sonnet
    # python c1_2_dataset_creation_heydar/qald10/3_query_generation.py --no-resume  # Start fresh
