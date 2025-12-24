import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import io
import random
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from openai import OpenAI
import tqdm

try:
    from c1_2_dataset_creation_heydar.utils.operation_utils import operation_descriptions, apply_operation, get_operations_for_datatype
    from c1_2_dataset_creation_heydar.utils.parse_generations import extract_query_text, extract_aggregation
    from c1_2_dataset_creation_heydar.prompts.query_generation_v2 import QUERY_GENERATION_PROMPT
except:
    from utils.operation_utils import operation_descriptions, apply_operation, get_operations_for_datatype
    from utils.parse_generations import extract_query_text, extract_aggregation
    from prompts.query_generation_v2 import QUERY_GENERATION_PROMPT

from datetime import datetime
from statistics import mean
from collections import Counter
import math


class PropertySelector:
    """
    Intelligent property selection and operation balancing based on Mahta's method.
    Provides strategies for selecting diverse properties and balanced operations.
    """

    def __init__(self, property_num="all", selection_strategy="random", max_props=None):
        """
        Initialize the property selector.

        Args:
            property_num: Strategy for number of properties to select ("all" or "log")
            selection_strategy: How to prioritize properties ("random" or "least")
            max_props: Maximum number of properties to select per query (None = unlimited)
        """
        self.property_num = property_num
        self.selection_strategy = selection_strategy
        self.max_props = max_props

        # Track usage across all queries
        self.used_property_operation_combos = {}
        self.all_props_usage = Counter()

    def select_properties(self, properties):
        """
        Select a subset of properties based on the configured strategy.

        Args:
            properties: List of property dictionaries with 'property_id' key

        Returns:
            List of selected property dictionaries
        """
        if not properties:
            return []

        # Determine how many properties to select
        if self.property_num == "all":
            num_to_select = len(properties)
        elif self.property_num == "log":
            num_to_select = math.floor(math.log(len(properties), 2)) + 1
        else:
            raise ValueError(f"Invalid property_num strategy: {self.property_num}")

        # Apply max_props limit if specified
        if self.max_props is not None:
            num_to_select = min(num_to_select, self.max_props)

        # Select properties based on strategy
        if self.selection_strategy == "random":
            selected_properties = random.sample(properties, min(num_to_select, len(properties)))
        elif self.selection_strategy == "least":
            # Prioritize properties that have been used least across all queries
            def property_scarcity(prop):
                prop_id = prop.get('property_id')
                return self.all_props_usage.get(prop_id, 0)

            sorted_properties = sorted(properties, key=property_scarcity)
            selected_properties = sorted_properties[:num_to_select]
        else:
            raise ValueError(f"Invalid selection_strategy: {self.selection_strategy}")

        # Track property usage
        for prop in selected_properties:
            prop_id = prop.get('property_id')
            if prop_id:
                self.all_props_usage[prop_id] += 1
                if prop_id not in self.used_property_operation_combos:
                    self.used_property_operation_combos[prop_id] = {}

        return selected_properties

    def select_operation(self, property_id, datatype):
        """
        Select an operation for a property, balancing usage across operations.

        Args:
            property_id: The property ID
            datatype: The property datatype

        Returns:
            Selected operation string
        """
        # Get valid operations for this datatype
        valid_ops = get_operations_for_datatype(datatype)

        if not valid_ops:
            return None

        # Initialize tracking for this property if needed
        if property_id not in self.used_property_operation_combos:
            self.used_property_operation_combos[property_id] = {}

        # Get operation counts for this property
        op_counts = [(op, self.used_property_operation_combos[property_id].get(op, 0))
                     for op in valid_ops]

        # Sort by usage count (least-used first)
        op_counts_sorted = sorted(op_counts, key=lambda x: x[1])
        min_count = op_counts_sorted[0][1]

        # Select randomly from least-used operations
        least_used_ops = [op for op, count in op_counts_sorted if count == min_count]
        selected_op = random.choice(least_used_ops)

        # Update usage count
        self.used_property_operation_combos[property_id][selected_op] = \
            self.used_property_operation_combos[property_id].get(selected_op, 0) + 1

        return selected_op

    def get_usage_stats(self):
        """
        Get statistics about property and operation usage.

        Returns:
            Dictionary with usage statistics
        """
        total_props_used = sum(self.all_props_usage.values())
        unique_props_used = len(self.all_props_usage)

        return {
            "total_property_uses": total_props_used,
            "unique_properties_used": unique_props_used,
            "property_usage": dict(self.all_props_usage.most_common()),
            "operation_combos": dict(self.used_property_operation_combos)
        }


def parse_value_for_aggregation(value_data, datatype):
    """
    Parse a value from Wikidata into a numeric form for aggregation.

    Args:
        value_data: Value dictionary from Wikidata
        datatype: Property datatype (Time, Quantity, etc.)

    Returns:
        Numeric value or None
    """
    try:
        if datatype == 'Time':
            # Parse dates and extract years
            val = value_data.get('value', '')
            if isinstance(val, str) and 'T' in val:
                dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                return dt.year
        elif datatype == 'Quantity':
            # Extract numeric values
            val = value_data.get('value', '')
            if isinstance(val, (int, float)):
                return float(val)
            elif isinstance(val, str):
                # Try to extract number from string (remove + prefix if present)
                num_str = val.lstrip('+').split()[0]
                return float(num_str)
    except Exception as e:
        return None

    return None


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

    # Extract all values and parse them
    numeric_values = []
    for item_values in property_values.values():
        for val in item_values:
            parsed_val = parse_value_for_aggregation(val, datatype)
            if parsed_val is not None:
                numeric_values.append(parsed_val)

    if not numeric_values:
        return None

    try:
        return apply_operation(aggregation_function, numeric_values)
    except Exception as e:
        print(f"      Error calculating answer: {e}")
        return None


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
    sparql.setTimeout(30)

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
        results = sparql.query().convert()

        entity_labels = {}
        for result in results["results"]["bindings"]:
            item_uri = result["item"]["value"]
            item_id = item_uri.split("/")[-1]
            entity_labels[item_id] = result["itemLabel"]["value"]

        return entity_labels
    except Exception as e:
        print(f"  Error querying entity labels: {e}")
        return {}


def generate_total_recall_query(client, model_name, temperature, original_query, entity_class_label,
                                  entity_class_id, entity_class_description, property_label, property_id,
                                  property_description, datatype, operation, entity_values, instance_count,
                                  unit_label=None, unit_id=None, point_in_time=None):
    """
    Generate a new Total Recall query using LLM with detailed context.

    Args:
        client: OpenAI client instance
        model_name: Name of the model to use
        temperature: Temperature for generation
        original_query: The original query from the dataset
        entity_class_label: Label of the entity class
        entity_class_id: ID of the entity class
        entity_class_description: Description of the entity class
        property_label: Label of the property
        property_id: ID of the property
        property_description: Description of the property
        datatype: Datatype of the property
        operation: Aggregation operation to use
        entity_values: List of dicts with entity_id, entity_label, and value
        instance_count: Total number of instances
        unit_label: Optional unit label
        unit_id: Optional unit ID
        point_in_time: Optional point in time

    Returns:
        Response object with completion
    """
    # Build the prompt inputs with original query at the top
    prompt_inputs_buffer = io.StringIO()
    print(f"original-query: {original_query}", file=prompt_inputs_buffer)
    print(f"class (set of entities): {entity_class_label} ({entity_class_id}) - {entity_class_description}", file=prompt_inputs_buffer)
    print(f"attribute: {property_label} ({property_id}) - {property_description}", file=prompt_inputs_buffer)

    if unit_id is not None and unit_id != "Q199":  # excluding unit "1" (Q199)
        print(f"unit: {unit_label} ({unit_id})", file=prompt_inputs_buffer)

    if point_in_time is not None:
        print(f"point-in-time: {point_in_time}", file=prompt_inputs_buffer)

    print(f"aggregation-operation: {operation} - {operation_descriptions.get(operation, '')}", file=prompt_inputs_buffer)
    print(f"entity-values: ({instance_count} instances):", file=prompt_inputs_buffer)

    for record in entity_values:
        print(f"  {record['entity_id']}: {record['entity_label']} -> {record['value']}", file=prompt_inputs_buffer)

    prompt_inputs = prompt_inputs_buffer.getvalue()
    prompt_inputs_buffer.close()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful query generation assistant."},
                {"role": "user", "content": QUERY_GENERATION_PROMPT.format(inputs=prompt_inputs)}
            ],
            temperature=temperature,
        )

        return response, prompt_inputs

    except Exception as e:
        print(f"    Error generating query with LLM: {e}")
        return None, prompt_inputs


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


def process_dataset_for_valid_pairs(dataset_file, output_file, queries_file, log_file, model_name, temperature, seed,
                                     property_num="all", selection_strategy="random", max_props=None, resume=True):
    """
    Loop through the dataset file, and for each row find valid pairs of
    (intermediate_qids, property) where all items have a value for that property.

    For each valid pair, generate a Total Recall query using LLM.
    Each valid pair is written as a separate entry to the output file.

    Args:
        dataset_file: Path to input dataset
        output_file: Path to output file for full generations
        queries_file: Path to output file for parsed queries
        log_file: Path to log file
        model_name: Model name to use
        temperature: Temperature for generation
        seed: Random seed
        property_num: Property selection strategy ("all" or "log")
        selection_strategy: Property prioritization strategy ("random" or "least")
        max_props: Maximum properties to select per query (None = unlimited)
        resume: If True, resume from last processed QID
    """
    random.seed(seed)
    print(f"Processing dataset: {dataset_file}")
    print(f"Using model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Random seed: {seed}")
    print(f"Property selection: {property_num} (strategy: {selection_strategy}, max: {max_props})")

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

    # Initialize PropertySelector for intelligent property selection
    property_selector = PropertySelector(
        property_num=property_num,
        selection_strategy=selection_strategy,
        max_props=max_props
    )

    generations = []
    queries = []
    valid_pairs_count = 0
    total_rows = 0
    skipped_rows = 0
    found_last_qid = False if last_processed_qid else True

    # Open output files
    file_mode = 'a' if (resume and last_processed_qid) else 'w'

    with open(dataset_file, 'r', encoding='utf-8') as f_in, \
         open(log_file, file_mode, encoding='utf-8') as log:

        for line_num, line in tqdm.tqdm(enumerate(f_in, 1), desc="Processing queries"):
            
            if line_num == 3:
                break
            
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
                    continue

                # Get extra info from unified format
                extra = sample.get('extra', {})
                entity_type_info = extra.get('intermediate_qids_instances_of', {})
                entity_class_label = entity_type_info.get('label', 'entity') if entity_type_info else 'entity'
                entity_class_id = entity_type_info.get('id', '') if entity_type_info else ''
                entity_class_description = entity_type_info.get('description', '') if entity_type_info else ''

                # Get entity labels once for all properties
                print(f"\n[Query {line_num}] Fetching entity labels for {len(intermediate_qids)} items...", end=" ")
                entity_labels = get_entity_labels(intermediate_qids)
                print("✓" if entity_labels else "✗")

                # Select properties using intelligent selection strategy
                selected_properties = property_selector.select_properties(aggregatable_properties)
                print(f"  Selected {len(selected_properties)}/{len(aggregatable_properties)} properties for this query")

                # Try each selected property to find valid pairs
                for prop in selected_properties:
                    property_id = prop.get('property_id')
                    property_label = prop.get('property_label')
                    property_description = prop.get('property_description', '')
                    datatype = prop.get('datatype', '')

                    if not property_id:
                        continue

                    print(f"  Property {property_id} ({property_label})...", end=" ")

                    # Get property values for all intermediate items
                    property_values = get_property_values_for_items(intermediate_qids, property_id)

                    if property_values is None:
                        print("✗ Not all items have values")
                        continue

                    print("✓ Valid pair")

                    # Select operation using intelligent balancing
                    operation = property_selector.select_operation(property_id, datatype)
                    if not operation:
                        print("    ✗ No valid operations for this datatype")
                        continue

                    # Build entity_values list for prompt
                    entity_values_list = []
                    for qid in intermediate_qids:
                        qid_values = property_values.get(qid, [])
                        if qid_values:
                            parsed_val = parse_value_for_aggregation(qid_values[0], datatype)
                            if parsed_val is not None:
                                entity_values_list.append({
                                    'entity_id': qid,
                                    'entity_label': entity_labels.get(qid, qid),
                                    'value': parsed_val
                                })

                    if not entity_values_list:
                        print("    ✗ No valid values for aggregation")
                        continue

                    # Calculate ground truth
                    ground_truth = calculate_answer(property_values, operation, datatype)

                    if ground_truth is None:
                        print("    ✗ Failed to calculate ground truth")
                        continue

                    # Generate query using LLM
                    print(f"    Generating query with {operation}...", end=" ")
                    response, prompt_inputs = generate_total_recall_query(
                        client=client,
                        model_name=model_name,
                        temperature=temperature,
                        original_query=sample.get('query', ''),
                        entity_class_label=entity_class_label,
                        entity_class_id=entity_class_id,
                        entity_class_description=entity_class_description,
                        property_label=property_label,
                        property_id=property_id,
                        property_description=property_description,
                        datatype=datatype,
                        operation=operation,
                        entity_values=entity_values_list,
                        instance_count=len(intermediate_qids),
                        unit_label=None,  # TODO: extract from property values
                        unit_id=None,
                        point_in_time=None  # TODO: extract from property values
                    )

                    if response is None:
                        print("✗ Failed")
                        continue

                    print("✓")

                    # Log the prompt and response
                    print("PROMPT:\n" + prompt_inputs, file=log)
                    print(response.choices[0].message.content.strip(), file=log)
                    print(f"GROUND TRUTH VALUE: {ground_truth}", file=log)
                    print("\n---------------------------------------------------------------------------------------------------\n", file=log)

                    # Extract query text
                    generated_text = response.choices[0].message.content.strip()
                    query_text = extract_query_text(generated_text)

                    if not query_text:
                        print("    ✗ Failed to extract query text")
                        continue

                    # Create generation record
                    generation = {
                        "qid": f"{current_qid}_{property_id.lower()}",
                        "original_query": sample.get('query', ''),
                        "subclass_id": entity_class_id,
                        "subclass_label": entity_class_label,
                        "subclass_description": entity_class_description,
                        "property_id": property_id,
                        "property": {
                            "property_info": {
                                "label": property_label,
                                "property_id": property_id,
                                "description": property_description,
                                "datatype": datatype
                            },
                            "entities_values": entity_values_list
                        },
                        "operation": operation,
                        "ground_truth": ground_truth,
                        "completion": response.to_dict()
                    }

                    # Create query record
                    query = {
                        "id": f"{current_qid}_{property_id.lower()}",
                        "question": query_text,
                        "answer": {
                            "type": datatype,
                            "value": ground_truth,
                        }
                    }

                    generations.append(generation)
                    queries.append(query)
                    valid_pairs_count += 1

                    # Be respectful to Wikidata's servers
                    time.sleep(0.5)

            except json.JSONDecodeError as e:
                print(f"\n[Row {line_num}] Error parsing JSON: {e}")
                continue
            except Exception as e:
                print(f"\n[Row {line_num}] Error processing sample: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Get usage statistics from PropertySelector
    usage_stats = property_selector.get_usage_stats()

    # Write output files
    print("\nWriting output files...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for gen in generations:
            f_out.write(json.dumps(gen, ensure_ascii=False) + '\n')

    with open(queries_file, 'w', encoding='utf-8') as f_out:
        for query in queries:
            f_out.write(json.dumps(query, ensure_ascii=False) + '\n')

    # Write usage statistics to log file
    with open(log_file, 'a', encoding='utf-8') as log:
        print("\n" + "="*80, file=log)
        print("PROPERTY SELECTION STATISTICS", file=log)
        print("="*80, file=log)
        print(f"Total property uses: {usage_stats['total_property_uses']}", file=log)
        print(f"Unique properties used: {usage_stats['unique_properties_used']}", file=log)
        print(f"\nProperty usage counts:", file=log)
        print(json.dumps(usage_stats['property_usage'], indent=2), file=log)
        print(f"\nOperation combinations:", file=log)
        print(json.dumps(usage_stats['operation_combos'], indent=2), file=log)

    print(f"\n\nProcessing complete!")
    if skipped_rows > 0:
        print(f"Skipped rows (already processed): {skipped_rows}")
    print(f"Total rows processed: {total_rows}")
    print(f"Total valid pairs found: {valid_pairs_count}")
    print(f"Unique properties used: {usage_stats['unique_properties_used']}")
    print(f"Generations saved to: {output_file}")
    print(f"Queries saved to: {queries_file}")
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Total Recall queries from valid item-property pairs")
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to the input dataset file with properties"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file for full generation records"
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        required=True,
        help="Path to the output file for parsed queries"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to the log file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name to use for query generation (e.g., gpt-4o-mini, gpt-4o)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--property_num",
        type=str,
        default="all",
        choices=["all", "log"],
        help="Strategy for number of properties to select per query: 'all' (use all properties) or 'log' (logarithmic selection)"
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="random",
        choices=["random", "least"],
        help="Property selection strategy: 'random' (random selection) or 'least' (prioritize least-used properties)"
    )
    parser.add_argument(
        "--max_props",
        type=int,
        default=None,
        help="Maximum number of properties to select per query (None = unlimited)"
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
        dataset_file=args.dataset_file,
        output_file=args.output_file,
        queries_file=args.queries_file,
        log_file=args.log_file,
        model_name=args.model,
        temperature=args.temperature,
        seed=args.seed,
        property_num=args.property_num,
        selection_strategy=args.selection_strategy,
        max_props=args.max_props,
        resume=args.resume
    )

    # Usage Examples:
    #
    # 1. Default (use all properties, random selection, random operations):
    # python c1_2_dataset_creation_heydar/3_query_generation.py \
    #     --dataset_file corpus_datasets/dataset_creation_heydar/quest/test_quest_with_properties.jsonl \
    #     --output_file corpus_datasets/dataset_creation_heydar/quest/generations.jsonl \
    #     --queries_file corpus_datasets/dataset_creation_heydar/quest/queries.jsonl \
    #     --log_file corpus_datasets/dataset_creation_heydar/quest/query_generation.log \
    #     --model gpt-4o-mini \
    #     --temperature 0.7 \
    #     --seed 42
    #
    # 2. Logarithmic selection with max 5 properties, prioritize rare properties:
    # python c1_2_dataset_creation_heydar/3_query_generation.py \
    #     --dataset_file corpus_datasets/dataset_creation_heydar/quest/test_quest_with_properties.jsonl \
    #     --output_file corpus_datasets/dataset_creation_heydar/quest/generations.jsonl \
    #     --queries_file corpus_datasets/dataset_creation_heydar/quest/queries.jsonl \
    #     --log_file corpus_datasets/dataset_creation_heydar/quest/query_generation.log \
    #     --model gpt-4o-mini \
    #     --property_num log \
    #     --selection_strategy least \
    #     --max_props 5
    #
    # 3. Limit to 3 properties per query with balanced operations:
    # python c1_2_dataset_creation_heydar/3_query_generation.py \
    #     --dataset_file corpus_datasets/dataset_creation_heydar/quest/test_quest_with_properties.jsonl \
    #     --output_file corpus_datasets/dataset_creation_heydar/quest/generations.jsonl \
    #     --queries_file corpus_datasets/dataset_creation_heydar/quest/queries.jsonl \
    #     --log_file corpus_datasets/dataset_creation_heydar/quest/query_generation.log \
    #     --model gpt-4o-mini \
    #     --max_props 3
