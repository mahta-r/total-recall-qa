import os
import io
import random
import math
import json
from textwrap import indent
import tqdm
import argparse
from collections import Counter
from openai import OpenAI

from io_utils import read_json_from_file, write_jsonl_to_file, read_text_from_file
from wikidata.prop_utils import prop_operation_mapping
from wikidata.operation_utils import operation_descriptions, apply_operation
from query_generation.parse_generations import extract_query_text
from prompts.query_generation.prompt_v2 import QUERY_GENERATION_PROMPT



class CandidateGenerator:

    def __init__(self, query_generation_classes, prop_num, selection_strategy, max_props):
        self.query_generation_classes = query_generation_classes
        self.prop_num = prop_num
        self.selection = selection_strategy
        self.max_props = max_props

        self.used_property_operation_combos = {}

        all_props = [
            prop_id
            for subclass in self.query_generation_classes
            for prop_id in subclass['candidate_properties'].keys()
        ]
        self.all_props_available = Counter(all_props)        

    def __iter__(self):
        for subclass in self.query_generation_classes:
            properties = subclass['candidate_properties']
            
            if not properties:
                continue

            property_ids = self.select_properties(properties.keys())

            for prop_id in property_ids:
                selected_op = self.select_operation(prop_id)
                yield subclass, prop_id, selected_op


    def select_properties(self, property_ids):
        assert len(property_ids) > 0

        if self.prop_num == "all":
            num_to_select = len(property_ids)
        elif self.prop_num == "log":
            num_to_select = math.floor(math.log(len(property_ids), 2)) + 1
        else:
            raise ValueError(f"Invalid property per subclass strategy: {self.prop_num}")

        num_to_select = min(num_to_select, self.max_props)

        if self.selection == "random":
            selected_properties = random.sample(property_ids, num_to_select)
        elif self.selection == "least":
            # TODO: optimize selection
            property_count = lambda prop_id: sum(self.used_property_operation_combos.get(prop_id, {}).values())
            property_scarcity = lambda prop_id: self.all_props_available[prop_id]
            selected_properties = sorted(property_ids, key=property_scarcity)[:num_to_select]
        else:
            raise ValueError(f"Invalid property selection strategy: {self.selection}")

        for prop_id in selected_properties:
            if prop_id not in self.used_property_operation_combos:
                self.used_property_operation_combos[prop_id] = {}

        return selected_properties

    
    def select_operation(self, prop_id):
        valid_ops = prop_operation_mapping[prop_id]
        # property_operation_count = lambda op: self.used_property_operation_combos.get(prop_id, {}).get(op, 0)
        # selected_op = sorted(valid_ops, key=property_operation_count)[0]

        op_counts = [(op, self.used_property_operation_combos[prop_id].get(op, 0)) for op in valid_ops]
        op_counts_sorted = sorted(op_counts, key=lambda x: x[1])
        min_count = op_counts_sorted[0][1]
        least_used_ops = [op for op, count in op_counts_sorted if count == min_count]
        selected_op = random.choice(least_used_ops)

        if selected_op not in self.used_property_operation_combos[prop_id]:
            self.used_property_operation_combos[prop_id][selected_op] = 0
        self.used_property_operation_combos[prop_id][selected_op] += 1

        return selected_op


    

def generate_queries(args):
    random.seed(args.seed)
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    query_generation_candidates = read_json_from_file(args.query_candidates_path)
    generations = []
    queries = []


    all_props = []    
    with open(args.log_file, 'w', encoding='utf-8') as log:

        for subclass, property_id, operation in tqdm.tqdm(CandidateGenerator(
            query_generation_candidates, 
            args.property_num, 
            args.property_selection, 
            args.max_props
        )):  

            class_property_record = subclass['candidate_properties'][property_id]
            prop = class_property_record["property_info"]
            entity_values = class_property_record["entities_values"]
            point_in_time = class_property_record.get("point_in_time", None)    
            unit_id, unit_label = class_property_record.get("unit", (None, None))

            ground_truth_value = apply_operation(operation, [record['value'] for record in entity_values])

            all_props.append(prop['label'])

            prompt_inputs_buffer = io.StringIO()
            print(f"class (set of entities): {subclass['label']} ({subclass['id']}) - {subclass['description']}", file=prompt_inputs_buffer)
            print(f"attribute: {prop['label']} ({prop['property_id']}) - {prop['description']}", file=prompt_inputs_buffer)
            if unit_id is not None and unit_id != "Q199": # excluding unit "1" (Q199)
                print(f"unit: {unit_label} ({unit_id})", file=prompt_inputs_buffer)
            if point_in_time is not None:
                print(f"point-in-time: {point_in_time}", file=prompt_inputs_buffer)
            print(f"aggregation-operation: {operation} - {operation_descriptions[operation]}", file=prompt_inputs_buffer)
            print(f"entity-values: ({subclass['instance_count']} instances):", file=prompt_inputs_buffer)
            for record in entity_values:
                print(f"  {record['entity_id']}: {record['entity_label']} -> {record['value']}", file=prompt_inputs_buffer)

            prompt_inputs = prompt_inputs_buffer.getvalue()
            prompt_inputs_buffer.close()

            resp = client.chat.completions.create(
                model=args.model, 
                messages=[
                    {"role": "system", "content": "You are a helpful query generation assistant."},
                    {"role": "user", "content": QUERY_GENERATION_PROMPT.format(inputs=prompt_inputs)}
                ],
                temperature=args.temperature,
            )

            print("PROMPT:\n" + prompt_inputs, file=log)
            print(resp.choices[0].message.content.strip(), file=log)
            print(f"GROUND TRUTH VALUE: {ground_truth_value}", file=log)
            # print(f"EXTRACTED QUERY: --| {extract_query_text(resp.choices[0].message.content.strip())} |--", file=log)
            print("\n---------------------------------------------------------------------------------------------------\n", file=log)
            

            generation = {
                "subclass_id": subclass['id'],
                "subclass_label": subclass['label'],
                "subclass_description": subclass['description'],
                "property_id": property_id,
                "property": class_property_record,
                "operation": operation,
                "ground_truth": ground_truth_value,
                "completion": resp.to_dict()
            }

            query = {
                "id": f"{subclass['id']}_{property_id}",
                "question": extract_query_text(resp.choices[0].message.content.strip()),
                "answer": {
                    "type": prop["datatype"],
                    "value": ground_truth_value,
                }
            }

            generations.append(generation)
            queries.append(query)


        write_jsonl_to_file(args.output_file, generations)
        write_jsonl_to_file(args.queries_file, queries)

        prop_counts = dict(Counter(all_props).most_common())
        print(f"total class-property pairs: {sum(prop_counts.values())}", file=log)
        print(f"total unique properties used: {len(prop_counts)}", file=log)
        print(f"all properties used: {json.dumps(prop_counts, indent=2)}", file=log)
        

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_candidates_path", type=str, required=True, help="Path to wikidata query candidates")
    parser.add_argument("--property_num", type=str, default="all", help="Strategy for number of properties to select per subclass: all, log")
    parser.add_argument("--max_props", type=int, default=2, help="Maximum number of properties to select per subclass")
    parser.add_argument("--property_selection", type=str, default="all", help="Selection strategy for properties: random, least")
    
    parser.add_argument("--model", type=str, required=True, help="Model to use for generation e.g gpt-4o-mini, gpt-4o, etc.")    
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file location for generations")
    parser.add_argument("--queries_file", type=str, required=True, help="Output JSON file location for parsed queries")
    parser.add_argument("--log_file", type=str, required=True, help="Log file location")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")

    args = parser.parse_args()
    generate_queries(args) 


    # python -m query_generation.generate_v2 \
    #     --query_candidates_path data/data_creation_runs/V2/quantity_query_generation_candidates_v2_unit_time.json \
    #     --property_num all \
    #     --max_props 11 \
    #     --property_selection least \
    #     --model gpt-5-mini \
    #     --log_file data/generations/V2/query_generation.log \
    #     --output_file data/generations/V2/generations.json \
    #     --queries_file data/generations/V2/queries.json
    