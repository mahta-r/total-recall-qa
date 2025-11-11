import os
import io
import random
import math
import tqdm
import argparse
from collections import Counter
from openai import OpenAI

from io_utils import read_json_from_file, write_json_to_file, read_text_from_file
from wikidata.prop_utils import prop_type_mapping, prop_operation_mapping



def select_number_properties(properties, num_props, generated_types, prop_type_mapping, prop_operation_mapping):
    numerical_props = [prop for prop in properties if prop_type_mapping[prop['property_id']] == 'number']
    
    random.shuffle(numerical_props)

    if len(numerical_props) <= num_props:
        selected_props = numerical_props
    else:
        selected_props = sorted(numerical_props, key=lambda p: len(generated_types.get(p['property_id'], [])))[:num_props]

    selected_ops = []
    for prop in selected_props:
        generated_ops = Counter(generated_types.get(prop['property_id'], []))
        possible_ops = prop_operation_mapping[prop['property_id']]

        random.shuffle(possible_ops)

        selected_ops.append(sorted(possible_ops, key=lambda op: generated_ops.get(op, 0))[0])

    for prop, op in zip(selected_props, selected_ops):
        if prop['property_id'] not in generated_types:
            generated_types[prop['property_id']] = []
        generated_types[prop['property_id']].append(op)

    return selected_props, selected_ops


def select_properties(properties, selection = "all", seed = 42):
    if selection == "all":
        return properties
    elif selection == "random_sample":
        random.seed(seed)
        num_props = math.floor(math.log(len(properties), 2)) + 1
        return random.sample(properties, num_props)


def generate_queries(args):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    query_generation_candidates = read_json_from_file(args.query_candidates_path)
    prompt_template = read_text_from_file(args.prompt_template_path)
    generations = []

    with open(args.log_file, 'w', encoding='utf-8') as log:
        for subclass in tqdm.tqdm(query_generation_candidates):
            properties = subclass['shared_properties']
            if not properties:
                continue
            
            # select a subset of properties to generate queries for
            selected_properties = select_properties(properties, selection="random_sample")

            for prop in selected_properties:
                example_for_prompt = io.StringIO()    
                
                print(f"set of entities: {subclass['label']} ({subclass['id']}): {subclass['description']}", file=example_for_prompt)
                print(f"attribute: {prop['label']} ({prop['property_id']}, {prop['description']})", file=example_for_prompt)
                print(f"{prop['count']} instances:", file=example_for_prompt)
                
                attribute_values = subclass['property_values'][prop['property_id']]
                for record in attribute_values:
                    if isinstance(record['value'], dict):
                        value = f"{record['value']['id']}: {record['value']['label']} ({record['value']['description']})"
                    else:
                        value = record['value']
                    print(f"  {record['entity_id']}: {record['entity_label']} -> {value}", file=example_for_prompt)

                prompt_inputs = example_for_prompt.getvalue()
                example_for_prompt.close()

                resp = client.chat.completions.create(
                    model=args.model, 
                    messages=[
                        {"role": "system", "content": "You are a helpful query generation assistant."},
                        {"role": "user", "content": prompt_template.format(inputs=prompt_inputs)}
                    ],
                    temperature=args.temperature,
                )

                print("PROMPT:\n", prompt_inputs, file=log)
                print(resp.choices[0].message.content.strip(), file=log)
                print("\n--------------------------------------\n", file=log)
                
                generations.append(resp.to_dict())

        write_json_to_file(args.output_file, generations)

        
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_candidates_path", type=str, required=True, help="Path to wikidata query candidates")
    parser.add_argument("--property_selection", type=str, default="all", help="Selection method for properties: all, random_sample")
    parser.add_argument("--prompt_template_path", type=str, required=True, help="Path to prompt template file")
    parser.add_argument("--model", type=str, required=True, help="Model to use for generation e.g gpt-4o-mini, gpt-4o, etc.")    
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file location for generations")
    parser.add_argument("--log_file", type=str, required=True, help="Log file location")

    args = parser.parse_args()
    
    generate_queries(args) 