import argparse
from utils.sparql_utils import *
from utils.prop_utils import NO_AGGREGATION_PROPS, INTERNAL_WIKI_PROPS
from utils.io_utils import write_json_to_file



def fetch_candidates(args):
    classes = get_classes_with_single_quantity()
    print(f"Total classes with single quantity property: {len(classes)}")

    classes_with_quantity = [subclass for subclass in classes if subclass['quantity'].isdigit()]
    print(f"Classes with numeric quantity property: {len(classes_with_quantity)}")
    
    print("Counting instances...")
    classes_with_count = add_instance_counts(classes_with_quantity)
    classes_multi_instance = [subclass for subclass in classes_with_count if subclass['instance_count'] > 1]
    print(f"Classes with more than one instance: {len(classes_multi_instance)}")
    classes_instance_limit = [subclass for subclass in classes_multi_instance if subclass['instance_count'] <= args.max_instance_count]
    print(f"Classes with instance count <= 100: {len(classes_instance_limit)}")

    classes_complete = [subclass for subclass in classes_instance_limit if subclass['instance_count'] == int(subclass['quantity'])]
    print(f"Classes where instance count matches quantity property: {len(classes_complete)}")

    print("Fetching instances...")
    for subclass in tqdm.tqdm(classes_complete):
        instances = get_instances_of_class(subclass["id"], limit=100)
        subclass['instances'] = instances
        subclass['num_en_wikipedia_links'] = sum([inst.get('wikipedia', None) is not None for inst in instances])

    classes_complete_en_wiki = [subclass for subclass in classes_complete if subclass['num_en_wikipedia_links']==subclass['instance_count']]
    print(f"Classes with all instances linked to English Wikipedia: {len(classes_complete_en_wiki)}")

    print("Fetching shared properties...")
    for subclass in tqdm.tqdm(classes_complete_en_wiki):
        properties = get_properties_of_subclass(subclass["id"])
        filtered_properties = [prop for prop in properties if prop['property_id'] not in INTERNAL_WIKI_PROPS]

        subclass['shared_properties'] = []
        subclass['property_values'] = {}
        for prop in filtered_properties:
            if prop['count'] == subclass['instance_count']: # ensure each property has a single value for each instance
                if is_property_fully_populated(subclass["id"], prop["property_id"]):
                    values = get_entity_property_values(subclass["id"], prop["property_id"])
                    # need at least 2 different values for an attribute
                    all_values = set([value['value'] for value in values])
                    if len(all_values) < 2:
                        continue
                    for value in values:
                        if isinstance(value['value'], str) and 'wikidata' in value['value']:
                            value['value'] = get_label_and_description(value['value'].split("/")[-1]) # link to another entity
                    # skip attributes linking to entities without labels
                    if any(isinstance(value['value'],dict) and is_wikidata_id(value['value']['label']) for value in values):
                        continue
                    subclass['shared_properties'].append(prop)
                    subclass['property_values'][prop['property_id']] = values

    query_generation_candidates = []
    for subclass in classes_complete_en_wiki:
        can_be_aggregated = [prop['property_id'] not in NO_AGGREGATION_PROPS for prop in subclass['shared_properties']]
        if any(can_be_aggregated):
            subclass['shared_properties'] = [prop for prop in subclass['shared_properties'] if prop['property_id'] not in NO_AGGREGATION_PROPS]
            query_generation_candidates.append(subclass)

    write_json_to_file(args.output_file, query_generation_candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="c3_dataset_augmentation/mahta_code/outputs/results.jsonl", help="Output file location")
    parser.add_argument("--max_instance_count", type=int, default=100, help="Maximum instance count")
    args = parser.parse_args()
    
    fetch_candidates(args)
    

# python c3_dataset_augmentation/mahta_code/get_query_candidates.py
    
    