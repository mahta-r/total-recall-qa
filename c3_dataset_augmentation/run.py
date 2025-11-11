import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse

from c3_dataset_augmentation.mahta_code.utils.sparql_utils import *
from c3_dataset_augmentation.mahta_code.utils.prop_utils import NO_AGGREGATION_PROPS, INTERNAL_WIKI_PROPS
from c3_dataset_augmentation.utils.general_utils import (
    get_properties_of_item,
    get_property_values,
    get_entity_info
)

def fetch_candidates(args):
    
    results = {}
    if os.path.exists(args.dataset_file):
        with open(args.dataset_file, 'r', encoding='utf-8') as in_f:
            for idx, line in enumerate(tqdm.tqdm(in_f)):
                if idx == 5:
                    break
                
                sample = json.loads(line)
                file_id, qid, query, instances = sample['file_id'], sample['qid'], sample['query'], sample['intermidate_list']
                
                if any(x is None for x in instances):
                    print(f"sample {qid} has been skiped due to the None item!")
                    continue
            
                # getting instances value
                instances_val = get_entity_info(instances)
            
                # Fetching shared properties...
                properties_obj = {}
                for entity_item in instances:
                    properties = get_properties_of_item(entity_item)
                    filtered_properties = [prop for prop in properties if prop['property_id'] not in INTERNAL_WIKI_PROPS]
                    properties_obj[entity_item] = filtered_properties
    
                shared_ids = set.intersection(*[
                    set(p["property_id"] for p in props)
                    for props in properties_obj.values()
                ])
                shared_props_ = {}
                for props in properties_obj.values():
                    for p in props:
                        pid = p["property_id"]
                        if pid in shared_ids and pid not in shared_props_:
                            shared_props_[pid] = p  # store full object
                shared_props = list(shared_props_.values())
                
                # Fetching the property values 
                property_values = get_property_values(instances, shared_ids)
                
                results[qid] = {
                    "file_id": file_id,
                    "qid": qid,
                    "query": query,
                    "instance_count": len(instances_val),
                    "instances": instances_val,
                    "shared_properties": shared_props,
                    "property_values": property_values
                }
    
    
    output_path = "corpus_datasets/qald_aggregation_samples/properties_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)         
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_file = "corpus_datasets/qald_aggregation_samples/wikidata_totallist.jsonl"
    
    fetch_candidates(args)
    
    
    # python c3_dataset_augmentation/run.py

