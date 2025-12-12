import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
from c1_2_qald_dataset_augmentation.mahta_code.utils.sparql_utils import *
# from c1_2_qald_dataset_augmentation.mahta_code.utils.prop_utils import NO_AGGREGATION_PROPS, INTERNAL_WIKI_PROPS
from c1_2_qald_dataset_augmentation.utils.general_utils import (
    get_properties_of_item,
    get_property_values,
    get_entity_info
)
import re
from datetime import datetime


def is_numeric_value(v):
    """Returns True if value looks numeric."""
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        return re.match(r"^-?\d+(\.\d+)?$", v.strip()) is not None
    return False

def is_datetime_value(v):
    """Returns True if string can be parsed as date."""
    if isinstance(v, str):
        try:
            datetime.fromisoformat(v.replace("Z",""))
            return True
        except:
            return False
    return False

def looks_numeric_property(p):
    """Check numeric-like datatype or constraints."""
    dt = p.get("datatype", "").lower()

    # Primary Wikidata numeric datatypes
    numeric_like = {"quantity", "time", "globe-coordinate", "external-id"}
    if dt in numeric_like:
        return True

    # Property constraints (if available)
    constraints = p.get("constraints", [])
    signal_constraints = {
        "allowed units", "range constraint",
        "integer constraint", "value type constraint"
    }
    if any(c.lower() in str(constraints).lower() for c in signal_constraints):
        return True

    # Heuristic: check sample values
    values = p.get("values", [])
    if values:
        numeric_count = sum(is_numeric_value(v) for v in values)
        time_count    = sum(is_datetime_value(v) for v in values)

        # More than half numeric-like? → aggregatable
        if numeric_count >= len(values) * 0.5:
            return True
        
        # More than half time-like? → aggregatable
        if time_count >= len(values) * 0.5:
            return True

    return False

def filter_aggregatable_properties(shared_props):
    """Return only properties where aggregation is possible."""
    return [p for p in shared_props if looks_numeric_property(p)]



def fetch_candidates(args):
    numeric_props = []
    with open("corpus_datasets/wikidata_numeric_properties.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            numeric_props.append(item['id'])
    
    itemlist_props = []
    with open("corpus_datasets/wikidata_itemlist_properties.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            itemlist_props.append(item['id'])
    
    results = {}
    if os.path.exists(args.dataset_file):
        with open(args.dataset_file, 'r', encoding='utf-8') as in_f:
            for idx, line in enumerate(tqdm.tqdm(in_f)):
                # if idx == 30:
                #     break
                
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
                    # filtered_properties = [prop for prop in properties if prop['property_id'] not in INTERNAL_WIKI_PROPS]
                    # properties_obj[entity_item] = filtered_properties
                    properties_obj[entity_item] = properties
    
                # shared_ids = set.intersection(*[
                #     set(p["property_id"] for p in props)
                #     for props in properties_obj.values()
                # ])
                sets = [
                    set(p["property_id"] for p in props)
                    for props in properties_obj.values()
                ]
                if not sets:
                    print(f"No properties found for sample {qid}, skipping.")
                    continue
                shared_ids = sets[0].intersection(*sets[1:])

                shared_props_ = {}
                for props in properties_obj.values():
                    for p in props:
                        pid = p["property_id"]
                        if pid in shared_ids and pid not in shared_props_:
                            shared_props_[pid] = p
                shared_props = list(shared_props_.values())
                
                # # Filtering out aggregation properties
                # shared_props = [
                #     prop for prop in shared_props
                #     if prop['property_id'] in numeric_props
                # ]
                
                
                # filtered_shared_ids = [
                #     prop['property_id'] for prop in shared_props
                #     if prop['property_id'] in numeric_props
                # ]
                aggregatable_props = filter_aggregatable_properties(shared_props)
                aggregatable_ids = [prop['property_id'] for prop in aggregatable_props]
                
                # Fetching the property values 
                property_values = get_property_values(instances, aggregatable_ids)
                
                results[qid] = {
                    "file_id": file_id,
                    "qid": qid,
                    "query": query,
                    "instance_count": len(instances_val),
                    "instances": instances_val,
                    "aggregatable_properties": aggregatable_props,
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
    
    # python c1_2_qald_dataset_augmentation/query_augmentation.py

