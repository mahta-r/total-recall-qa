import argparse
from wikidata.sparql_utils import *
from wikidata.prop_utils import (
    NO_AGGREGATION_PROPS, 
    INTERNAL_WIKI_PROPS, 
    VALID_PROP_DATATYPES,
    POINT_IN_TIME_QUALIFIER
)
from wikidata.data_utils import (
    normalize_numerical_types, 
    normalize_value_units
)
from io_utils import write_json_to_file




def single_value_per_entity(entity2values2qualifiers):
    num_values_for_each = sorted(
        [len(values2qualifiers["values"].keys()) for values2qualifiers in entity2values2qualifiers.values()], 
        reverse=True
    )
    return all(val == 1 for val in num_values_for_each)



def find_shared_point_in_time(qualifiers2qualvals2entities, all_entity_ids):
    shared_point_in_time = []
    point_in_time_values = qualifiers2qualvals2entities.get(POINT_IN_TIME_QUALIFIER,{}).get("entities_values",{})
    for point_in_time, entity_vals in point_in_time_values.items():
        # check if all entities have this point in time qualifier value
        entity_ids_for_this_time = [entity_id for (entity_id, entity_label, value, unit_id, unit_label) in entity_vals]
        if all(entity_id in entity_ids_for_this_time for entity_id in all_entity_ids):
            shared_point_in_time.append(point_in_time)

    return shared_point_in_time



def group_by_qualifiers(entity2values2qualifiers):
    qualifiers2qualvals2entities = {}
    for entity_id, values2qualifiers in entity2values2qualifiers.items():
        entity_label = values2qualifiers['entity_label']
        stmt, value_dict = next(iter(values2qualifiers["values"].items()))
        value, unit_id, unit_label = value_dict["value"], value_dict['unit_id'], value_dict["unit_label"]
        qualifiers = value_dict["qualifiers"]
        for qual_id, qual_dict in qualifiers.items():
            if qual_id not in qualifiers2qualvals2entities:
                qualifiers2qualvals2entities[qual_id] = {
                    "qual_label": qual_dict['label'],
                    "entities_values": {}
                }
            for val in qual_dict['values']:
                if val not in qualifiers2qualvals2entities[qual_id]["entities_values"]:
                    qualifiers2qualvals2entities[qual_id]["entities_values"][val] = []
                qualifiers2qualvals2entities[qual_id]["entities_values"][val].append((entity_id, entity_label, value, unit_id, unit_label))

    return qualifiers2qualvals2entities



def add_candidate_property(subclass, prop_id, prop, shared_point_in_time, entity2values2qualifiers): 
    if 'candidate_properties' not in subclass:
        subclass['candidate_properties'] = {}
    
    point_in_time_val = shared_point_in_time[0] if len(shared_point_in_time) > 0 else None
    subclass['candidate_properties'][prop_id] = {
        "property_info": prop,
        "point_in_time": point_in_time_val,
        "entities_values": []
    }
    for entity_id, values2qualifiers in entity2values2qualifiers.items():
        entity_label = values2qualifiers['entity_label']
        stmt, value_dict = next(iter(values2qualifiers["values"].items()))
        value, unit_id, unit_label = value_dict["value"], value_dict['unit_id'], value_dict["unit_label"]

        subclass['candidate_properties'][prop_id]["entities_values"].append({
            "entity_id": entity_id,
            "entity_label": entity_label,
            "value": value,
            "unit_id": unit_id,
            "unit_label": unit_label
        })



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
    print(f"Classes with instance count <= {args.max_instance_count}: {len(classes_instance_limit)}")

    classes_complete = [subclass for subclass in classes_instance_limit if subclass['instance_count'] == int(subclass['quantity'])]
    print(f"Classes where instance count matches quantity property: {len(classes_complete)}")

    print("Fetching instances...")
    for subclass in tqdm.tqdm(classes_complete):
        instances = get_instances_of_class(subclass["id"], limit=args.max_instance_count)
        subclass['instances'] = instances
        subclass['num_en_wikipedia_links'] = sum([inst.get('wikipedia', None) is not None for inst in instances])

    classes_complete_en_wiki = [subclass for subclass in classes_complete if subclass['num_en_wikipedia_links']==subclass['instance_count']]
    print(f"Classes with all instances linked to English Wikipedia: {len(classes_complete_en_wiki)}")

    print("Fetching shared properties & values...")
    for subclass in tqdm.tqdm(classes_complete_en_wiki):
        properties = get_properties_of_subclass(subclass["id"])
        filtered_properties = [prop for prop in properties if (
            prop['property_id'] not in INTERNAL_WIKI_PROPS
            and prop['property_id'] not in NO_AGGREGATION_PROPS
            and prop['datatype'] in VALID_PROP_DATATYPES
        )]

        subclass['shared_properties'] = {
            "Quantity": {},
            "GlobeCoordinate": {},
            "Time": {},
            "WikibaseItem": {}
        }
        for prop in filtered_properties:
            # ensure each property has a single value for each instance
            if prop['count'] == subclass['instance_count'] and is_property_fully_populated(subclass["id"], prop["property_id"]):
                if prop['datatype'] == 'Quantity':
                    entity2values2qualifiers = get_numerical_values(subclass["id"], prop["property_id"])
                    subclass['shared_properties']['Quantity'][prop["property_id"]] = (prop, entity2values2qualifiers)

    print("Selecting candidate properties...")
    num_subclass_prop_pairs = 0
    num_pairs_single_value = 0
    num_pairs_single_value_shared_time = 0
    for subclass in classes_complete_en_wiki:
        for prop_id,(prop,entity2values2qualifiers) in subclass['shared_properties']['Quantity'].items():
            num_subclass_prop_pairs += 1

            if single_value_per_entity(entity2values2qualifiers): # each entity has only one value
                num_pairs_single_value += 1
                
                all_entity_ids = set(entity2values2qualifiers.keys())
                qualifiers2qualvals2entities = group_by_qualifiers(entity2values2qualifiers)
                shared_point_in_time = find_shared_point_in_time(qualifiers2qualvals2entities, all_entity_ids)

                # either this property has no point in time qualifier for any entity, 
                # or all entity values share the same point in time qualifier
                if POINT_IN_TIME_QUALIFIER not in qualifiers2qualvals2entities or len(shared_point_in_time) > 0:
                    # TODO: if multiple shared point in time qualifiers found, 
                    # all can be used for creating different queries
                    add_candidate_property(subclass, prop_id, prop, shared_point_in_time, entity2values2qualifiers)
                    num_pairs_single_value_shared_time += 1
            else:
                # TODO: look into multi-value cases; 
                # these probably have only one "preferred" value per entity, so might be valid as candidates
                pass

    print(f"{len(classes_complete_en_wiki)} classes")
    print(f"{num_subclass_prop_pairs} total class-property pairs")
    print(f"{num_pairs_single_value}/{num_subclass_prop_pairs} had single values per entity")
    print(f"{num_pairs_single_value_shared_time}/{num_pairs_single_value} had a shared point in time")


    print("Normalizing units & values for numerical properties...")
    for subclass in classes_complete_en_wiki:
        defective_props = []

        for prop_id, prop_info in subclass['candidate_properties'].items():
            entity_values = prop_info["entities_values"]
            num_entities = len(entity_values)
            num_with_units = sum([entity_info['unit_id'] is not None for entity_info in entity_values])
            all_units = set([(entity_info['unit_id'],entity_info['unit_label']) for entity_info in entity_values if entity_info['unit_id'] is not None])
            
            if num_entities != num_with_units:
                defective_props.append(prop_id)
            else:
                normalize_numerical_types(entity_values)
                if len(all_units) != 1:
                    unit, _ = normalize_value_units(prop_id, entity_values)
                else:
                    unit = next(iter(all_units))
                prop_info["unit"] = unit

            all_values = set([entity_info['value'] for entity_info in entity_values])
            if len(all_values) < 2:
                defective_props.append(prop_id)
        
        # remove defective props from candidate properties
        for defective_prop in defective_props:
            if defective_prop in subclass['candidate_properties']:
                del subclass['candidate_properties'][defective_prop]

    query_generation_candidates = []
    for subclass in classes_complete_en_wiki:
        if len(subclass['candidate_properties'].items()) > 0:
            query_generation_candidates.append(subclass)

    write_json_to_file(args.output_file, query_generation_candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True, help="Output file location")
    parser.add_argument("--max_instance_count", type=int, default=100, help="Maximum instance count")
    args = parser.parse_args()
    
    fetch_candidates(args)
    
    
    