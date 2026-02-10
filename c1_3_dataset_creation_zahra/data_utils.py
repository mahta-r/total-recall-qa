import random
import re
from collections import defaultdict, Counter
from typing import Dict, Any, List

from unit_utils import UNIT_FACTORS, UNIT_CANONICAL


NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")



def get_order_key(values):
    
    def sort_key(v):
        has_number = any(ch.isdigit() for ch in v)
        if has_number:
            m = NUMBER_RE.search(v)
            number = float(m.group())
        else:
            number = 0.0
        return number

    if all(any(ch.isdigit() for ch in v) for v in values):
        order_key = {v: idx for idx, v in enumerate(values)}
    else:
        order_key = {v: sort_key(v) for v in values}
    
    return order_key




def parse_quantity_string(s: str):
    s = s.strip()

    m = NUMBER_RE.search(s)
    if not m:
        raise ValueError(f"No numeric value found in: {s}")

    value = float(m.group())
    unit = s[m.end():]

    unit = unit.strip()
    unit = unit.lstrip("-") 
    unit = unit.strip()

    if unit == "":
        unit_pair = None
    else:
        unit = unit.lower()
        unit = re.sub(r"\s+", " ", unit)
        unit_long = UNIT_CANONICAL[unit]
        unit_pair = (unit, unit_long)
    

    return value, unit_pair



def convert_unit(value: float, src_unit: str, dst_unit: str) -> float:
    
    if ( 
        src_unit is None or 
        dst_unit is None or
        src_unit[1] == dst_unit[1]
    ):
        return value

    if (src_unit[1], dst_unit[1]) not in UNIT_FACTORS:
        raise ValueError(f"Unsupported unit conversion: {src_unit} → {dst_unit} for value {value}")

    return UNIT_FACTORS[(src_unit[1], dst_unit[1])] * value


def normalize_value_units(entity_values: List[Dict], in_place: bool = True) -> List[Dict]:
    
    all_units = [entity_info["value_node"]["unit"] for entity_info in entity_values]
    common_unit = random.choice(all_units)

    # unit_counts = Counter(all_units)
    # most_common_units = unit_counts.most_common()
    # common_unit = most_common_units[0][0]

    normalized_values = []
    for entity_info in entity_values:
        converted_value = convert_unit(
            value=entity_info["value_node"]["operation_value"], 
            src_unit=entity_info["value_node"]["unit"], 
            dst_unit=common_unit
        )
        
        if common_unit == entity_info["value_node"]["unit"]:
            assert converted_value == entity_info["value_node"]["operation_value"]
        
        if in_place:
            entity_info["value_node"]["operation_value"] = converted_value
            entity_info["value_node"]["unit"] = common_unit
        else:
            normalized_values.append({
                "entity_id": entity_info["entity_id"],
                "entity_label": entity_info["entity_label"],
                "value_node": {
                    "value": entity_info["value_node"]["value"],
                    "operation_value": converted_value,
                    "unit": common_unit
                }
            })

    return (common_unit, entity_values) if in_place else (common_unit, normalized_values)



def normalize_numerical_type(value: Any) -> float:
    
    if isinstance(value, (int,float)):
        numerical_value, unit = float(value), None
    elif isinstance(value, str):
        numerical_value, unit = parse_quantity_string(value)
    else:
        raise ValueError(f"Unexpected value type for Quantity: {type(value)}")
    
    return numerical_value, unit




def format_values_by_datatype(
    feature: str, 
    feature_datatype: str, 
    feature_values: Any, 
    product_list: List[Dict[str, Any]], 
):
    property_record = {
        'label': feature,
        'datatype': feature_datatype,
        'possible_values': feature_values,
        'entity_count': sum(feature in product for product in product_list),
    }

    entity_values = []

    if feature_datatype == "Quantity":        

        for product in product_list:
            if feature in product:
                value = product[feature]
                numerical_value, unit = normalize_numerical_type(value)
                
                if isinstance(feature_values, list):
                    assert value in feature_values
                else:
                    assert numerical_value >= feature_values['minimum']  
                    assert numerical_value <= feature_values['maximum']

                value_node = {'value': value, 'operation_value': numerical_value, 'unit': unit}                
                entity_values.append({
                    'entity_id': product['index'],
                    'entity_label': product['ID'],
                    'value_node': value_node,
                })
        
        common_unit, entity_values = normalize_value_units(entity_values)
        property_record['unit'] = common_unit
        
        # if len(all_units) > 1:
        #     # print(f"{feature}: {', '.join(f"[{u[0]} | {u[1]}]" for u in all_units)}")
        #     print(f"{feature}: {', '.join(all_units)}")
        
        property_record['entities_values'] = entity_values
        all_values = set([entity_info['value_node']['operation_value'] for entity_info in entity_values])
        if len(all_values) >= 2:
            return property_record
        

    elif feature_datatype == "Date":

        for product in product_list:
            if feature in product:
                value = product[feature]
                date_value = int(float(value))
                
                if isinstance(feature_values, list):
                    assert value in feature_values
                else:    
                    assert date_value >= feature_values['minimum'] and date_value <= feature_values['maximum']

                value_node = {'value': date_value, 'operation_value': date_value}
                entity_values.append({
                    'entity_id': product['index'],
                    'entity_label': product['ID'],
                    'value_node': value_node,
                })
        
        property_record['entities_values'] = entity_values
        all_values = set([entity_info['value_node']['operation_value'] for entity_info in entity_values])
        if len(all_values) >= 2:
            return property_record
        
    elif feature_datatype in ["String", "OrderedString"]:

        for product in product_list:
            if feature in product:
                value = product[feature]
                assert isinstance(value, str)
                assert value in feature_values
                
                value_node = {'value': value, 'operation_value': value}
                entity_values.append({
                    'entity_id': product['index'],
                    'entity_label': product['ID'],
                    'value_node': value_node,
                })
        
        property_record['entities_values'] = entity_values
        all_values = set([entity_info['value_node']['operation_value'] for entity_info in entity_values])
        if len(all_values) >= 2:
            return property_record

    else:
        raise ValueError(f"Unsupported feature datatype: {feature_datatype}")
    
    return None