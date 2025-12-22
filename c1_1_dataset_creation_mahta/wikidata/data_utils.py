from collections import Counter
from typing import List, Dict


# Each entry defines: (src_unit, dst_unit): factor
# such that value_in_dst_unit = factor * value_in_src_unit
UNIT_FACTORS = {
    # --- area ---
    ("Q232291", "Q712226"): 2.58999,           # square mile → square kilometre
    ("Q712226", "Q232291"): 0.38610,           # square kilometre → square mile
    ("Q35852", "Q712226"): 0.01,               # hectare → square kilometre
    ("Q712226", "Q35852"): 100.0,              # square kilometre → hectare

    # --- elevation (length) ---
    ("Q3710", "Q11573"): 0.3048,               # foot → metre
    ("Q11573", "Q3710"): 3.28084,               # metre → foot

    # --- mass ---
    ("Q2655272", "Q11570"): 1e18,              # exagram → kilogram
    ("Q11570", "Q2655272"): 1e-18,             # kilogram → exagram

    # --- mean age ---
    ("Q199", "Q577"): 1,                       # 1 → year
    ("Q577", "Q199"): 1,                       # year → 1
}


def convert_unit(value: float, src_unit: str, dst_unit: str) -> float:
    
    if src_unit == dst_unit or src_unit is None or dst_unit is None:
        return value

    if (src_unit, dst_unit) not in UNIT_FACTORS:
        raise ValueError(f"Unsupported unit conversion: {src_unit} → {dst_unit}")

    return UNIT_FACTORS[(src_unit, dst_unit)] * value


def normalize_value_units(property_id: str, entity_values: List[Dict], in_place: bool = True) -> List[Dict]:
    
    unit_counts = Counter((value["unit_id"],value["unit_label"]) for value in entity_values)
    majority_unit_id, majority_unit_label = unit_counts.most_common(1)[0][0]

    normalized_values = []
    for entity_value in entity_values:
        converted_value = convert_unit(value=entity_value["value"], src_unit=entity_value["unit_id"], dst_unit=majority_unit_id)
        
        if majority_unit_id == entity_value["unit_id"]:
            assert majority_unit_label == entity_value["unit_label"]
            assert converted_value == entity_value["value"]
        
        if in_place:
            entity_value["value"] = converted_value
            entity_value["unit_id"] = majority_unit_id
            entity_value["unit_label"] = majority_unit_label
        else:
            normalized_values.append({
                "entity_id": entity_value["entity_id"],
                "entity_label": entity_value["entity_label"],
                "value": converted_value,
                "unit_id": majority_unit_id,
                "unit_label": majority_unit_label,
            })

    common_unit = (majority_unit_id, majority_unit_label)

    return (common_unit, entity_values) if in_place else (common_unit, normalized_values)


def normalize_numerical_types(entity_values: List[Dict], in_place: bool = True) -> List[Dict]:
    for entity_value in entity_values:
        value_str = entity_value['value']
        if isinstance(value_str, (int, float)):
            continue
        entity_value['value'] = float(value_str)

