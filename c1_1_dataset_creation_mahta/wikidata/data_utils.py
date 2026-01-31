from collections import Counter
import copy
import json
from typing import List, Dict

from .sparql_utils import find_shared_superclass, is_wikidata_id



def format_value(datatype, entity_value_node):
    if datatype == "Quantity":
        value_str = f"{entity_value_node['value']}"
        unit_id = entity_value_node.get('unit_id', None)
        if unit_id is not None and unit_id != "Q199": # excluding unit "1" (Q199)
            unit_label = entity_value_node['unit_label']
            value_str += f" (unit: {unit_label})"

    elif datatype == "Time":
        value_str = format_time_for_prompt(property_time=entity_value_node)
        calendar_id = entity_value_node.get('calendar_id', None)
        if calendar_id is not None:
            calendar_label = entity_value_node['calendar_label']
            value_str += f" (calendar: {calendar_label})"
    
    elif datatype == "GlobeCoordinate":
        latitude, longitude = entity_value_node['value']
        value_str = f"({latitude}, {longitude})"

    elif datatype == "WikibaseItem":
        item_labels = [ev['value_item_label'] for ev in entity_value_node]
        value_str = ", ".join(item_labels)

    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

    return value_str



# Each entry defines: (src_unit, dst_unit): factor
# such that value_in_dst_unit = factor * value_in_src_unit
UNIT_FACTORS = {
    # --- area ---
    ("Q232291", "Q712226"): 2.58999,           # square mile → square kilometre
    ("Q712226", "Q232291"): 0.38610,           # square kilometre → square mile
    ("Q35852", "Q712226"): 0.01,               # hectare → square kilometre
    ("Q712226", "Q35852"): 100.0,              # square kilometre → hectare
    ("Q25343", "Q712226"): 1e-6,               # square metre → square kilometre
    ("Q712226", "Q25343"): 1e6,                # square kilometre → square metre

    # --- elevation (length) ---
    ("Q3710", "Q11573"): 0.3048,               # foot → metre
    ("Q11573", "Q3710"): 3.28084,              # metre → foot

    # --- mass ---
    ("Q2655272", "Q11570"): 1e18,              # exagram → kilogram
    ("Q11570", "Q2655272"): 1e-18,             # kilogram → exagram

    # --- mean age ---
    ("Q199", "Q577"): 1,                       # 1 → year
    ("Q577", "Q199"): 1,                       # year → 1
}

CURRENCY_UNITS = {
    "Q4917",    # US dollar
    "Q4916",    # Euro
}

def convert_unit(value: float, src_unit: str, dst_unit: str) -> float:
    
    if (src_unit == dst_unit or 
        src_unit is None or 
        dst_unit is None or
        src_unit == 'Q199' or
        dst_unit == 'Q199'):
        return value

    if (src_unit, dst_unit) not in UNIT_FACTORS:
        raise ValueError(f"Unsupported unit conversion: {src_unit} → {dst_unit} for value {value}")


    return UNIT_FACTORS[(src_unit, dst_unit)] * value


def normalize_value_units(entity_values: List[Dict], in_place: bool = True) -> List[Dict]:
    
    unit_counts = Counter(
        (entity_info["value_node"]["unit_id"],entity_info["value_node"]["unit_label"]) 
        for entity_info in entity_values
    )
    most_common_units = unit_counts.most_common()
    majority_unit_id, majority_unit_label = most_common_units[0][0]
    
    if majority_unit_id == 'Q199' and len(most_common_units) > 1:
        # prefer units other than "1" if available
        majority_unit_id, majority_unit_label = most_common_units[1][0]
    
    if majority_unit_id in CURRENCY_UNITS and len(unit_counts) > 1:
        # currency conversion is tricky as conversion rates vary over time
        return None, None

    normalized_values = []
    for entity_info in entity_values:
        converted_value = convert_unit(
            value=entity_info["value_node"]["value"], 
            src_unit=entity_info["value_node"]["unit_id"], 
            dst_unit=majority_unit_id
        )
        
        if majority_unit_id == entity_info["value_node"]["unit_id"]:
            assert majority_unit_label == entity_info["value_node"]["unit_label"]
            assert converted_value == entity_info["value_node"]["value"]
        
        if in_place:
            entity_info["value_node"]["value"] = converted_value
            entity_info["value_node"]["unit_id"] = majority_unit_id
            entity_info["value_node"]["unit_label"] = majority_unit_label
        else:
            normalized_values.append({
                "entity_id": entity_info["entity_id"],
                "entity_label": entity_info["entity_label"],
                "value_node": {
                    "value": converted_value,
                    "unit_id": majority_unit_id,
                    "unit_label": majority_unit_label
                }
            })

    common_unit = (majority_unit_id, majority_unit_label)

    return (common_unit, entity_values) if in_place else (common_unit, normalized_values)


def normalize_numerical_types(entity_values: List[Dict]) -> List[Dict]:
    for entity_value in entity_values:
        value_str = entity_value['value_node']['value']
        if isinstance(value_str, (int, float)):
            continue
        entity_value['value_node']['value'] = float(value_str)

    return entity_values


def normalize_coordinate_types(entity_values: List[Dict]) -> List[Dict]:
    for entity_value in entity_values:
        latitude_str, longitude_str = entity_value['value_node']['value']
        entity_value['value_node']['value'] = (float(latitude_str), float(longitude_str))
    return entity_values



from datetime import datetime
from collections import defaultdict
from calendar import monthrange

INF_PAST = datetime.min
INF_FUTURE = datetime.max

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def format_time_for_prompt(
    *,
    shared_time=None,        # (start, end, precision)
    property_time=None       # {"value", "precision", "calendar_label"}
):
    """
    Returns a human-readable string suitable for LLM prompts and aggregation.
    Exactly one of shared_time or property_time must be provided.
    """

    assert (shared_time is None) or (property_time is None)

    if shared_time:
        start, end, precision = shared_time

        if precision == "day":
            assert (start.date() == end.date()) or end.year == 9999
        elif precision == "month":
            assert (start.year == end.year and start.month == end.month) or end.year == 9999
        elif precision == "year":
            assert (start.year == end.year) or end.year == 9999

        dt = start

    else:
        dt = property_time["value"]
        precision = property_time["precision"]
    

    if precision == "year":
        out = f"{dt.year}"
    elif precision == "month":
        out = f"{dt.year}/{dt.month} ({MONTH_NAMES[dt.month]} {dt.year})"
    elif precision == "day":
        out = f"{dt.year}/{dt.month}/{dt.day} ({dt.day} {MONTH_NAMES[dt.month]} {dt.year})"
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    return out


def normalize_time_precisions(entity_values: List[Dict]) -> List[Dict]:
    for entity_info in entity_values:
        time_str = entity_info['value_node']['value']
        wikidata_precision = entity_info['value_node'].get('precision', None)
        dt, prec = parse_time_point(time_str, wikidata_precision)
        if dt is None or prec is None:
            return None, None
        entity_info['value_node']['value'] = dt
        entity_info['value_node']['precision'] = prec
    
    all_precisions = set(
        [entity_info['value_node']['precision'] for entity_info in entity_values]
    )
    
    # normalize to the coarsest precision
    if "year" in all_precisions:
        target_precision = "year"
    elif "month" in all_precisions:
        target_precision = "month"
    else:
        target_precision = "day"
        
    for entity_info in entity_values:
        dt = entity_info['value_node']['value']
        if target_precision == "year":
            dt = dt.replace(month=1, day=1)
        elif target_precision == "month":
            dt = dt.replace(day=1)
        entity_info['value_node']['value'] = dt
        entity_info['value_node']['precision'] = target_precision

    return target_precision, entity_values



def parse_time_point(time_str, wikidata_precision=None):
    """
    Parse time string into (datetime, precision)
    Precision inferred from day/month values if wikidata precision missing.
    """
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", ""))
        
        if wikidata_precision :
            precision = {
                9: "year",
                10: "month",
                11: "day"
            }[int(wikidata_precision)]
        else:
            if dt.month == 1 and dt.day == 1:
                precision = "year"
            elif dt.day == 1:
                precision = "month"
            else:
                precision = "day"
        
        return dt, precision
    
    except (ValueError, KeyError):
        return None, None


def expand_time_point(dt, precision, direction):
    """
    Expand a time point to a date boundary.
    All returned values are dates (YYYY-MM-DD), no time-of-day.
    """

    if precision == "year":
        if direction == "start":
            return dt.replace(month=1, day=1)
        else:
            return dt.replace(month=12, day=31)

    if precision == "month":
        if direction == "start":
            return dt.replace(day=1)
        else:
            last_day = monthrange(dt.year, dt.month)[1]
            return dt.replace(day=last_day)

    if precision == "day":
        return dt # day precision → no expansion needed

    return dt


def intersect_time_intervals(a, b):
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    if start <= end:
        return (start, end)
    return None


def infer_interval_precision(start, end):
    # single day
    if start == end:
        return "day"

    # year precision: whole years (possibly multiple)
    if (
        start.month == 1 and start.day == 1 and
        end.month == 12 and end.day == 31
    ):
        return "year"

    # month precision: whole months (possibly across years)
    if (
        start.day == 1 and # is first day of month
        end.day == monthrange(end.year, end.month)[1] # is last day of month
    ):
        return "month"

    # otherwise needs day precision
    return "day"



def find_shared_time(entity2values2qualifiers):
    """
    Returns:
      - (shared_start, shared_end, precision) |  None
      - dict: entity_id -> list of supporting statement_ids | None
    """

    entity_intervals = {}
    any_time_present = False

    # ---------- Step 1: build statement-level intervals ----------
    for entity_id, values2qualifiers in entity2values2qualifiers.items():
        stmt_intervals = []

        for stmt_id, stmt_data in values2qualifiers["values"].items():
            quals = stmt_data.get("qualifiers", {})

            # --- P585: point-in-time statements ---
            points = quals.get("P585", {}).get("values", [])
            for t in points:
                any_time_present = True
                dt, prec = parse_time_point(t)
                if dt is None:
                    continue
                start = expand_time_point(dt, prec, "start")
                end = expand_time_point(dt, prec, "end")
                stmt_intervals.append(((start, end), stmt_id))

            # --- P580 / P582: interval statements ---
            starts = quals.get("P580", {}).get("values", [])
            ends = quals.get("P582", {}).get("values", [])
            if starts or ends:
                any_time_present = True
                
                # start_points = [parse_time_point(t) for t in starts] if starts else [(None, None)]
                if starts:
                    start_points = []
                    for t in starts: 
                        dt, prec = parse_time_point(t)
                        if dt is not None:
                            start_points.append((dt, prec))
                else:
                    start_points = [(None, None)]

                # end_points = [parse_time_point(t) for t in ends] if ends else [(None, None)]
                if ends:
                    end_points = []
                    for t in ends:
                        dt, prec = parse_time_point(t)
                        if dt is not None:
                            end_points.append((dt, prec))
                else:
                    end_points = [(None, None)]

                for start_dt, start_prec in start_points:
                    for end_dt, end_prec in end_points:
                        start = (
                            expand_time_point(start_dt, start_prec, "start")
                            if start_dt else INF_PAST
                        )
                        end = (
                            expand_time_point(end_dt, end_prec, "end")
                            if end_dt else INF_FUTURE
                        )

                        if start <= end:
                            stmt_intervals.append(((start, end), stmt_id))

        entity_intervals[entity_id] = stmt_intervals

    # ---------- Step 2: rule out trivial fail/success (either all or none must have time) ----------
    
    if any_time_present:
        for entity_id, intervals in entity_intervals.items():
            if not intervals:
                return None, None  # trivial fail
    else:
        # no time anywhere → trivial success
        return None, {
            eid: list(data["values"].keys())
            for eid, data in entity2values2qualifiers.items()
        }

    # ---------- Step 3: intersect across entities ----------
    
    shared_times = {}
    for entity_id, intervals in entity_intervals.items():
        entity_times = defaultdict(lambda: defaultdict(lambda: set()))

        # first entity initializes the candidate shared times
        if not shared_times:
            for interval, stmt_id in intervals:
                entity_times[interval][entity_id].add(stmt_id)
        else:
            for shared_interval in shared_times:
                prev_stmts = shared_times[shared_interval]
                for interval, stmt_id in intervals:
                    inter = intersect_time_intervals(shared_interval, interval)
                    if inter:
                        if inter not in entity_times:
                            entity_times[inter] = copy.deepcopy(prev_stmts)
                        entity_times[inter][entity_id].add(stmt_id)
        
        if not entity_times:
            return None, None 

        shared_times = entity_times

    # ---------- Step 4: select latest shared time ----------

    # return latest (most recent) valid shared time & values as it's most likely to appear in wikipedia text
    latest_interval = max(
        list(shared_times.keys()),
        key=lambda x: x[1]
    )
    interval_statements = shared_times[latest_interval]
    all_shared_statements = statements_valid_in_interval(entity_intervals, latest_interval)
    for entity_id, entity_stmts in interval_statements.items():
        for stmt in entity_stmts:
            assert stmt in all_shared_statements[entity_id]
    precision = infer_interval_precision(*latest_interval)
    return (latest_interval[0], latest_interval[1], precision), all_shared_statements

    # for interval in sorted(list(shared_times.keys()), key=lambda x: x[1], reverse=True):
    #     statements = shared_times[interval]
    #     if any(len(entity_stmts) > 1 for entity_stmts in statements.values()):
    #         continue  # skip intervals with multiple supporting statements for any entity
    #     precision = infer_interval_precision(*interval)
    #     return (interval[0], interval[1], precision), statements

    # return None, None # no valid shared time found



def statements_valid_in_interval(entity_intervals, shared_interval):
    """
    Returns:
      entity_id -> list of statement_ids valid for the entire shared_interval
    """
    shared_start, shared_end = shared_interval
    result = defaultdict(lambda: set())

    for entity_id, intervals in entity_intervals.items():
        valid = set([
            stmt_id
            for (start, end), stmt_id in intervals
            if start <= shared_start and end >= shared_end
        ])
        result[entity_id] = valid

    return result



def extract_values_for_aggregation(prop, statements, entity2values2qualifiers):
    
    if prop['datatype'] == 'WikibaseItem':
        entity_values = []
        for entity_id, values2qualifiers in entity2values2qualifiers.items():
            values_for_entity = []
            for stmt_id in statements[entity_id]:
                values_for_entity.append({k:v for k,v in values2qualifiers['values'][stmt_id].items() if k != 'qualifiers'})
            entity_values.append({
                "entity_id": entity_id,
                "entity_label": values2qualifiers['entity_label'],
                "value_node": values_for_entity
            })

        item_list_per_entity = [
            [(item['value_item_id'], item['value_item_label']) for item in entity_value['value_node']]
            for entity_value in entity_values
        ]
        
        all_unique_items = set.union(*[set(lst) for lst in item_list_per_entity])
        shared_items = set.intersection(*[set(lst) for lst in item_list_per_entity])
        item_counts = Counter([item for item_list in item_list_per_entity for item in item_list])

        # only use properties that have english labels for all items
        if all(not is_wikidata_id(item_label) for item_id,item_label in all_unique_items):
            
            max_item_per_entity = 20
            if len(entity_values) > 2 and all(len(item_list) < max_item_per_entity for item_list in item_list_per_entity):
                
                # can only aggregate over 1-to-1 or 1-to-many properties that are not all the same items or all different items
                # require entities that appear at least twice but not in all
                
                num_entities = len(entity_values)
                min_count = 1 if num_entities == 2 else 2
                valid_items = [item for item, count in item_counts.items() if min_count <= count < num_entities]
                
                # if len(shared_items) > 0 and len(shared_items) < len(all_unique_items):
                if len(shared_items) < len(all_unique_items) and len(valid_items) > 0:
                    # find shared superclass of all items (for specifying in query)
                    shared_classes = find_shared_superclass([item_id for item_id,item_label in list(all_unique_items)])
                    if shared_classes:
                        prop['item_class'] = shared_classes[0] 
                        return prop, entity_values
    else:
        # we skip all candidates with multiple values (statements) at a given shared time for any entity 
        # as it's unclear which statement to use per entity (skips about 5% of candidates)
        if not any(len(stmts) > 1 for stmts in statements.values()):
            entity_values = []
            for entity_id, values2qualifiers in entity2values2qualifiers.items():
                single_stmt_id = next(iter(statements[entity_id]))
                value_for_entity = {k:v for k,v in values2qualifiers['values'][single_stmt_id].items() if k != 'qualifiers'}
                entity_values.append({
                    "entity_id": entity_id,
                    "entity_label": values2qualifiers['entity_label'],
                    "value_node": value_for_entity
                })
            
            if prop['datatype'] == 'Quantity':
                num_entities = len(entity_values)
                num_with_units = sum(
                    [entity_info['value_node']['unit_id'] is not None for entity_info in entity_values]
                )
                
                if num_entities == num_with_units:
                    entity_values = normalize_numerical_types(entity_values)
                    common_unit, entity_values = normalize_value_units(entity_values)
                    if common_unit:
                        prop["unit"] = common_unit

                        all_values = set([entity_info['value_node']['value'] for entity_info in entity_values])
                        if len(all_values) >= 2:
                            return prop, entity_values
                    
            if prop['datatype'] == 'Time':
                num_entities = len(entity_values)
                all_calendars = [
                    (entity_info['value_node']['calendar_id'], entity_info['value_node']['calendar_label'])
                    for entity_info in entity_values 
                    if entity_info['value_node']['calendar_id'] is not None
                ]
                num_with_calendars = len(all_calendars)
                
                if num_entities == num_with_calendars:
                    if len(set(all_calendars)) == 1:
                        common_calendar = next(iter(all_calendars))
                        prop["calendar"] = common_calendar

                        precision, entity_values = normalize_time_precisions(entity_values)
                        if entity_values and precision:
                            prop['precision'] = precision

                            # TODO: test time parsing/conversions
                            all_values = set([
                                (entity_info['value_node']['value'], entity_info['value_node']['precision'])
                                for entity_info in entity_values
                            ])
                            if len(all_values) >= 2:
                                return prop, entity_values
                            
            if prop['datatype'] == 'GlobeCoordinate':
                num_entities = len(entity_values)
                all_globes = [
                    (entity_info['value_node']['globe_id'], entity_info['value_node']['globe_label'])
                    for entity_info in entity_values 
                    if entity_info['value_node']['globe_id'] is not None
                ]
                num_with_globe = len(all_globes)
                
                if num_entities == num_with_globe:
                    if len(set(all_globes)) == 1:
                        common_globe = next(iter(all_globes))
                        prop["globe"] = common_globe

                        entity_values = normalize_coordinate_types(entity_values)
                    
                        all_values = set([entity_info['value_node']['value'] for entity_info in entity_values])
                        if len(all_values) >= 2:
                            return prop, entity_values
                        
    return None, None


def extract_multihop_candidates(prop, statements, entity2values2qualifiers, max_instance_count=50):
    if not entity2values2qualifiers or not statements:
        return None, None, None
    
    if prop['datatype'] == 'WikibaseItem':
        entity_values = []
        for entity_id, values2qualifiers in entity2values2qualifiers.items():
            values_for_entity = []
            for stmt_id in statements[entity_id]:
                values_for_entity.append({k:v for k,v in values2qualifiers['values'][stmt_id].items() if k != 'qualifiers'})
            entity_values.append({
                "entity_id": entity_id,
                "entity_label": values2qualifiers['entity_label'],
                "value_node": values_for_entity
            })

        item_list_per_entity = [
            set(
                [(item['value_item_id'], item['value_item_label']) for item in entity_value['value_node']]
            ) 
            for entity_value in entity_values
        ]
        all_items = set.union(*item_list_per_entity)
        shared_items = set.intersection(*item_list_per_entity)
        value_counts = Counter([frozenset(item_list) for item_list in item_list_per_entity])

        src_entities = [(entity_value['entity_id'], entity_value['entity_label']) for entity_value in entity_values]
        all_items_original = set([item for item in all_items if item not in src_entities])

        # only use properties that have english labels for all items
        if all(not is_wikidata_id(item_label) for item_id,item_label in all_items):
            
            # # use properties that weren't used for aggregation 
            # if not (len(shared_items) > 0 and len(shared_items) < len(all_items)):
            if len(shared_items) < len(all_items) and len(all_items)/len(src_entities) > 0.25:
                if len(all_items) <= max_instance_count:
                    # find shared superclass of all items (for specifying in query)
                    shared_classes = find_shared_superclass([item_id for item_id,item_label in list(all_items)])
                    if shared_classes:
                        prop['item_class'] = shared_classes[0] 
                        return prop, entity_values, all_items
                        # return prop, entity_values, all_items_original
    
    return None, None, None
