import math
import numpy as np
from statistics import median
from collections import defaultdict
from itertools import combinations    

from wikidata.data_utils import format_value




def is_valid_for_values(operation, input_entity_values):
    if operation in {"SUM_TOP_K", "SUM_BOTTOM_K", "AVG_TOP_K", "AVG_BOTTOM_K", "COUNT_LT_X", "COUNT_GT_X", "MEDIAN"}:
        return len(input_entity_values) >= 3
    
    elif operation in {"SUM", "AVG", "MAX"}:
        return len(input_entity_values) >= 2
    
    elif operation in {"MIN"}:
        # Ensure final answer is non-zero
        values = [entity_value['value_node']['value'] for entity_value in input_entity_values]
        return len(input_entity_values) >= 2 and min(values) != 0

    elif operation in {"DIFFERENCE(MAX−MIN)"}:
        # Ensure final ansewer is non-zero
        values = [entity_value['value_node']['value'] for entity_value in input_entity_values]
        return len(input_entity_values) >= 2 and max(values) != min(values)
    
    elif operation in {"RATIO(MAX/MIN)"}:
        # Ensure no division-by-zero from values
        values = [entity_value['value_node']['value'] for entity_value in input_entity_values]
        return len(input_entity_values) >= 2 and min(values) != 0 and max(values) != 0
    
    elif operation in {"PAIRWISE_MAX_DISTANCE", "PAIRWISE_MIN_DISTANCE"}:
        # Ensure no duplicate coordinates (causing zero distance)
        values = [entity_value['value_node']['value'] for entity_value in input_entity_values]
        return len(input_entity_values) >= 3 
        # and len(values) == len(set(values))
    
    elif operation in {"COUNT_DIRECTIONAL", "COUNT_WITHIN_RADIUS"}:
        return len(input_entity_values) >= 2
    
    elif operation in {"COUNT_BEFORE", "COUNT_AFTER"}:
        return len(input_entity_values) >= 3
    
    elif operation in {"EARLIEST", "LATEST"}:
        # Earliest/Latest only support year to yield simple numerical answer
        precisions = [entity_value['value_node']['precision'] for entity_value in input_entity_values]
        return len(input_entity_values) >= 2 and all(p == 'year' for p in precisions)
    
    elif operation in {"NTH_EARLIEST", "NTH_LATEST"}:
        # Earliest/Latest only support year to yield simple numerical answer
        precisions = [entity_value['value_node']['precision'] for entity_value in input_entity_values]
        return len(input_entity_values) >= 3 and all(p == 'year' for p in precisions)
    
    elif operation in {"TIME_BETWEEN_FIRST_LAST"}:
        return len(input_entity_values) >= 2

    elif operation in {"COUNT_VALUE"}:
        return len(input_entity_values) >= 2
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


def numerical_aggregation(
    operation, 
    input_entity_values,
    rand=None
):
    values = [entity_value['value_node']['value'] for entity_value in input_entity_values]
    arr = np.array(values, dtype=float)
    op = operation.upper()
    
    if op == "SUM":
        return op, "", float(np.sum(arr))
    
    elif op == "SUM_TOP_K":
        k = rand.randint(2, max(2, int(len(arr) * 0.6)))
        top_k_sum = float(np.sum(np.sort(arr)[-k:]))
        return op, f"K={k}", top_k_sum
    
    elif op == "SUM_BOTTOM_K":
        k = rand.randint(2, max(2, int(len(arr) * 0.6)))
        bottom_k_sum = float(np.sum(np.sort(arr)[:k]))
        return op, f"K={k}", bottom_k_sum
    
    elif op == "AVG":
        return op, "", float(np.mean(arr))
    
    elif op == "AVG_TOP_K":
        k = rand.randint(2, max(2, int(len(arr) * 0.6)))
        top_k_avg = float(np.mean(np.sort(arr)[-k:]))
        return op, f"K={k}", top_k_avg
    
    elif op == "AVG_BOTTOM_K":
        k = rand.randint(2, max(2, int(len(arr) * 0.6)))
        bottom_k_avg = float(np.mean(np.sort(arr)[:k]))
        return op, f"K={k}", bottom_k_avg
    
    elif op == "MAX":
        return op, "", float(np.max(arr))
    
    elif op == "MIN":
        return op, "", float(np.min(arr))
    
    elif op == "MEDIAN":
        return op, "", float(median(arr))
    
    elif op == "DIFFERENCE(MAX−MIN)":
        return op, "", float(np.max(arr) - np.min(arr))
    
    elif op == "RATIO(MAX/MIN)":
        return op, "", float(np.max(arr) / np.min(arr))
    
    elif op == "COUNT_LT_X":
        min_ratio = 0.3
        max_ratio = 0.9
        ratio = 0
        while ratio == 0 or ratio < min_ratio or ratio > max_ratio:
            x_entity_value = rand.choice(input_entity_values)
            x = float(x_entity_value['value_node']['value'])
            count = int(np.sum(arr < x))
            ratio = count / len(arr)
            min_ratio -= 0.01
            max_ratio += 0.01
        x_entity_string = format_value("Quantity", x_entity_value['value_node'])
        return op, f"X={x_entity_string} [{x_entity_value['entity_label']}]", count

    elif op == "COUNT_GT_X":
        min_ratio, max_ratio = 0.3, 0.9
        ratio = 0
        while ratio == 0 or ratio < min_ratio or ratio > max_ratio:
            x_entity_value = rand.choice(input_entity_values)
            x = float(x_entity_value['value_node']['value'])
            count = int(np.sum(arr > x))
            ratio = count / len(arr)
            min_ratio -= 0.01
            max_ratio += 0.01
        x_entity_string = format_value("Quantity", x_entity_value['value_node'])
        return op, f"X={x_entity_string} [{x_entity_value['entity_label']}]", count
    
    else:
        raise ValueError(f"Unknown numerical operation: {operation}")


def coordinates_aggregation(
    operation, 
    input_entity_values,
    rand=None
):    
    coordinates = [entity_value['value_node']['value'] for entity_value in input_entity_values]
    op = operation.upper()

    if op == "PAIRWISE_MAX_DISTANCE":
        max_dist = 0.0
        for (lat1, lon1), (lat2, lon2) in combinations(coordinates, 2):
            d = haversine(lat1, lon1, lat2, lon2)
            if d > max_dist:
                max_dist = d
        return op, "Unit=km", max_dist

    elif op == "PAIRWISE_MIN_DISTANCE":
        min_dist = float('inf')
        for (lat1, lon1), (lat2, lon2) in combinations(coordinates, 2):
            d = haversine(lat1, lon1, lat2, lon2)
            if d < min_dist:
                min_dist = d
        return op, "Unit=km", min_dist
    
    elif op == "COUNT_WITHIN_RADIUS":
        lats = [lat for lat, _ in coordinates]
        lons = [lon for _, lon in coordinates]
        all_dists = [
            haversine(lat1, lon1, lat2, lon2)
            for (lat1, lon1), (lat2, lon2) in combinations(coordinates, 2)
        ]
        max_dist = max(all_dists)
        min_ratio, max_ratio = 0.3, 0.9
        ratio = 0
        while ratio == 0 or ratio < min_ratio or ratio > max_ratio:
            lat_ref = round(rand.uniform(min(lats), max(lats)), 3)
            lon_ref = round(rand.uniform(min(lons), max(lons)), 3)
            dists = [haversine(lat_ref, lon_ref, lat, lon) for lat, lon in coordinates]
            radius = round(rand.uniform(0.05 * max_dist, 0.7 * max_dist), 2) # in km
            count = sum(d <= radius for d in dists)
            ratio = count / len(coordinates)
            min_ratio -= 0.01
            max_ratio += 0.01
        return op, f"Radius={radius:.2f}km, Reference=({lat_ref}, {lon_ref})", count

    elif op == "COUNT_DIRECTIONAL":
        lats = [lat for lat, _ in coordinates]
        lons = [lon for _, lon in coordinates]
        directions = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
        min_ratio, max_ratio = 0.3, 0.9
        ratio = 0
        while ratio == 0 or ratio < min_ratio or ratio > max_ratio:
            lat_ref = round(rand.uniform(min(lats), max(lats)), 3)
            lon_ref = round(rand.uniform(min(lons), max(lons)), 3)
            direction = rand.choice(directions)
            if direction == "N":
                count = sum(lat > lat_ref for lat, _ in coordinates)
            elif direction == "S":
                count = sum(lat < lat_ref for lat, _ in coordinates)
            elif direction == "E":
                count = sum(lon > lon_ref for _, lon in coordinates)
            elif direction == "W":
                count = sum(lon < lon_ref for _, lon in coordinates)
            elif direction == "NE":
                count = sum(lat > lat_ref and lon > lon_ref for lat, lon in coordinates)
            elif direction == "NW":
                count = sum(lat > lat_ref and lon < lon_ref for lat, lon in coordinates)
            elif direction == "SE":
                count = sum(lat < lat_ref and lon > lon_ref for lat, lon in coordinates)
            elif direction == "SW":
                count = sum(lat < lat_ref and lon < lon_ref for lat, lon in coordinates)
            ratio = count / len(coordinates)
            min_ratio -= 0.01
            max_ratio += 0.01
        return op, f"Direction={direction}, Reference=({lat_ref}, {lon_ref})", count

    else:
        raise ValueError(f"Unknown coordinate operation: {operation}")


def temporal_aggregation(
    operation, 
    input_entity_values, 
    rand=None
):
    datetimes = [entity_value['value_node']['value'] for entity_value in input_entity_values]
    arr = sorted(datetimes)
    op = operation.upper()

    if op == "EARLIEST":
        year = arr[0].year
        return op, "Precision=year", year

    elif op == "LATEST":
        year = arr[-1].year
        return op, "Precision=year", year

    elif op == "NTH_EARLIEST":
        n = rand.randint(2, max(2, int(len(arr)/2)))
        year = arr[n - 1].year
        return op, f"Precision=year, N={n}", year

    elif op == "NTH_LATEST":
        n = rand.randint(2, max(2, int(len(arr)/2)))
        year = arr[-n].year
        return op, f"Precision=year, N={n}", year

    elif op == "COUNT_BEFORE":
        min_ratio, max_ratio = 0.3, 0.9
        ratio = 0
        while ratio == 0 or ratio < min_ratio or ratio > max_ratio:
            ref_entity_value = rand.choice(input_entity_values)
            ref = ref_entity_value['value_node']['value']
            count = sum(dt < ref for dt in arr)
            ratio = count / len(arr)
            min_ratio -= 0.01
            max_ratio += 0.01
        reference_entity_string = format_value("Time", ref_entity_value['value_node'])
        return op, f"Reference={reference_entity_string} [{ref_entity_value['entity_label']}]", count
    
    elif op == "COUNT_AFTER":
        min_ratio, max_ratio = 0.3, 0.9
        ratio = 0
        while ratio == 0 or ratio < min_ratio or ratio > max_ratio:
            ref_entity_value = rand.choice(input_entity_values)
            ref = ref_entity_value['value_node']['value']
            count = sum(dt > ref for dt in arr)
            ratio = count / len(arr)
            min_ratio -= 0.01
            max_ratio += 0.01
        reference_entity_string = format_value("Time", ref_entity_value['value_node'])
        return op, f"Reference={reference_entity_string} [{ref_entity_value['entity_label']}]", count
    
    elif op == "TIME_BETWEEN_FIRST_LAST":
        precision = input_entity_values[0]['value_node']['precision']
        span = time_diff(arr[0], arr[-1], precision)
        return op, f"Precision={precision}s", span

    else:
        raise ValueError(f"Unknown temporal operation: {operation}")


def items_aggregation(
    operation,
    input_entity_values,
    rand=None,
):
    if operation == "COUNT_VALUE":
        
        items_per_entity = [
            [(item["value_item_id"], item["value_item_label"]) for item in entity_value["value_node"]]
            for entity_value in input_entity_values
        ]
        item2entities = defaultdict(set)
        for idx, item_lst in enumerate(items_per_entity):
            for item_id, item_label in item_lst:
                item2entities[(item_id,item_label)].add(idx)
        universe = list(item2entities.keys())

        operators = ["COUNT_HAS_X", "COUNT_NOT_HAS_X"]
        if len(universe) >= 3:
            operators += ["COUNT_HAS_ANY_X", "COUNT_NOT_HAS_ANY_X"]

        min_ratio, max_ratio = 0.3, 0.7
        ratio = 0
        while ratio == 0 or ratio < min_ratio or ratio > max_ratio:

            op = rand.choice(operators)
            
            if op in {"COUNT_HAS_X", "COUNT_NOT_HAS_X"}:
                k = 1
            else:
                k = rand.choice([2, 3])
                k = min(k, max(2, len(universe) // 2)) 
            reference = rand.sample(universe, k)

            if op == "COUNT_HAS_X":
                assert len(reference) == 1
                count = sum(reference[0] in item_lst for item_lst in items_per_entity)
                assert count == len(item2entities[reference[0]])
            if op == "COUNT_NOT_HAS_X":
                assert len(reference) == 1
                count = sum(reference[0] not in item_lst for item_lst in items_per_entity)
                assert count == len(items_per_entity) - len(item2entities[reference[0]])
            if op == "COUNT_HAS_ANY_X":
                assert len(reference) > 1
                count = sum(any(ref in item_lst for ref in reference) for item_lst in items_per_entity)
            if op == "COUNT_NOT_HAS_ANY_X":
                assert len(reference) > 1
                count = sum(not any(ref in item_lst for ref in reference) for item_lst in items_per_entity)

            ratio = count / len(input_entity_values)
            min_ratio -= 0.01
            max_ratio += 0.01
        
        reference_entity_str = ", ".join(ref[1] for ref in reference)
        return op, f"Reference=[{reference_entity_str}]", count

    else:
        raise ValueError(f"Unknown relational operation: {operation}")
        


def time_diff(a, b, precision):
    if precision == "year":
        return b.year - a.year
    elif precision == "month":
        return (b.year - a.year) * 12 + (b.month - a.month)
    elif precision == "day":
        return (b - a).days
    else:
        raise ValueError(f"Unknown precision: {precision}")


def haversine(lat1, lon1, lat2, lon2):
    EARTH_RADIUS_KM = 6371.0
    # convert degrees → radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c



##################################################################################
# Constraint filters for GlobeCoordinate, Quantity, Time, WikibaseItem datatypes #
##################################################################################


def coordinates_filter(
    coordinates,
    pass_rate_range=(0.5, 0.9),
    rand=None,
):  

    directions_1d = ["N", "S", "E", "W"]
    directions_2d = ["NE", "NW", "SE", "SW"]
    
    N = len(coordinates)
    assert N >= 3 # Need at least 3 coordinates to apply filter and have multiple passed points

    min_pass_count = 2 if N == 3 else 3
    low = max(pass_rate_range[0], min_pass_count / N)
    high = pass_rate_range[1]
    
    passed_indices = []
    while len(passed_indices) / N < low or len(passed_indices) / N > high:
        pass_rate = rand.uniform(low, high)
        direction = rand.choice(directions_1d + directions_2d)
    
        def quantile(values, q):
            idx = min(N - 1, int(q * N))
            return values[idx]
        
        lats = sorted(coord[0] for coord, idx in coordinates)
        lons = sorted(coord[1] for coord, idx in coordinates)

        # --- choose reference coordinate ---
        
        if direction in directions_1d:
            if direction == "N":
                ref_lat = quantile(lats, 1 - pass_rate)
                ref_lon = None
            elif direction == "S":
                ref_lat = quantile(lats, pass_rate)
                ref_lon = None
            elif direction == "E":
                ref_lon = quantile(lons, 1 - pass_rate)
                ref_lat = None
            elif direction == "W":
                ref_lon = quantile(lons, pass_rate)
                ref_lat = None
        elif direction in directions_2d:  
            q = math.sqrt(pass_rate)
            if "N" in direction:
                ref_lat = quantile(lats, 1 - q)
            else:
                ref_lat = quantile(lats, q)
            if "E" in direction:
                ref_lon = quantile(lons, 1 - q)
            else:
                ref_lon = quantile(lons, q)

        # --- choose actual datapoint as reference ---
        
        def dist_to_ref(coord):
            return (
                abs(coord[0] - ref_lat if ref_lat is not None else 0) + 
                abs(coord[1] - ref_lon if ref_lon is not None else 0)
            )
        
        (reference_lat, reference_lon), reference_idx = min(coordinates, key=lambda coord: dist_to_ref(coord[0]))
        
        # --- apply filter ---
        
        def passes_constraint(coord):
            lat, lon = coord
            if direction == "N":  return lat > reference_lat
            if direction == "S":  return lat < reference_lat
            if direction == "E":  return lon > reference_lon
            if direction == "W":  return lon < reference_lon
            if direction == "NE": return lat > reference_lat and lon > reference_lon
            if direction == "NW": return lat > reference_lat and lon < reference_lon
            if direction == "SE": return lat < reference_lat and lon > reference_lon
            if direction == "SW": return lat < reference_lat and lon < reference_lon

        passed_indices = [idx for coord, idx in coordinates if passes_constraint(coord)]

    return passed_indices, reference_idx, direction



def numerical_filter(
    values,
    pass_rate_range=(0.5, 0.9),
    rand=None,
):
    
    N = len(values)
    assert len(values) >= 3
    
    min_pass_count = 2 if N == 3 else 3
    low = max(pass_rate_range[0], min_pass_count / N)
    high = pass_rate_range[1]

    passed_indices = []
    while len(passed_indices) / N < low or len(passed_indices) / N > high:
        pass_rate = rand.uniform(low, high)
        direction = rand.choice(["GT", "GTE", "LT", "LTE"])

        def quantile(values, q):
            idx = min(N - 1, int(q * N))
            return values[idx]

        vals = sorted(values, key=lambda v: v[0])

        # --- choose reference value ---

        if direction in {"GT", "GTE"}:
            reference, reference_idx = quantile(vals, 1 - pass_rate)
        elif direction in {"LT", "LTE"}:
            reference, reference_idx = quantile(vals, pass_rate)

        def passes_constraint(value):
            if direction == "GT":  return value > reference
            if direction == "GTE": return value >= reference
            if direction == "LT":  return value < reference
            if direction == "LTE": return value <= reference
        
        passed_indices = [idx for val, idx in values if passes_constraint(val)]

    return passed_indices, reference_idx, direction


def temporal_filter(
    datetimes,
    pass_rate_range=(0.5, 0.9),
    rand=None,
):

    N = len(datetimes)
    assert N >= 3

    min_pass_count = 2 if N == 3 else 3
    low = max(pass_rate_range[0], min_pass_count / N)
    high = pass_rate_range[1]

    passed_indices = []
    while len(passed_indices) / N < low or len(passed_indices) / N > high:
        pass_rate = rand.uniform(low, high)
        direction = rand.choice(["BEFORE", "AFTER"])

        def quantile(values, q):
            idx = min(N - 1, int(q * N))
            return values[idx]

        dates = sorted(datetimes, key=lambda dt: dt[0])

        # --- choose reference time ---
        
        if direction == "AFTER": 
            reference, reference_idx = quantile(dates, 1 - pass_rate)
        elif direction == "BEFORE":
            reference, reference_idx = quantile(dates, pass_rate)

        def passes_constraint(value):
            if direction == "AFTER":  return value > reference
            if direction == "BEFORE": return value < reference

        passed_indices = [idx for val, idx in datetimes if passes_constraint(val)]

    return passed_indices, reference_idx, direction




def items_filter(
    items,
    pass_rate_range=(0.5, 0.9),
    rand=None,
    max_retries=50
):

    N = len(items)
    assert N >= 3

    min_pass_count = 2 if N == 3 else 3
    low = max(pass_rate_range[0], min_pass_count / N)
    high = pass_rate_range[1]

    passed_indices = []
    num_tries = 0
    while len(passed_indices) / N < low or len(passed_indices) / N > high:
        if num_tries > max_retries:
            return None,None,None  
        num_tries += 1
        
        operators = ["HAS","NOT_HAS","HAS_ANY","NOT_HAS_ANY","HAS_ALL","NOT_HAS_ALL"]
        operator = rand.choice(operators)

        item2indices = defaultdict(list)
        for lst_idx,lst in enumerate(items): 
            for item_idx,item in enumerate(lst):
                item2indices[item].append((lst_idx, item_idx))

        universe = list(item2indices.keys())
        assert universe
        if operator in {"HAS", "NOT_HAS"}:
            k = 1
        else:
            k = rand.choice([2, 3])
            k = min(k, int(len(universe)/2))
            k = max(k, 2)

        reference = rand.sample(universe, k)
        reference_idx = [item2indices[ref][0] for ref in reference]
        assert reference 
        
        def passes_constraint(lst):
            if operator == "HAS":         return reference[0] in lst
            if operator == "NOT_HAS":     return reference[0] not in lst
            if operator == "HAS_ANY":     return any(ref in lst for ref in reference)
            if operator == "NOT_HAS_ANY": return not any(ref in lst for ref in reference)
            if operator == "HAS_ALL":     return all(ref in lst for ref in reference)
            if operator == "NOT_HAS_ALL": return not all(ref in lst for ref in reference)

        passed_indices = [
            idx for idx, item_lst in enumerate(items)
            if passes_constraint(item_lst)
        ]

    return passed_indices, reference_idx, operator
