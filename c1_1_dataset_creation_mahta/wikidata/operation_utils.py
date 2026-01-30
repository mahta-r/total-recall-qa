from collections import defaultdict
import json
import numpy as np
from statistics import mode, median
import random
import math


operation_descriptions = {
  "SUM": "sum / total of all values ",
  "AVG": "average / arithmetic mean of all values",
  "MAX": "max / largest / highest value among all entities",
  "MIN": "min / smallest / lowest value among all entities",
  "MEDIAN": "median of all values",
  "MODE": "mode / most frequently occurring value",
  "VARIANCE": "variance of all values",
  "STDDEV": "standard deviation of all values",
  "MAX_DIFF": "difference between the largest and smallest values",
  "RATIO(MAX/MIN)": "ratio between the largest and smallest values, showing relative range"
}


def apply_operation(operation, values):
    if not values:
        return None

    arr = np.array(values, dtype=float)
    op = operation.upper()

    if op == "SUM":
        return float(np.sum(arr))
    elif op == "AVG":
        return float(np.mean(arr))
    elif op == "MAX":
        return float(np.max(arr))
    elif op == "MIN":
        return float(np.min(arr))
    elif op == "MEDIAN":
        return float(median(arr))
    elif op == "VARIANCE":
        return float(np.var(arr))
    elif op == "STDDEV":
        return float(np.std(arr))
    elif op == "MAX_DIFF":
        return float(np.max(arr) - np.min(arr))
    elif op == "RATIO(MAX/MIN)":
        min_val = np.min(arr)
        return float(np.max(arr) / min_val) if min_val != 0 else None
    else:
        raise ValueError(f"Unknown operation: {operation}")
    



def coordinates_filter(
    coordinates,
    pass_rate_range=(0.5, 0.9),
    seed=42
):  
    rand = random.Random(seed)

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
        # reference_idx = min(range(len(coordinates)), key=lambda idx: dist_to_ref(coordinates[idx][0]))
        # reference_lat, reference_lon = coordinates[reference_idx]
        
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
    seed=None
):
    rand = random.Random(seed)
    
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
    seed=None
):
    rand = random.Random(seed)

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
    seed=None,
    max_retries=50
):
    rand = random.Random(seed)

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
