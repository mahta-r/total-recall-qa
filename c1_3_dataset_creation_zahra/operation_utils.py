import numpy as np
from statistics import median
from collections import defaultdict




def numerical_filter(
    values,
    pass_rate_range=(0.01, 0.9),
    rand=None,
    max_retries=50
):
    N = len(values)
    assert N >= 3
    
    min_pass_count = 2 if N == 3 else 3
    low = max(pass_rate_range[0], min_pass_count / N)
    high = pass_rate_range[1]

    unique_values = list(set(val for val, idx in values))
    if len(unique_values) < 2:
        return None, None, None

    passed_indices = []
    num_tries = 0
    while len(passed_indices) / N < low or len(passed_indices) / N > high:
        if num_tries > max_retries:
            return None,None,None  
        num_tries += 1

        direction = rand.choice(["GT", "GTE", "LT", "LTE"])
        # reference, reference_idx = rand.choice(values)
        reference = rand.choice(unique_values)
        
        def passes_constraint(value):
            if direction == "GT":  return value > reference
            if direction == "GTE": return value >= reference
            if direction == "LT":  return value < reference
            if direction == "LTE": return value <= reference
        
        passed_indices = [idx for val, idx in values if passes_constraint(val)]

    return passed_indices, reference, direction



def temporal_filter(
    datetimes,
    pass_rate_range=(0.01, 0.9),
    rand=None,
    max_retries=50
):
    N = len(datetimes)
    assert N >= 3

    min_pass_count = 2 if N == 3 else 3
    low = max(pass_rate_range[0], min_pass_count / N)
    high = pass_rate_range[1]

    unique_dates = list(set(date for date, idx in datetimes))
    if len(unique_dates) < 2:
        return None, None, None
    
    passed_indices = []
    num_tries = 0
    while len(passed_indices) / N < low or len(passed_indices) / N > high:
        if num_tries > max_retries:
            return None,None,None  
        num_tries += 1
        
        direction = rand.choice(["BEFORE", "AFTER"])
        # reference, reference_idx = rand.choice(datetimes)
        reference = rand.choice(unique_dates)

        def passes_constraint(value):
            if direction == "AFTER":  return value > reference
            if direction == "BEFORE": return value < reference

        passed_indices = [idx for val, idx in datetimes if passes_constraint(val)]

    return passed_indices, reference, direction



def ordered_type_filter(
    types,
    pass_rate_range=(0.01, 0.9),
    rand=None,
    max_retries=50,
    sort_order_key=None
):
    if sort_order_key is None:
        raise ValueError("sort_order_key must be provided for constraints based on OrderedString")
    
    N = len(types)
    assert N >= 3

    min_pass_count = 2 if N == 3 else 3
    low = max(pass_rate_range[0], min_pass_count / N)
    high = pass_rate_range[1]

    all_types = list(sort_order_key.keys())
    valid_references = sorted(all_types, key=lambda t: sort_order_key[t])[1:-1]
    if len(valid_references) < 1:
        return None, None, None

    passed_indices = []
    num_tries = 0
    while len(passed_indices) / N < low or len(passed_indices) / N > high:
        if num_tries > max_retries:
            return None,None,None  
        num_tries += 1
        
        direction = rand.choice(["MORE_THAN", "LESS_THAN"])
        # reference, reference_idx = rand.choice(types)
        reference = rand.choice(valid_references)

        def passes_constraint(value):
            if direction == "MORE_THAN":  return sort_order_key[value] > sort_order_key[reference]
            if direction == "LESS_THAN":  return sort_order_key[value] < sort_order_key[reference]

        passed_indices = [idx for val, idx in types if passes_constraint(val)]

    return passed_indices, reference, direction



def type_filter(
    types,
    pass_rate_range=(0.01, 0.9),
    rand=None,
    max_retries=50
):
    N = len(types)
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
        
        type2indices = defaultdict(list)
        for type_str,idx in types:
            type2indices[type_str].append(idx)

        universe = list(type2indices.keys())
        assert universe

        if len(universe) >= 2:
            # operators = ["IS","IS_NOT","IS_ANY","IS_NOT_ANY"]
            operators = ["IS","IS_NOT","IS_ANY"]
        else:
            operators = ["IS","IS_NOT"]
        operator = rand.choice(operators)

        if operator in {"IS", "IS_NOT"}:
            k = 1
        else:
            k = rand.choice([2, 3])
            k = min(k, len(universe))

        # sample a contiguous span of length k from universe
        if k == 1:
            reference = [rand.choice(universe)]
        else:
            start = rand.randint(0, len(universe) - k)
            reference = universe[start:start + k]
        # reference = rand.sample(universe, k)
        assert reference
        
        def passes_constraint(type_str):
            if operator == "IS":         return type_str == reference[0]
            if operator == "IS_NOT":     return type_str != reference[0]
            if operator == "IS_ANY":     return any(ref == type_str for ref in reference)
            if operator == "IS_NOT_ANY": return not any(ref == type_str for ref in reference)

        passed_indices = [
            idx for type_str,idx in types
            if passes_constraint(type_str)
        ]

    return passed_indices, reference, operator



# ================================================= VALIDATION =================================================

def is_valid_for_values(operation, input_entity_values):
    """Check if an aggregation operation can be applied to the given entity values."""
    n = len(input_entity_values)

    if operation in {"SUM_TOP_K", "SUM_BOTTOM_K", "AVG_TOP_K", "AVG_BOTTOM_K", "MEDIAN"}:
        return n >= 3

    elif operation in {"SUM", "AVG", "MAX"}:
        return n >= 2

    elif operation == "MIN":
        values = [ev['value_node']['operation_value'] for ev in input_entity_values]
        return n >= 2 and min(values) != 0

    elif operation == "DIFFERENCE(MAX-MIN)":
        values = [ev['value_node']['operation_value'] for ev in input_entity_values]
        return n >= 2 and max(values) != min(values)
    
    elif operation == "MOST_COMMON":
        values = [ev['value_node']['operation_value'] for ev in input_entity_values]
        value_counts = defaultdict(int)
        for v in values:
            value_counts[v] += 1
        most_common_count = max(value_counts.values())
        num_most_common = sum(c == most_common_count for c in value_counts.values())
        return n >= 3 and num_most_common == 1

    elif operation == "RATIO(MAX/MIN)":
        values = [ev['value_node']['operation_value'] for ev in input_entity_values]
        return n >= 2 and min(values) != 0 and max(values) != 0

    elif operation in {"EARLIEST", "LATEST", "TIME_BETWEEN_FIRST_LAST"}:
        return n >= 2

    elif operation in {"NTH_EARLIEST", "NTH_LATEST"}:
        return n >= 3

    elif operation == "PERCENTAGE":
        return n >= 2

    else:
        raise ValueError(f"Unknown operation: {operation}")


# ================================================= AGGREGATION FUNCTIONS =================================================

def numerical_aggregation(operation, input_entity_values, rand=None):
    """
    Apply a numerical aggregation operation to entity values.
    Returns (operation_name, operation_args_string, float_result).
    """
    values = [ev['value_node']['operation_value'] for ev in input_entity_values]
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
    
    elif op == "MOST_COMMON":
        value_counts = defaultdict(int)
        for v in arr:
            value_counts[v] += 1
        mode_value = max(values, key=lambda v: value_counts[v])
        return op, "", float(mode_value)

    elif op == "DIFFERENCE(MAX-MIN)":
        return op, "", float(np.max(arr) - np.min(arr))

    elif op == "RATIO(MAX/MIN)":
        return op, "", float(np.max(arr) / np.min(arr))

    else:
        raise ValueError(f"Unknown numerical operation: {operation}")


def temporal_aggregation(operation, input_entity_values, rand=None):
    """
    Apply a temporal aggregation operation to entity values.
    Dates are integer years. Returns (operation_name, args_string, result).
    """
    years = [ev['value_node']['operation_value'] for ev in input_entity_values]
    arr = sorted(years)
    op = operation.upper()

    if op == "EARLIEST":
        return op, "", arr[0]

    elif op == "LATEST":
        return op, "", arr[-1]

    elif op == "NTH_EARLIEST":
        n = rand.randint(2, max(2, int(len(arr) / 2)))
        return op, f"N={n}", arr[n - 1]

    elif op == "NTH_LATEST":
        n = rand.randint(2, max(2, int(len(arr) / 2)))
        return op, f"N={n}", arr[-n]

    elif op == "TIME_BETWEEN_FIRST_LAST":
        span = arr[-1] - arr[0]
        return op, "Unit=years", span

    else:
        raise ValueError(f"Unknown temporal operation: {operation}")


def string_aggregation(operation, input_entity_values, rand=None):
    """
    Apply a string aggregation operation (PERCENTAGE) to entity values.
    Returns (operation_name, args_string, float_percentage).
    """
    values = [ev['value_node']['operation_value'] for ev in input_entity_values]
    op = operation.upper()

    if op == "PERCENTAGE":
        value_counts = defaultdict(int)
        for v in values:
            value_counts[v] += 1

        # prefer values that aren't 100% or too rare
        candidates = [v for v, c in value_counts.items()
                      if 0.1 <= c / len(values) <= 0.9]
        if not candidates:
            candidates = list(value_counts.keys())

        selected_value = rand.choice(candidates)
        percentage = 100.0 * value_counts[selected_value] / len(values)
        return op, f"Value={selected_value}", percentage

    else:
        raise ValueError(f"Unknown string operation: {operation}")


def count_aggregation(input_entity_values):
    """
    Property-less aggregation: just count the entities.
    Returns (operation_name, args_string, int_count).
    """
    return "COUNT", "", len(input_entity_values)
