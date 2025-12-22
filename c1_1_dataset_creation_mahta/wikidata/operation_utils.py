import numpy as np
from statistics import mode, median


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
