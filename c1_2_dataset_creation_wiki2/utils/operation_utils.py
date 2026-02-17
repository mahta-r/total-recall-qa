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
  "RATIO(MAX/MIN)": "ratio between the largest and smallest values, showing relative range",
  "COUNT": "count / number of entities",
  "EARLIEST": "earliest / minimum value (for time/dates)",
  "LATEST": "latest / maximum value (for time/dates)"
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
    elif op == "MAX" or op == "LATEST":
        return float(np.max(arr))
    elif op == "MIN" or op == "EARLIEST":
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
    elif op == "COUNT":
        return len(arr)
    else:
        raise ValueError(f"Unknown operation: {operation}")


# Property to operation mapping based on datatype
def get_operations_for_datatype(datatype):
    """
    Get valid operations for a given datatype.

    Args:
        datatype: Property datatype (Time, Quantity, GlobeCoordinate, WikibaseItem)

    Returns:
        List of valid operation names
    """
    if datatype == "Time":
        return ["EARLIEST", "LATEST", "AVG"]
    elif datatype == "Quantity":
        return ["SUM", "AVG", "MAX", "MIN", "COUNT"]
    elif datatype == "WikibaseItem":
        # For entity lists, COUNT is the only valid operation
        # This counts how many entities have a specific value
        return ["COUNT"]
    elif datatype == "GlobeCoordinate":
        return ["COUNT"]
    else:
        return ["COUNT"]
