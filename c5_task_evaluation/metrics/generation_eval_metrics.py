import re


def normalize_number(value, decimals=2):
    """Rounds numeric value to fixed decimals for comparison."""
    import math
    if value is None or math.isnan(value):
        return None
    return round(float(value), decimals)


def safe_float_convert(value):
    """Safely convert a value to float, return None if not possible."""
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# Regex to find first number in text (int or decimal, optional minus, optional comma thousands)
_NUMBER_RE = re.compile(r'[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d*\.\d+)')


def extract_number_from_string(s):
    """
    Extract the first number from a string, stripping text/units.
    E.g. "8269 km" -> 8269.0, "8270.79237550612" -> 8270.79237550612, "The value is 1,234.5" -> 1234.5.
    Returns float or None if no number found.
    """
    if s is None or (isinstance(s, str) and s.strip() == ''):
        return None
    s = str(s).strip()
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    num_str = m.group(0).replace(',', '')
    try:
        return float(num_str)
    except ValueError:
        return None


def clean_prediction_for_numeric(prediction):
    """
    Extract only the numeric part from prediction (remove text/units).
    E.g. "8269 km" -> "8269", "1,234.5 units" -> "1234.5".
    Returns string suitable for comparison; if no number found, returns original string.
    """
    num = extract_number_from_string(prediction)
    if num is None:
        return str(prediction).strip() if prediction is not None else ''
    # Prefer integer representation when equal
    if num == int(num):
        return str(int(num))
    return str(num)


def clean_gold_for_numeric(gold, decimals=2):
    """
    Normalize gold answer to a numeric string with fixed decimals.
    E.g. "8270.79237550612" -> "8270.79", "8269 km" -> "8269.00".
    Returns string; if no number found, returns original string stripped.
    """
    num = extract_number_from_string(gold)
    if num is None:
        return str(gold).strip() if gold is not None else ''
    return f'{num:.{decimals}f}'


def soft_exact_match(prediction, gold, decimals=3, tolerance_pct=None):
    """
    Compute soft exact match with optional tolerance.

    Prediction and gold are cleaned for numeric comparison when possible:
    - Prediction: text/units stripped, first number extracted (e.g. "8269 km" -> 8269).
    - Gold: first number extracted and rounded to 2 decimals for comparison (e.g. "8270.79237550612" -> 8270.79).

    Args:
        prediction: Predicted value (str, float, or None)
        gold: Gold answer value (str, float, or None)
        decimals: Number of decimals for rounding in comparison (default: 3)
        tolerance_pct: Tolerance percentage (e.g., 5.0 for 5% tolerance).
                      If None, uses exact match only.

    Returns:
        dict: Contains 'exact_match' (bool) and if tolerance_pct provided, 'soft_match' (bool)
    """
    # Try to extract numbers from text (e.g. "8269 km" -> 8269, "8270.792..." -> 8270.79...)
    pred_float = extract_number_from_string(prediction)
    gold_float = extract_number_from_string(gold)

    # Fallback: pure numeric string without extra text
    if pred_float is None:
        pred_float = safe_float_convert(prediction)
    if gold_float is None:
        gold_float = safe_float_convert(gold)

    # If either is None, no match
    if pred_float is None or gold_float is None:
        result = {'exact_match': False}
        if tolerance_pct is not None:
            result['soft_match'] = False
        return result

    # Normalize numbers for comparison (gold rounded to 2 decimals for canonical form)
    npred = normalize_number(pred_float, decimals)
    ngold = normalize_number(gold_float, decimals)

    if npred is None or ngold is None:
        result = {'exact_match': False}
        if tolerance_pct is not None:
            result['soft_match'] = False
        return result

    # Exact match
    exact = (npred == ngold)
    result = {'exact_match': exact}

    # Soft match with tolerance
    if tolerance_pct is not None:
        abs_err = abs(npred - ngold)
        soft = abs_err <= (tolerance_pct / 100.0) * abs(ngold) if ngold != 0 else (abs_err == 0)
        result['soft_match'] = soft

    return result
