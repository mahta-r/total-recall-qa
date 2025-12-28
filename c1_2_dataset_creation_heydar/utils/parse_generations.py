import re

def extract_query_text(text):
    """
    Extract the query text from LLM generation.

    Args:
        text: Generated text from LLM

    Returns:
        Query text string or None if invalid format
    """
    query_pattern = r"\[Query\]\s*(.+)"
    agg_pattern = r"\[Aggregation\]\s*(.+)"
    ans_pattern = r"\[Answer\]\s*(.+)"

    query_match = re.search(query_pattern, text)
    agg_match = re.search(agg_pattern, text)
    ans_match = re.search(ans_pattern, text)

    if not (query_match and agg_match and ans_match):
        print("Invalid format: missing one of [Query], [Aggregation], or [Answer]")
        return None

    query_text = query_match.group(1).strip()
    return query_text


def extract_aggregation(text):
    """
    Extract the aggregation operation from LLM generation.

    Args:
        text: Generated text from LLM

    Returns:
        Aggregation operation string or None if invalid format
    """
    agg_pattern = r"\[Aggregation\]\s*(.+)"
    agg_match = re.search(agg_pattern, text)

    if agg_match:
        return agg_match.group(1).strip()
    return None


def extract_answer(text):
    """
    Extract the answer from LLM generation.

    Args:
        text: Generated text from LLM

    Returns:
        Answer string or None if invalid format
    """
    ans_pattern = r"\[Answer\]\s*(.+)"
    ans_match = re.search(ans_pattern, text)

    if ans_match:
        answer_str = ans_match.group(1).strip()
        # Try to parse as number
        try:
            # Remove any non-numeric characters except decimal point and minus
            numeric_str = re.sub(r'[^\d.-]', '', answer_str)
            if numeric_str:
                return float(numeric_str) if '.' in numeric_str else int(numeric_str)
        except ValueError:
            pass
        return answer_str
    return None
