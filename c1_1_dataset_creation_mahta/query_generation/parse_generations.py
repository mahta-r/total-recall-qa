import re

def extract_query_text(text):
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