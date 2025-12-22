import math
import json
import argparse
from typing import List, Tuple

from io_utils import read_jsonl_from_file


def normalize_number(value, decimals=2):
    """Rounds numeric value to fixed decimals for comparison."""
    if value is None or math.isnan(value):
        return None
    return round(float(value), decimals)


def evaluate_predictions(
    preds: List[float],
    golds: List[float],
    decimals: int = 3,
    tolerance_pct: List[float] = [1.0, 5.0, 10.0, 20.0, 50.0, 90.0]
) -> dict:

    assert len(preds) == len(golds), "Predictions and golds must align"

    n = len(preds)
    exact, abs_errs, rel_errs = 0, [], []
    soft = {pct: 0 for pct in tolerance_pct}
    records = []

    for i, (p, g) in enumerate(zip(preds, golds)):
        npred, ngold = normalize_number(p, decimals), normalize_number(g, decimals)

        if npred is None or ngold is None:
            records.append({"id": i, "pred": p, "gold": g, "exact": False, "soft": False})
            continue

        abs_err = abs(npred - ngold)
        rel_err = abs_err / abs(ngold) if ngold != 0 else float("inf")

        abs_errs.append(abs_err)
        rel_errs.append(rel_err)

        is_exact = npred == ngold
        is_soft = {}
        for pct in tolerance_pct:
            is_soft[pct] = abs_err <= (pct / 100.0) * abs(ngold)
            if is_soft[pct]:
                soft[pct] += 1

        if is_exact:
            exact += 1

        records.append({
            "id": i,
            "pred": npred,
            "gold": ngold,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "exact": is_exact,
            "soft": is_soft
        })

    summary = {
        "n": n,
        "exact_match": exact / n,
        "soft_accuracy": {pct: soft[pct] / n for pct in soft},
        "mean_abs_error": sum(abs_errs) / len(abs_errs) if abs_errs else None,
        "mean_rel_error": sum(rel_errs) / len(rel_errs) if rel_errs else None,
    }

    return summary, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_answers_path", 
        type=str, 
        required=True, 
        help="Path to answer results JSONL file"
    )
    parser.add_argument(
        "--decimals", 
        type=int, 
        default=3, 
        help="Number of decimals to round numeric values for comparison"
    )
    parser.add_argument(
        "--tolerance_pct", 
        type=float, 
        nargs='+', 
        default=[1.0, 5.0, 10.0, 20.0, 50.0, 90.0], 
        help="List of tolerance percentages for soft accuracy computation"
    )
    args = parser.parse_args()
    
    query_answers = read_jsonl_from_file(args.query_answers_path)
    predictions = [qa['predicted_answer'] for qa in query_answers]
    gold_answers = [qa['gold_answer'] for qa in query_answers]

    metrics, _ = evaluate_predictions(predictions, gold_answers, decimals=args.decimals, tolerance_pct=args.tolerance_pct)
    print(json.dumps(metrics, indent=2))