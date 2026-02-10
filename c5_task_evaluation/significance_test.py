#!/usr/bin/env python3
"""
Statistical significance testing for retrieval or generation runs.

Compares two systems using per-query metrics. Uses paired Wilcoxon signed-rank
test (non-parametric) and optional bootstrap for a 95% CI of the difference.

Retrieval metrics: entity_recall@k (e.g. entity_recall@10).
Generation metrics: exact_match, soft_match_1.0, soft_match_5.0, soft_match_10.0, etc.

Usage (retrieval):
    python c5_task_evaluation/significance_test.py \
        run_output/run_1/qald10_quest_test/retrieval_bm25/evaluation_results_per_query_metrics.jsonl \
        run_output/run_1/qald10_quest_test/retrieval_spladepp/evaluation_results_per_query_metrics.jsonl \
        --metric entity_recall@10 --names bm25 spladepp

Usage (generation):
    python c5_task_evaluation/significance_test.py \
        run_output/run_1/qald10_quest_test/generation_gpt-5.2_single_retrieval_e5/evaluation_results_per_query_metrics.jsonl \
        run_output/run_1/qald10_quest_test/generation_gpt-5.2_single_retrieval_oracle/evaluation_results_per_query_metrics.jsonl \
        --metric exact_match --names e5 oracle
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_per_query_metrics(path: str) -> dict:
    """Load per-query metrics from JSONL. Returns dict qid -> {metric: value, ...}."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("qid")
            if qid is not None:
                data[qid] = row
    return data


def get_metric_value(row: dict, metric: str):
    """Get metric value from a row. Supports soft_match_X by reading from soft_matches dict."""
    if metric.startswith("soft_match_") and metric != "soft_match":
        try:
            pct = float(metric.replace("soft_match_", ""))
        except ValueError:
            return row.get(metric)
        soft_matches = row.get("soft_matches") or {}
        val = soft_matches.get(pct) if isinstance(soft_matches, dict) else None
        if val is None:
            val = soft_matches.get(str(pct))
        return 1.0 if val else 0.0
    return row.get(metric)


def paired_wilcoxon(scores_a: np.ndarray, scores_b: np.ndarray):
    """Paired Wilcoxon signed-rank test. Returns statistic, p-value (two-sided)."""
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(scores_a, scores_b, alternative="two-sided")
        return stat, p
    except ImportError:
        return None, None


def bootstrap_ci_diff(scores_a: np.ndarray, scores_b: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95):
    """Bootstrap 95% CI for the difference (mean_b - mean_a). Returns (low, high)."""
    n = len(scores_a)
    diffs = scores_b - scores_a
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs.append(np.mean(diffs[idx]))
    boot_diffs = np.array(boot_diffs)
    alpha = 1 - ci
    low = np.percentile(boot_diffs, 100 * alpha / 2)
    high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return low, high


def main():
    parser = argparse.ArgumentParser(
        description="Paired significance test for two retrieval per-query metric files."
    )
    parser.add_argument(
        "metrics_a",
        type=str,
        help="Path to first system's evaluation_results_per_query_metrics.jsonl",
    )
    parser.add_argument(
        "metrics_b",
        type=str,
        help="Path to second system's evaluation_results_per_query_metrics.jsonl",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="entity_recall@10",
        help="Metric to compare (e.g. entity_recall@10, entity_recall@100, entity_recall)",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs=2,
        default=["SystemA", "SystemB"],
        metavar=("NAME_A", "NAME_B"),
        help="Display names for the two systems",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for CI (0 to disable). Default: 10000",
    )
    args = parser.parse_args()

    path_a = Path(args.metrics_a)
    path_b = Path(args.metrics_b)
    if not path_a.exists():
        print(f"Error: File not found: {path_a}", file=sys.stderr)
        sys.exit(1)
    if not path_b.exists():
        print(f"Error: File not found: {path_b}", file=sys.stderr)
        sys.exit(1)

    name_a, name_b = args.names
    data_a = load_per_query_metrics(str(path_a))
    data_b = load_per_query_metrics(str(path_b))

    common_qids = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not common_qids:
        print("Error: No common query IDs between the two files.", file=sys.stderr)
        sys.exit(1)

    metric = args.metric
    def _score(data, q):
        v = get_metric_value(data[q], metric)
        return float(v) if v is not None else np.nan
    scores_a = np.array([_score(data_a, q) for q in common_qids])
    scores_b = np.array([_score(data_b, q) for q in common_qids])

    if np.any(np.isnan(scores_a)) or np.any(np.isnan(scores_b)):
        print(f"Warning: Some queries missing metric '{metric}'; filling with 0.", file=sys.stderr)
        scores_a = np.nan_to_num(scores_a, nan=0.0)
        scores_b = np.nan_to_num(scores_b, nan=0.0)

    n = len(common_qids)
    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))
    diff = mean_b - mean_a

    print(f"\n=== Significance test: {name_a} vs {name_b} ===")
    print(f"Metric: {metric}")
    print(f"Queries (paired): {n}")
    print(f"Mean {name_a}: {mean_a:.4f}")
    print(f"Mean {name_b}: {mean_b:.4f}")
    print(f"Difference (B - A): {diff:+.4f}")

    stat, p_value = paired_wilcoxon(scores_a, scores_b)
    if p_value is not None:
        print(f"\nPaired Wilcoxon signed-rank test (two-sided):")
        print(f"  p-value: {p_value:.4f}")
        significant = p_value < args.alpha
        print(f"  Significant at alpha={args.alpha}: {'Yes' if significant else 'No'}")
    else:
        print("\nInstall scipy for Wilcoxon test: pip install scipy")

    if args.bootstrap > 0:
        low, high = bootstrap_ci_diff(scores_a, scores_b, n_bootstrap=args.bootstrap)
        print(f"\nBootstrap 95% CI for difference (B - A): [{low:.4f}, {high:.4f}]")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
