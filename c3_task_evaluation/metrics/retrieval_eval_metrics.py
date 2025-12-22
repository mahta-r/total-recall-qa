import numpy as np
from typing import List, Dict, Any


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = None) -> float:
    """
    Calculate Recall@k - the proportion of relevant documents retrieved.

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        gold_ids: List of gold/relevant document IDs
        k: Number of top documents to consider (if None, use all retrieved docs)

    Returns:
        float: Recall@k score (0.0 to 1.0)
    """
    if not gold_ids:
        return 0.0

    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)

    intersection = len(retrieved_set & gold_set)
    recall = intersection / len(gold_set)

    return recall


def precision_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = None) -> float:
    """
    Calculate Precision@k - the proportion of retrieved documents that are relevant.

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        gold_ids: List of gold/relevant document IDs
        k: Number of top documents to consider (if None, use all retrieved docs)

    Returns:
        float: Precision@k score (0.0 to 1.0)
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if not retrieved_ids:
        return 0.0

    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)

    intersection = len(retrieved_set & gold_set)
    precision = intersection / len(retrieved_ids)

    return precision


def mean_reciprocal_rank(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) - the reciprocal of the rank of the first relevant document.

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        gold_ids: List of gold/relevant document IDs

    Returns:
        float: MRR score (0.0 to 1.0)
    """
    if not gold_ids or not retrieved_ids:
        return 0.0

    gold_set = set(gold_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank

    return 0.0


def average_precision(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """
    Calculate Average Precision (AP) - the average of precision values at each relevant document position.

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        gold_ids: List of gold/relevant document IDs

    Returns:
        float: AP score (0.0 to 1.0)
    """
    if not gold_ids or not retrieved_ids:
        return 0.0

    gold_set = set(gold_ids)
    num_relevant = 0
    sum_precisions = 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            num_relevant += 1
            precision_at_rank = num_relevant / rank
            sum_precisions += precision_at_rank

    if num_relevant == 0:
        return 0.0

    return sum_precisions / len(gold_set)


def ndcg_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        gold_ids: List of gold/relevant document IDs
        k: Number of top documents to consider (if None, use all retrieved docs)

    Returns:
        float: NDCG@k score (0.0 to 1.0)
    """
    if not gold_ids or not retrieved_ids:
        return 0.0

    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    gold_set = set(gold_ids)

    # Calculate DCG
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        relevance = 1.0 if doc_id in gold_set else 0.0
        dcg += relevance / np.log2(rank + 1)

    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    num_relevant = min(len(gold_ids), len(retrieved_ids))
    for rank in range(1, num_relevant + 1):
        idcg += 1.0 / np.log2(rank + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = None) -> float:
    """
    Calculate Hit Rate@k - binary indicator of whether at least one relevant document is retrieved.

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        gold_ids: List of gold/relevant document IDs
        k: Number of top documents to consider (if None, use all retrieved docs)

    Returns:
        float: Hit rate (0.0 or 1.0)
    """
    if not gold_ids or not retrieved_ids:
        return 0.0

    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)

    return 1.0 if len(retrieved_set & gold_set) > 0 else 0.0


def evaluate_retrieval_ranking(retrieved_ids: List[str], gold_ids: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    """
    Compute multiple retrieval metrics for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        gold_ids: List of gold/relevant document IDs
        k_values: List of k values to compute metrics for

    Returns:
        dict: Dictionary containing various retrieval metrics
    """
    metrics = {
        'num_retrieved': len(retrieved_ids),
        'num_gold': len(gold_ids),
        'mrr': mean_reciprocal_rank(retrieved_ids, gold_ids),
        'map': average_precision(retrieved_ids, gold_ids),
    }

    # Compute metrics at different k values
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, gold_ids, k)
        metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, gold_ids, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, gold_ids, k)
        metrics[f'hit@{k}'] = hit_rate_at_k(retrieved_ids, gold_ids, k)

    # Also compute overall metrics without k limit
    metrics['recall'] = recall_at_k(retrieved_ids, gold_ids)
    metrics['precision'] = precision_at_k(retrieved_ids, gold_ids)
    metrics['ndcg'] = ndcg_at_k(retrieved_ids, gold_ids)

    return metrics
