import numpy as np
from typing import List, Dict, Any, Set


def extract_entity_id(passage_id: str) -> str:
    """
    Extract entity ID from a passage ID.
    
    Passage IDs are formatted as 'entity_id-passage_number' (e.g., '15641-0001').
    This function extracts the entity_id part (e.g., '15641').
    
    Args:
        passage_id: Passage ID in format 'entity_id-passage_number'
        
    Returns:
        str: The entity ID
    """
    # Split on '-' and take all but the last part (in case entity_id contains '-')
    parts = passage_id.rsplit('-', 1)
    return parts[0] if len(parts) > 1 else passage_id


def entity_recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = None) -> float:
    """
    Calculate Entity Recall@k - the proportion of unique entities covered by retrieved passages.
    
    Unlike standard recall which counts passages, entity recall counts unique entities.
    Multiple passages from the same entity only count as covering that entity once.
    
    Example:
        - Query has gold passages from 2 entities (15641, 42558)
        - Retrieving 5 passages all from entity 15641 gives recall = 1/2 = 0.5
        - Retrieving 1 passage from each entity gives recall = 2/2 = 1.0
    
    Args:
        retrieved_ids: List of retrieved passage IDs (in ranked order)
        gold_ids: List of gold/relevant passage IDs
        k: Number of top passages to consider (if None, use all retrieved)
        
    Returns:
        float: Entity recall score (0.0 to 1.0)
    """
    if not gold_ids:
        return 0.0
    
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    
    # Extract unique entities from gold passages
    gold_entities = set(extract_entity_id(pid) for pid in gold_ids)
    
    # Extract unique entities from retrieved passages
    retrieved_entities = set(extract_entity_id(pid) for pid in retrieved_ids)
    
    # Count how many gold entities are covered
    covered_entities = gold_entities & retrieved_entities
    
    return len(covered_entities) / len(gold_entities)


def entity_hit_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = None) -> float:
    """
    Calculate Entity Hit@k - binary indicator of whether at least one gold entity is covered.
    
    Args:
        retrieved_ids: List of retrieved passage IDs (in ranked order)
        gold_ids: List of gold/relevant passage IDs
        k: Number of top passages to consider (if None, use all retrieved)
        
    Returns:
        float: Entity hit rate (0.0 or 1.0)
    """
    if not gold_ids or not retrieved_ids:
        return 0.0
    
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    
    gold_entities = set(extract_entity_id(pid) for pid in gold_ids)
    retrieved_entities = set(extract_entity_id(pid) for pid in retrieved_ids)
    
    return 1.0 if len(gold_entities & retrieved_entities) > 0 else 0.0


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


def evaluate_entity_retrieval(retrieved_ids: List[str], gold_ids: List[str], k_values: List[int] = [1, 3, 5, 10, 20, 50, 100]) -> Dict[str, Any]:
    """
    Compute entity-based retrieval metrics for a single query.
    
    Entity recall measures how many unique entities (Wikipedia pages) are covered
    by retrieved passages, not how many individual passages are retrieved.
    
    Args:
        retrieved_ids: List of retrieved passage IDs (in ranked order)
        gold_ids: List of gold/relevant passage IDs
        k_values: List of k values to compute metrics for
        
    Returns:
        dict: Dictionary containing entity-based retrieval metrics
    """
    # Extract entities
    gold_entities = set(extract_entity_id(pid) for pid in gold_ids)
    retrieved_entities = set(extract_entity_id(pid) for pid in retrieved_ids)
    
    metrics = {
        'num_retrieved_passages': len(retrieved_ids),
        'num_gold_passages': len(gold_ids),
        'num_gold_entities': len(gold_entities),
        'num_retrieved_entities': len(retrieved_entities),
    }
    
    # Compute entity recall at different k values
    for k in k_values:
        metrics[f'entity_recall@{k}'] = entity_recall_at_k(retrieved_ids, gold_ids, k)
        metrics[f'entity_hit@{k}'] = entity_hit_at_k(retrieved_ids, gold_ids, k)
    
    # Overall entity recall (all retrieved)
    metrics['entity_recall'] = entity_recall_at_k(retrieved_ids, gold_ids)
    
    return metrics


def load_qrels(qrel_file_path: str) -> Dict[str, List[str]]:
    """
    Load qrels from a TREC-format file.
    
    Qrel format: query_id 0 passage_id relevance
    Example: 13_p569 0 15641-0001 1
    
    Args:
        qrel_file_path: Path to the qrel file
        
    Returns:
        dict: Mapping from query_id to list of relevant passage IDs
    """
    qrels = {}
    with open(qrel_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                passage_id = parts[2]
                relevance = int(parts[3])
                
                if relevance > 0:
                    if query_id not in qrels:
                        qrels[query_id] = []
                    qrels[query_id].append(passage_id)
    
    return qrels


def get_gold_entities_per_query(qrels: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """
    Extract unique gold entities for each query from qrels.
    
    Args:
        qrels: Mapping from query_id to list of relevant passage IDs
        
    Returns:
        dict: Mapping from query_id to set of gold entity IDs
    """
    return {
        query_id: set(extract_entity_id(pid) for pid in passage_ids)
        for query_id, passage_ids in qrels.items()
    }
