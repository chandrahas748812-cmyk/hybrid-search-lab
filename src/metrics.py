"""precision@k and MRR."""


def precision_at_k(retrieved: list[str], relevant: list[str], k: int = 3) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0
