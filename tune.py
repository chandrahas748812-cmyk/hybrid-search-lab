"""Grid search over blend weight (alpha) and retrieval depth (top_k)."""

from src.corpus import DOCUMENTS, QUERIES
from src.search import DenseRetriever, HybridRetriever
from src.metrics import precision_at_k

print("Loading dense model...")
try:
    dense = DenseRetriever(DOCUMENTS)
except Exception as e:
    print(f"dense unavailable ({e}); exiting")
    raise SystemExit(1)

hybrid = HybridRetriever(DOCUMENTS, dense=dense)

print(f"{'alpha':>6} {'top_k':>6} {'P@3':>6}")
print("-" * 22)
best = (0, None, None)
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    for top_k in [3, 5, 10]:
        scores = []
        for query, relevant in QUERIES.items():
            ids = [d for d, _ in hybrid.search(query, top_k=top_k, alpha=alpha)]
            scores.append(precision_at_k(ids, relevant, 3))
        avg = sum(scores) / len(scores)
        print(f"{alpha:>6.2f} {top_k:>6d} {avg:>6.3f}")
        if avg > best[0]:
            best = (avg, alpha, top_k)

print(f"\nBest: P@3={best[0]:.3f} at alpha={best[1]}, top_k={best[2]}")
