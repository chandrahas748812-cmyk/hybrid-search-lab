"""BM25, dense, and hybrid retrievers with score fusion."""

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.bm25 = BM25Okapi([d["text"].lower().split() for d in documents])

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i]["id"], float(scores[i])) for i in idx]


class DenseRetriever:
    def __init__(self, documents, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [d["text"] for d in documents], normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i]["id"], float(scores[i])) for i in idx]


def _minmax(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    """Fuses BM25 and dense rankings via normalized weighted sum."""

    def __init__(self, documents, dense: DenseRetriever | None = None):
        self.bm25 = BM25Retriever(documents)
        self.dense = dense
        self.doc_ids = [d["id"] for d in documents]

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> list[tuple[str, float]]:
        n = len(self.doc_ids)
        b = dict(self.bm25.search(query, top_k=n))
        b_norm = dict(zip(b.keys(), _minmax(list(b.values()))))

        if self.dense is not None:
            d = dict(self.dense.search(query, top_k=n))
            d_norm = dict(zip(d.keys(), _minmax(list(d.values()))))
        else:
            d_norm = {}

        fused = {
            doc_id: (1 - alpha) * b_norm.get(doc_id, 0.0) + alpha * d_norm.get(doc_id, 0.0)
            for doc_id in self.doc_ids
        }
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
