"""Cross-encoder reranker with graceful fallback."""

DOC_TEXT = {}  # filled by example.py


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception as e:
            print(f"(reranker unavailable: {e} — skipping rerank stage)")

    def rerank(self, query: str, doc_ids: list[str], top_k: int = 3) -> list[tuple[str, float]]:
        if self.model is None:
            return [(d, 0.0) for d in doc_ids[:top_k]]
        pairs = [(query, DOC_TEXT[d]) for d in doc_ids]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        return [(d, float(s)) for d, s in ranked[:top_k]]
