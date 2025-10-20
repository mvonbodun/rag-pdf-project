# src/ir_bm25.py
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, chunks):  # chunks: List[{"id":..., "text":...}]
        self.ids = [c["id"] for c in chunks]
        self.tokens = [c["text"].split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, top_k: int):
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]
