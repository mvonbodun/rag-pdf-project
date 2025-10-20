# src/rerank.py
from typing import List, Tuple
import cohere

def cohere_rerank(query: str, candidates: List[Tuple[str, str]], top_k: int, api_key: str):
    """
    candidates: list of (chunk_id, text)
    returns: reordered list of (chunk_id, text)
    """
    client = cohere.Client(api_key)
    res = client.rerank(
        model="rerank-3.5",  # or 3 if 3.5 not available
        query=query,
        documents=[t for _, t in candidates],
        top_n=top_k
    )
    # cohere returns indices into the original candidate list
    ranked = [(candidates[r.index][0], candidates[r.index][1]) for r in res.results]
    return ranked
