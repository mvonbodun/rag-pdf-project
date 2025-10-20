# src/config.py
from pydantic import BaseModel, Field
from typing import List, Literal

class QAItem(BaseModel):
    id: str
    question: str
    relevant_chunk_ids: List[str]
    granularity: Literal["paragraph","section","page","multi-hop"] = "paragraph"
    difficulty: Literal["easy","medium","hard"] = "medium"
