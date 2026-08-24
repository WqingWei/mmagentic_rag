from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Chunk(BaseModel):
    chunk_id: str
    manual_id: str
    section_title: str = ""
    section_path: str = ""
    text: str
    image_ids: List[str] = Field(default_factory=list)
    source_file: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(Chunk):
    score: float = 0.0
    score_source: str = ""
    rerank_score: Optional[float] = None
