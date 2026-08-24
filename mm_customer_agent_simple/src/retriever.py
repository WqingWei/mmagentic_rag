from pathlib import Path
from typing import Dict, List

from .bm25_store import BM25Store
from .config import Settings
from .embedding import EmbeddingClient
from .qdrant_store import QdrantStore
from .schema import RetrievedChunk


def rrf_fuse(
    vector_hits,
    bm25_hits,
    rrf_k: int = 60,
) -> List[RetrievedChunk]:
    fused: Dict[str, RetrievedChunk] = {}
    scores: Dict[str, float] = {}
    sources: Dict[str, List[str]] = {}

    def add_hits(hits, source_name: str):
        for rank, (chunk, raw_score) in enumerate(hits, start=1):
            cid = chunk.chunk_id
            if cid not in fused:
                fused[cid] = RetrievedChunk(**chunk.model_dump(), score=0.0, score_source="")
                scores[cid] = 0.0
                sources[cid] = []
            scores[cid] += 1.0 / (rrf_k + rank)
            sources[cid].append(f"{source_name}:{raw_score:.4f}")

    add_hits(vector_hits, "vector")
    add_hits(bm25_hits, "bm25")

    results: List[RetrievedChunk] = []
    for cid, chunk in fused.items():
        chunk.score = scores[cid]
        chunk.score_source = ";".join(sources[cid])
        results.append(chunk)
    results.sort(key=lambda x: x.score, reverse=True)
    return results


class HybridRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = EmbeddingClient(settings)
        self.qdrant = QdrantStore(settings)
        self.bm25 = BM25Store.load(Path(settings.storage_dir) / "bm25.pkl")

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        query_vector = self.embedder.embed_query(query)
        vector_hits = self.qdrant.search(query_vector, top_k=self.settings.vector_top_k)
        bm25_hits = self.bm25.search(query, top_k=self.settings.bm25_top_k)
        candidates = rrf_fuse(vector_hits, bm25_hits, rrf_k=self.settings.rrf_k)
        return candidates
