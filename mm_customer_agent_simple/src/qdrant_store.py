from typing import List, Tuple
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .config import Settings
from .schema import Chunk


class QdrantStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
        self.collection = settings.qdrant_collection

    @staticmethod
    def _point_id(chunk: Chunk) -> str:
        return str(uuid5(NAMESPACE_URL, chunk.chunk_id))

    def recreate_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.settings.embedding_dim, distance=Distance.COSINE),
        )

    def upsert_chunks(self, chunks: List[Chunk], vectors: List[List[float]], batch_size: int = 128) -> None:
        assert len(chunks) == len(vectors)
        for i in range(0, len(chunks), batch_size):
            points = []
            for c, v in zip(chunks[i : i + batch_size], vectors[i : i + batch_size]):
                payload = c.model_dump()
                points.append(PointStruct(id=self._point_id(c), vector=v, payload=payload))
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: List[float], top_k: int = 50) -> List[Tuple[Chunk, float]]:
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        results: List[Tuple[Chunk, float]] = []
        for h in hits:
            results.append((Chunk(**h.payload), float(h.score)))
        return results
