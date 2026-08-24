from pathlib import Path
from typing import Dict, List, Tuple
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .config import Settings
from .customer_qa_data import CustomerQAEntry


class CustomerQAVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = Path(settings.customer_qa_qdrant_path)
        self.client = QdrantClient(path=str(self.path))
        self.query_collection = settings.customer_qa_query_collection
        self.answer_collection = settings.customer_qa_answer_collection

    def recreate_collections(self) -> None:
        for collection in [self.query_collection, self.answer_collection]:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_entries(
        self,
        entries: List[CustomerQAEntry],
        question_vectors: List[List[float]],
        answer_vectors: List[List[float]],
        batch_size: int = 128,
    ) -> None:
        assert len(entries) == len(question_vectors) == len(answer_vectors)
        self._upsert(self.query_collection, entries, question_vectors, batch_size)
        self._upsert(self.answer_collection, entries, answer_vectors, batch_size)

    def search_query(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[Dict, float]]:
        return self._search(self.query_collection, query_vector, top_k)

    def search_answer(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[Dict, float]]:
        return self._search(self.answer_collection, query_vector, top_k)

    def close(self) -> None:
        self.client.close()

    def _upsert(
        self,
        collection: str,
        entries: List[CustomerQAEntry],
        vectors: List[List[float]],
        batch_size: int,
    ) -> None:
        for i in range(0, len(entries), batch_size):
            points = []
            for entry, vector in zip(entries[i : i + batch_size], vectors[i : i + batch_size]):
                payload = {
                    "source_index": entry.index,
                    "question": entry.question,
                    "answer": entry.answer,
                }
                points.append(
                    PointStruct(
                        id=self._point_id(collection, entry),
                        vector=vector,
                        payload=payload,
                    )
                )
            self.client.upsert(collection_name=collection, points=points)

    def _search(
        self, collection: str, query_vector: List[float], top_k: int
    ) -> List[Tuple[Dict, float]]:
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        return [(dict(hit.payload or {}), float(hit.score)) for hit in hits]

    @staticmethod
    def _point_id(collection: str, entry: CustomerQAEntry) -> str:
        return str(uuid5(NAMESPACE_URL, f"{collection}:{entry.index}"))
