from typing import List

from openai import OpenAI

from .config import Settings


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )

    def embed_texts(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
                dimensions=self.settings.embedding_dim,
            )
            vectors.extend([item.embedding for item in resp.data])
        return vectors

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query], batch_size=1)[0]
