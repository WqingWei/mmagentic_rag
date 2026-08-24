from typing import List

import requests

from .config import Settings
from .schema import RetrievedChunk


class QwenReranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.url = settings.dashscope_rerank_base_url.rstrip("/") + "/reranks"

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        if not chunks:
            return []
        documents = [self._doc_text(c) for c in chunks]
        payload = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
            "instruct": "Given a customer service query, retrieve relevant manual passages that answer the query.",
        }
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # qwen3-rerank 兼容接口通常返回 output.results；少数 SDK/代理可能返回 results。
        results = data.get("output", {}).get("results") or data.get("results") or []
        reranked: List[RetrievedChunk] = []
        for item in results:
            idx = int(item["index"])
            c = chunks[idx]
            c.rerank_score = float(item.get("relevance_score", 0.0))
            reranked.append(c)
        return reranked

    @staticmethod
    def _doc_text(c: RetrievedChunk) -> str:
        return (
            f"说明书: {c.manual_id}\n"
            f"章节: {c.section_path}\n"
            f"图片: {', '.join(c.image_ids)}\n"
            f"内容:\n{c.text}"
        )
