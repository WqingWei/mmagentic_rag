import re
from typing import List, Optional

from .config import Settings
from .reranker import QwenReranker
from .retriever import HybridRetriever
from .schema import RetrievedChunk

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


class SearchManualTool:
    """Reusable manual-search tool for the agentic RAG loop."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.retriever = HybridRetriever(settings)
        self.reranker = QwenReranker(settings)

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        top_n = top_k or self.settings.rerank_top_k
        language = self._query_language(query)
        candidates = self.retriever.retrieve(query)
        candidates = self._filter_by_language(candidates, language)
        candidates = candidates[: max(top_n * 8, 30)]
        reranked = self.reranker.rerank(
            query=query,
            chunks=candidates,
            top_n=top_n,
        )
        return reranked[:top_n]

    @staticmethod
    def _query_language(query: str) -> str:
        cjk = len(CJK_RE.findall(query))
        letters = len(ASCII_LETTER_RE.findall(query))
        if cjk > 0:
            return "zh"
        if letters > 0:
            return "en"
        return "unknown"

    @staticmethod
    def _filter_by_language(
        chunks: List[RetrievedChunk], language: str
    ) -> List[RetrievedChunk]:
        if language not in {"zh", "en"}:
            return chunks
        filtered = [
            chunk
            for chunk in chunks
            if SearchManualTool._chunk_language(chunk) == language
        ]
        return filtered

    @staticmethod
    def _chunk_language(chunk: RetrievedChunk) -> str:
        identity = f"{chunk.manual_id} {chunk.source_file}"
        if CJK_RE.search(identity):
            return "zh"
        return "en"
