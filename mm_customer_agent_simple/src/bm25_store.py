import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .schema import Chunk
from .utils import tokenize_zh


class BM25Store:
    def __init__(self, bm25: BM25Okapi, chunks: List[Chunk]):
        self.bm25 = bm25
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "BM25Store":
        corpus_tokens = []
        for c in chunks:
            # 标题 + 正文 + 图片 ID 都进入 BM25，便于型号、故障码、图片名精确命中。
            text = f"{c.manual_id} {c.section_path} {c.text} {' '.join(c.image_ids)}"
            corpus_tokens.append(tokenize_zh(text))
        return cls(BM25Okapi(corpus_tokens), chunks)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": [c.model_dump() for c in self.chunks]}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Store":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        chunks = [Chunk(**row) for row in data["chunks"]]
        return cls(data["bm25"], chunks)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float]]:
        tokens = tokenize_zh(query)
        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0:
            return []
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in idxs if scores[int(i)] > 0]
