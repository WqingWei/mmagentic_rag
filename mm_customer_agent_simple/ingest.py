#!/usr/bin/env python3
import argparse
from pathlib import Path

from tqdm import tqdm

from src.bm25_store import BM25Store
from src.config import get_settings
from src.embedding import EmbeddingClient
from src.parser import load_manual_chunks
from src.qdrant_store import QdrantStore
from src.schema import Chunk
from src.utils import normalize_text, read_jsonl, stable_id, write_jsonl


def load_chunks_jsonl(path: str | Path, chunk_max_chars: int = 0, overlap_chars: int = 100) -> list[Chunk]:
    rows = read_jsonl(path)
    raw_chunks = [Chunk(**row) for row in rows]
    chunks: list[Chunk] = []
    for chunk in raw_chunks:
        text = normalize_text(chunk.text)
        if not chunk_max_chars or len(text) <= chunk_max_chars:
            chunks.append(chunk.model_copy(update={"text": text}))
            continue

        start = 0
        sub_idx = 0
        while start < len(text):
            end = min(start + chunk_max_chars, len(text))
            part = text[start:end]
            metadata = dict(chunk.metadata or {})
            metadata.update({"parent_chunk_id": chunk.chunk_id, "sub_index": sub_idx})
            cid = stable_id(f"{chunk.chunk_id}:{sub_idx}:{part}", prefix=chunk.manual_id)
            chunks.append(
                chunk.model_copy(
                    update={
                        "chunk_id": cid,
                        "text": part,
                        "metadata": metadata,
                    }
                )
            )
            if end >= len(text):
                break
            start = max(0, end - overlap_chars)
            sub_idx += 1

    if not chunks:
        raise ValueError(f"chunks jsonl is empty: {path}")
    empty_text_ids = [chunk.chunk_id for chunk in chunks if not chunk.text.strip()]
    if empty_text_ids:
        sample = ", ".join(empty_text_ids[:5])
        raise ValueError(f"chunks jsonl has empty text chunks: {sample}")
    too_long_ids = [chunk.chunk_id for chunk in chunks if len(chunk.text) > 8192]
    if too_long_ids:
        sample = ", ".join(too_long_ids[:5])
        raise ValueError(f"chunks jsonl has chunks longer than 8192 chars; use --chunk-max-chars 3000. Sample: {sample}")
    return chunks


def main():
    parser = argparse.ArgumentParser(description="按 # 切块，将说明书写入 Qdrant + BM25。")
    parser.add_argument("--manuals-dir", default=None, help="说明书目录，默认读取 .env 的 MANUALS_DIR")
    parser.add_argument("--chunks-jsonl", default="", help="直接使用已切好的 chunks.jsonl，不再从说明书重新切块")
    parser.add_argument("--chunk-max-chars", type=int, default=0, help="0 表示严格按 # 切块；大于 0 时会对超长章节二次切分")
    args = parser.parse_args()

    settings = get_settings()
    manuals_dir = args.manuals_dir or settings.manuals_dir
    storage_dir = Path(settings.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    if args.chunks_jsonl:
        print(f"读取已切好的 chunks: {args.chunks_jsonl}")
        chunks = load_chunks_jsonl(args.chunks_jsonl, chunk_max_chars=args.chunk_max_chars)
        print(f"读取完成: {len(chunks)} 个 chunk")
    else:
        print(f"读取说明书目录: {manuals_dir}")
        chunks = load_manual_chunks(manuals_dir, chunk_max_chars=args.chunk_max_chars)
        print(f"切块完成: {len(chunks)} 个 chunk")

    chunks_path = storage_dir / "chunks.jsonl"
    write_jsonl(chunks_path, [c.model_dump() for c in chunks])
    print(f"已保存 chunk 元数据: {chunks_path}")

    print("构建 BM25 索引...")
    bm25 = BM25Store.build(chunks)
    bm25_path = storage_dir / "bm25.pkl"
    bm25.save(bm25_path)
    print(f"已保存 BM25: {bm25_path}")

    print("调用 text-embedding-v4 生成向量...")
    embedder = EmbeddingClient(settings)
    texts = [f"{c.manual_id}\n{c.section_path}\n{c.text}\n图片:{','.join(c.image_ids)}" for c in chunks]
    vectors = []
    # text-embedding-v4 单次最多 10 条；EmbeddingClient 内部也按 10 批处理。
    for i in tqdm(range(0, len(texts), 10)):
        vectors.extend(embedder.embed_texts(texts[i : i + 10], batch_size=10))

    print("写入 Qdrant...")
    qdrant = QdrantStore(settings)
    qdrant.recreate_collection()
    qdrant.upsert_chunks(chunks, vectors)
    print(f"完成。collection = {settings.qdrant_collection}")


if __name__ == "__main__":
    main()
