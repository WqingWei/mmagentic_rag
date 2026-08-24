#!/usr/bin/env python3
from tqdm import tqdm

from src.config import get_settings
from src.customer_qa_data import (
    build_customer_qa_manifest,
    load_customer_qa_entries,
    resolve_customer_qa_path,
    write_customer_qa_manifest,
)
from src.customer_qa_vector_store import CustomerQAVectorStore
from src.embedding import EmbeddingClient


def main():
    settings = get_settings()
    qa_path = resolve_customer_qa_path(settings.customer_qa_path)
    entries, raw_count, deduped_count = load_customer_qa_entries(qa_path)
    if not entries:
        raise ValueError(f"customer QA file has no valid entries: {qa_path}")

    print(f"读取客服 FAQ: {qa_path}")
    print(
        f"原始条数: {raw_count}; 去重 question: {deduped_count}; "
        f"索引条数: {len(entries)}"
    )
    print(
        "FAQ Qdrant 独立路径: "
        f"{settings.customer_qa_qdrant_path}; "
        f"collections={settings.customer_qa_query_collection}, "
        f"{settings.customer_qa_answer_collection}"
    )

    embedder = EmbeddingClient(settings)
    question_texts = [entry.question for entry in entries]
    answer_texts = [entry.answer for entry in entries]

    print("生成 Query2Query question embeddings...")
    question_vectors = []
    for i in tqdm(range(0, len(question_texts), 10)):
        question_vectors.extend(embedder.embed_texts(question_texts[i : i + 10], batch_size=10))

    print("生成 Query2Answer answer embeddings...")
    answer_vectors = []
    for i in tqdm(range(0, len(answer_texts), 10)):
        answer_vectors.extend(embedder.embed_texts(answer_texts[i : i + 10], batch_size=10))

    store = CustomerQAVectorStore(settings)
    print("重建客服 FAQ Qdrant collections...")
    store.recreate_collections()
    store.upsert_entries(entries, question_vectors, answer_vectors)
    store.close()

    manifest = build_customer_qa_manifest(
        settings,
        qa_path,
        raw_count=raw_count,
        deduped_count=deduped_count,
        indexed_count=len(entries),
    )
    write_customer_qa_manifest(settings, manifest)
    print("客服 FAQ embedding 索引构建完成。")
    print(f"manifest 已写入: {settings.customer_qa_qdrant_path}/customer_qa_manifest.json")


if __name__ == "__main__":
    main()
