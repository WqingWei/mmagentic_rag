import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import Settings


@dataclass(frozen=True)
class CustomerQAEntry:
    index: int
    question: str
    answer: str


def resolve_customer_qa_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path
    if path.is_absolute():
        raise FileNotFoundError(f"customer QA file not found: {path}")

    package_root = Path(__file__).resolve().parents[1]
    workspace_root = package_root.parent
    candidates = [
        Path.cwd() / path,
        workspace_root / path,
        package_root / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "customer QA file not found. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_customer_qa_entries(path: Path) -> tuple[List[CustomerQAEntry], int, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"customer QA file must be a JSON array: {path}")

    entries: List[CustomerQAEntry] = []
    seen_questions = set()
    deduped_count = 0
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"customer QA row {idx} must be an object")
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"customer QA row {idx} must include question and answer")
        if question in seen_questions:
            deduped_count += 1
            continue
        seen_questions.add(question)
        entries.append(CustomerQAEntry(index=idx, question=question, answer=answer))
    return entries, len(data), deduped_count


def customer_qa_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def customer_qa_manifest_path(settings: Settings) -> Path:
    return Path(settings.customer_qa_qdrant_path) / "customer_qa_manifest.json"


def build_customer_qa_manifest(
    settings: Settings,
    qa_path: Path,
    raw_count: int,
    deduped_count: int,
    indexed_count: int,
) -> dict:
    return {
        "qa_path": str(qa_path.resolve()),
        "qa_sha256": customer_qa_file_hash(qa_path),
        "raw_count": raw_count,
        "deduped_count": deduped_count,
        "indexed_count": indexed_count,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "qdrant_path": settings.customer_qa_qdrant_path,
        "query_collection": settings.customer_qa_query_collection,
        "answer_collection": settings.customer_qa_answer_collection,
    }


def read_customer_qa_manifest(settings: Settings) -> dict | None:
    path = customer_qa_manifest_path(settings)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def write_customer_qa_manifest(settings: Settings, manifest: dict) -> None:
    path = customer_qa_manifest_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def manifest_matches(expected: dict, actual: dict | None) -> bool:
    if not actual:
        return False
    keys = [
        "qa_sha256",
        "indexed_count",
        "embedding_model",
        "embedding_dim",
        "query_collection",
        "answer_collection",
    ]
    return all(actual.get(key) == expected.get(key) for key in keys)
