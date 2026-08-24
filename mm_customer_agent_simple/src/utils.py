import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, List

import jieba


def stable_id(text: str, prefix: str = "chunk") -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def tokenize_zh(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    tokens = []
    for tok in jieba.cut(text):
        tok = tok.strip().lower()
        if tok:
            tokens.append(tok)
    return tokens


def read_jsonl(path: str | Path) -> List[dict]:
    p = Path(path)
    rows = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
