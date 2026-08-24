import re
from pathlib import Path
from typing import List, Tuple

from .schema import Chunk
from .utils import normalize_text, stable_id

IMAGE_PATTERNS = [
    # Markdown image: ![alt](path/to/drill0_04.png)
    re.compile(r"!\[[^\]]*\]\(([^)]+)\)"),
    # HTML image: <img src="xxx/Manual16_51.jpg">
    re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>", re.I),
    # Explicit image id tokens: [PIC:drill0_04] or <PIC:drill0_04>
    re.compile(r"[<\[]PIC[:：]([A-Za-z0-9_\-]+)[>\]]"),
]


def _image_id_from_path(path_or_id: str) -> str:
    value = path_or_id.strip()
    value = value.split("?")[0].split("#")[0]
    name = Path(value).name
    if "." in name:
        name = ".".join(name.split(".")[:-1])
    return name or value


def extract_image_ids(text: str) -> List[str]:
    ids: List[str] = []
    for pat in IMAGE_PATTERNS:
        for m in pat.finditer(text):
            image_id = _image_id_from_path(m.group(1))
            if image_id and image_id not in ids:
                ids.append(image_id)
    return ids


def split_by_hash_heading(markdown: str) -> List[Tuple[str, str, str]]:
    """按照 # 标题切块。返回 (section_title, section_path, section_text)。

    规则：
    - 遇到任意 #/##/### 标题就开启新块。
    - section_path 保留层级路径，便于检索和生成时知道上下文。
    - 没有标题的开头内容放入“未命名章节”。
    """
    text = normalize_text(markdown)
    lines = text.split("\n")

    sections: List[Tuple[str, str, str]] = []
    heading_stack: List[Tuple[int, str]] = []
    current_title = "未命名章节"
    current_path = "未命名章节"
    buf: List[str] = []

    def flush():
        nonlocal buf, current_title, current_path
        body = "\n".join(buf).strip()
        if body:
            sections.append((current_title, current_path, body))
        buf = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack[:] = [(lv, t) for lv, t in heading_stack if lv < level]
            heading_stack.append((level, title))
            current_title = title
            current_path = " > ".join(t for _, t in heading_stack)
            buf.append(line)
        else:
            buf.append(line)
    flush()
    return sections


def build_chunks_from_file(path: Path, chunk_max_chars: int = 0, overlap_chars: int = 100) -> List[Chunk]:
    """从单个 markdown/txt 文件生成 chunks。

    默认 chunk_max_chars=0，表示严格按 # 标题切块，不再二次拆分。
    如果说明书单个标题太长，可以设置 chunk_max_chars=800 做二次切分。
    """
    manual_id = path.stem
    raw = path.read_text(encoding="utf-8", errors="ignore")
    sections = split_by_hash_heading(raw)
    chunks: List[Chunk] = []

    for sec_idx, (title, section_path, body) in enumerate(sections):
        image_ids = extract_image_ids(body)
        body = normalize_text(body)
        if not chunk_max_chars or len(body) <= chunk_max_chars:
            chunk_texts = [body]
        else:
            chunk_texts = []
            start = 0
            while start < len(body):
                end = min(start + chunk_max_chars, len(body))
                chunk_texts.append(body[start:end])
                if end >= len(body):
                    break
                start = max(0, end - overlap_chars)

        for sub_idx, txt in enumerate(chunk_texts):
            cid = stable_id(f"{path.name}:{sec_idx}:{sub_idx}:{txt}", prefix=manual_id)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    manual_id=manual_id,
                    section_title=title,
                    section_path=section_path,
                    text=txt,
                    image_ids=image_ids,
                    source_file=str(path),
                    metadata={"section_index": sec_idx, "sub_index": sub_idx},
                )
            )
    return chunks


def load_manual_chunks(manuals_dir: str, chunk_max_chars: int = 0) -> List[Chunk]:
    root = Path(manuals_dir)
    files = sorted(list(root.glob("**/*.md")) + list(root.glob("**/*.txt")))
    if not files:
        raise FileNotFoundError(f"没有在 {root.resolve()} 找到 .md 或 .txt 说明书文件。")

    all_chunks: List[Chunk] = []
    for path in files:
        all_chunks.extend(build_chunks_from_file(path, chunk_max_chars=chunk_max_chars))
    return all_chunks
