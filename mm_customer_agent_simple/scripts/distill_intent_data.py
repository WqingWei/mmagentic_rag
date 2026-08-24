#!/usr/bin/env python3
"""Use the Prompt teacher model to label intent examples for student training.

Input is JSONL. Each row accepts ``query`` (or ``text``), optional ``history`` and
optional ``has_images``. Output is JSONL that can be consumed by
``train_intent_bert.py`` or converted directly to 0.6B instruction-tuning data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from mm_customer_agent_simple.src.agentscope_runtime import (  # noqa: E402
    AgentScopeCustomerServiceRuntime,
)
from mm_customer_agent_simple.src.config import Settings  # noqa: E402
from mm_customer_agent_simple.src.intent_model import (  # noqa: E402
    INTENT_TEACHER_PROMPT,
    PromptIntentClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="蒸馏四分类客服意图数据")
    parser.add_argument("--input", required=True, type=Path, help="未标注 JSONL")
    parser.add_argument("--output", required=True, type=Path, help="蒸馏后 JSONL")
    parser.add_argument(
        "--backend",
        choices=("prompt_api", "local_llm"),
        default="prompt_api",
        help="教师模型后端",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条；0 表示全部")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"第 {line_number} 行必须是 JSON 对象")
            yield value


def student_text(query: str, history: List[Dict[str, Any]], has_images: bool) -> str:
    history_text = "\n".join(
        f"{item.get('role', '')}: {item.get('content', '')}" for item in history[-6:]
    )
    return f"[HISTORY]\n{history_text}\n[QUERY]\n{query}\n[HAS_IMAGE]\n{has_images}"


async def distill(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.backend == "prompt_api" and not settings.intent_api_key:
        raise RuntimeError("prompt_api 教师模型需要设置 INTENT_API_KEY 或 DASHSCOPE_API_KEY")
    settings = replace(
        settings,
        intent_model_backend=args.backend,
        intent_fallback_to_rules=False,
    )
    runtime = AgentScopeCustomerServiceRuntime(settings)
    classifier = PromptIntentClassifier(
        settings,
        runtime,
        local=args.backend == "local_llm",
    )

    rows = list(read_jsonl(args.input))
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            query = str(row.get("query") or row.get("text") or "").strip()
            if not query:
                raise ValueError(f"第 {index} 条缺少 query/text")
            raw_history = row.get("history") or []
            history = raw_history if isinstance(raw_history, list) else []
            has_images = bool(row.get("has_images") or row.get("image_paths"))
            prediction = await classifier.classify(query, history, has_images)
            label_json = prediction.to_dict()
            output = {
                **row,
                "query": query,
                "history": history,
                "has_images": has_images,
                "student_text": student_text(query, history, has_images),
                "label": prediction.intent.value,
                "teacher_prediction": label_json,
                "sft_messages": [
                    {"role": "system", "content": "识别客服请求意图，只输出 JSON。"},
                    {
                        "role": "user",
                        "content": INTENT_TEACHER_PROMPT.format(
                            history=json.dumps(history, ensure_ascii=False),
                            query=query,
                            has_images=str(has_images).lower(),
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(label_json, ensure_ascii=False),
                    },
                ],
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{written}/{len(rows)}] {prediction.intent.value}: {query[:50]}")
    return written


def main() -> None:
    args = parse_args()
    written = asyncio.run(distill(args))
    print(f"完成：{written} 条 -> {args.output}")


if __name__ == "__main__":
    main()
