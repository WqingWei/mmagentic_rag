#!/usr/bin/env python3
"""Fine-tune a four-class BERT-compatible intent classifier from JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


LABELS = ("knowledge", "customer_service", "clarification", "mixed")
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练四分类 BERT 意图模型")
    parser.add_argument("--train", required=True, type=Path, help="蒸馏 JSONL")
    parser.add_argument("--output", required=True, type=Path, help="模型输出目录")
    parser.add_argument("--base-model", default="hfl/chinese-macbert-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = str(row.get("label", ""))
            if label not in LABEL2ID:
                raise ValueError(f"第 {line_number} 行标签无效：{label!r}")
            text = str(row.get("student_text") or row.get("query") or "").strip()
            if not text:
                raise ValueError(f"第 {line_number} 行缺少 student_text/query")
            texts.append(text)
            labels.append(LABEL2ID[label])
    if len(texts) < 8:
        raise ValueError("训练集至少需要 8 条样本，并建议四类数据均衡")
    missing = [label for label in LABELS if LABEL2ID[label] not in labels]
    if missing:
        raise ValueError(f"训练集缺少类别：{missing}")
    return texts, labels


def split_indices(labels: Sequence[int], ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0 < ratio < 1:
        raise ValueError("validation-ratio 必须在 0 和 1 之间")
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = {index: [] for index in range(len(LABELS))}
    for index, label in enumerate(labels):
        by_label[label].append(index)
    train_indices: List[int] = []
    validation_indices: List[int] = []
    for indices in by_label.values():
        rng.shuffle(indices)
        validation_size = max(1, round(len(indices) * ratio)) if len(indices) > 1 else 0
        validation_indices.extend(indices[:validation_size])
        train_indices.extend(indices[validation_size:])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def main() -> None:
    args = parse_args()
    import torch
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    texts, labels = load_rows(args.train)
    train_indices, validation_indices = split_indices(
        labels, args.validation_ratio, args.seed
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    class IntentDataset(Dataset):
        def __init__(self, indices: Sequence[int]):
            self.indices = list(indices)

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, item: int):
            index = self.indices[item]
            encoded = tokenizer(
                texts[index],
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(labels[index], dtype=torch.long),
            }

    train_loader = DataLoader(
        IntentDataset(train_indices), batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        IntentDataset(validation_indices), batch_size=args.batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.item())

        model.eval()
        truth: List[int] = []
        predictions: List[int] = []
        with torch.no_grad():
            for batch in validation_loader:
                truth.extend(batch["labels"].tolist())
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(**batch).logits
                predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        macro_f1 = f1_score(truth, predictions, average="macro", zero_division=0)
        average_loss = total_loss / max(len(train_loader), 1)
        print(f"epoch={epoch} loss={average_loss:.4f} validation_macro_f1={macro_f1:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metrics = {
        "labels": list(LABELS),
        "validation_macro_f1": f1_score(
            truth, predictions, average="macro", zero_division=0
        ),
        "classification_report": classification_report(
            truth,
            predictions,
            labels=list(range(len(LABELS))),
            target_names=list(LABELS),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=list(range(len(LABELS)))
        ).tolist(),
    }
    (args.output / "intent_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"模型已保存到 {args.output}")


if __name__ == "__main__":
    main()
