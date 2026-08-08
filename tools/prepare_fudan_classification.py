#!/usr/bin/env python3
"""从复旦新闻分类数据构造可复现的 SFT、RLOO 和验证子集。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


标签映射 = {
    "Politics": "政治",
    "Economy": "财经",
    "Sports": "体育",
    "Computer": "计算机",
}
系统提示 = (
    "你是中文新闻分类器。可选类别只有：政治、财经、体育、计算机。"
    "判断新闻的主要主题，只输出一个形如\\boxed{类别}的答案，不要解释。"
)


def 规范正文(text: str, max_chars: int) -> str:
    """清理来源前缀与多余空白，并限制教学样本长度。"""
    text = re.sub(r"^\s*输入:\s*", "", text)
    # 原始生成式分类提示末尾可能附带 20 个候选类别，这里去除与四分类无关的噪声。
    text = re.sub(r"\s*分类:\s*.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def 构造记录(text: str, label: str, source_label: str, record_id: str, 有答案: bool) -> dict:
    """按 ms-swift messages 格式构造一条分类记录。"""
    messages = [
        {"role": "system", "content": 系统提示},
        {
            "role": "user",
            "content": f"请判断下面新闻的主要类别。\n\n新闻：{text}\n\n只输出一个类别答案。",
        },
    ]
    if 有答案:
        messages.append({"role": "assistant", "content": f"\\boxed{{{label}}}"})
    return {
        "messages": messages,
        "label": label,
        "source_label": source_label,
        "record_id": record_id,
    }


def 写入_jsonl(path: Path, rows: list[dict]) -> None:
    """使用 UTF-8 和固定字段顺序写入 JSONL。"""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 文件摘要(path: Path) -> str:
    """计算文件的 SHA-256，便于检查教学数据是否被误改。"""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="准备复旦四分类 SFT 与 RLOO 教学数据")
    parser.add_argument("--source", type=Path, required=True, help="魔塔原始 CSV 文件")
    parser.add_argument("--output", type=Path, required=True, help="教学子集输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-per-class", type=int, default=80)
    parser.add_argument("--rl-per-class", type=int, default=240)
    parser.add_argument("--val-per-class", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=600)
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(f"找不到原始数据：{args.source}")

    text_labels: dict[str, set[str]] = defaultdict(set)
    source_count = 0
    with args.source.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            source_label = row["answer"].strip()
            if source_label in 标签映射:
                source_count += 1
                text = 规范正文(row["prompt"], args.max_chars)
                if text:
                    text_labels[text].add(source_label)

    # 先按模型实际看到的规范化正文全局去重，并丢弃跨标签冲突，防止集合间数据泄漏。
    grouped: dict[str, list[str]] = defaultdict(list)
    conflicting_texts = 0
    for text, source_labels in text_labels.items():
        if len(source_labels) != 1:
            conflicting_texts += 1
            continue
        grouped[next(iter(source_labels))].append(text)

    required = args.sft_per_class + args.rl_per_class + args.val_per_class
    for source_label in 标签映射:
        if len(grouped[source_label]) < required:
            raise RuntimeError(
                f"类别 {source_label} 只有 {len(grouped[source_label])} 条，少于所需 {required} 条"
            )

    randomizer = random.Random(args.seed)
    sft_rows: list[dict] = []
    rl_rows: list[dict] = []
    val_rows: list[dict] = []
    for source_label, label in 标签映射.items():
        texts = grouped[source_label].copy()
        randomizer.shuffle(texts)
        sft_end = args.sft_per_class
        rl_end = sft_end + args.rl_per_class
        val_end = rl_end + args.val_per_class
        partitions = (
            ("sft", texts[:sft_end], sft_rows, True),
            ("rl", texts[sft_end:rl_end], rl_rows, False),
            ("val", texts[rl_end:val_end], val_rows, True),
        )
        for split, selected, target, has_answer in partitions:
            for index, text in enumerate(selected):
                target.append(
                    构造记录(
                        text,
                        label,
                        source_label,
                        f"{source_label}-{split}-{index:04d}",
                        has_answer,
                    )
                )

    # 再次打乱，避免训练过程连续看到同一类别。
    randomizer.shuffle(sft_rows)
    randomizer.shuffle(rl_rows)
    randomizer.shuffle(val_rows)
    smoke_rows = []
    smoke_counts: Counter[str] = Counter()
    for row in rl_rows:
        if smoke_counts[row["label"]] < 4:
            smoke_rows.append(row)
            smoke_counts[row["label"]] += 1

    args.output.mkdir(parents=True, exist_ok=True)
    files = {
        "sft_train.jsonl": sft_rows,
        "rl_train.jsonl": rl_rows,
        "rl_smoke.jsonl": smoke_rows,
        "val.jsonl": val_rows,
    }
    for name, rows in files.items():
        写入_jsonl(args.output / name, rows)

    manifest = {
        "数据来源": "damo/zh_cls_fudan-news",
        "来源地址": "https://modelscope.cn/datasets/damo/zh_cls_fudan-news",
        "来源版本": "1810dce2722d76e714db8290c9a4de3f6c8340f2",
        "许可协议": "Apache-2.0",
        "随机种子": args.seed,
        "正文最大字符数": args.max_chars,
        "原始四类记录数": source_count,
        "规范化后全局唯一正文数": len(text_labels),
        "丢弃的跨标签冲突正文数": conflicting_texts,
        "去除的重复或冲突记录数": source_count - len(text_labels) + conflicting_texts,
        "标签映射": 标签映射,
        "文件": {
            name: {"样本数": len(rows), "SHA-256": 文件摘要(args.output / name)}
            for name, rows in files.items()
        },
    }
    (args.output / "checksums.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
