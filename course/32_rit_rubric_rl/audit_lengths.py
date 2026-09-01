#!/usr/bin/env python3
"""审计显式思维与短结构化数据的真实聊天模板 token 长度。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import mean


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]


def 分位数(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[min(int((len(ordered) - 1) * ratio), len(ordered) - 1)]


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="审计 RiT 数据 token 长度")
    parser.add_argument(
        "datasets",
        type=Path,
        nargs="*",
        default=[
            项目根目录 / "datasets/rit_audit/sft_train.jsonl",
            项目根目录 / "datasets/rit_audit/rl_train.jsonl",
            项目根目录 / "datasets/rit_audit/structured_sft_train.jsonl",
            项目根目录 / "datasets/rit_audit/structured_rl_train.jsonl",
        ],
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=项目根目录 / "models/Qwen3.5-0.8B-Base",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    from swift.model import get_processor

    processor = get_processor(str(args.model))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    reports = []
    failed = False
    for path in args.datasets:
        lengths = []
        over_limit = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                encoded = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=not path.name.startswith("structured_"),
                )
                tokens = (
                    encoded.get("input_ids", [])
                    if isinstance(encoded, Mapping)
                    else encoded
                )
                if tokens and isinstance(tokens[0], list):
                    tokens = tokens[0]
                length = len(tokens)
                lengths.append(length)
                if length > args.max_length:
                    over_limit.append({"record_id": row["record_id"], "tokens": length})
        report = {
            "dataset": str(path),
            "samples": len(lengths),
            "minimum": min(lengths),
            "p50": 分位数(lengths, 0.50),
            "p95": 分位数(lengths, 0.95),
            "maximum": max(lengths),
            "mean": mean(lengths),
            "over_limit": len(over_limit),
            "over_limit_examples": over_limit[:10],
        }
        failed = failed or bool(over_limit)
        reports.append(report)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit("存在超过训练 max_length 的 RiT 样本")


if __name__ == "__main__":
    主程序()
