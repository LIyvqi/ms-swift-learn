"""用模型聊天模板检查多轮 SFT 序列长度，避免静默删除超长样本。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path


def 分位数(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * ratio), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="审计多轮 SFT token 长度")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4608)
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    lengths_by_task: dict[str, list[int]] = defaultdict(list)
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            token_ids = (
                encoded.get("input_ids", [])
                if isinstance(encoded, Mapping)
                else encoded
            )
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            lengths_by_task[row["task"]].append(len(token_ids))

    result = {}
    for task, lengths in sorted(lengths_by_task.items()):
        result[task] = {
            "samples": len(lengths),
            "minimum": min(lengths),
            "p50": 分位数(lengths, 0.5),
            "p95": 分位数(lengths, 0.95),
            "maximum": max(lengths),
            "over_limit": sum(length > args.limit for length in lengths),
        }
    print(
        json.dumps({"limit": args.limit, "tasks": result}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
