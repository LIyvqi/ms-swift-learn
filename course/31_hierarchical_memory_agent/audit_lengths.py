#!/usr/bin/env python3
"""使用真实 Qwen 聊天模板审计多轮专家轨迹长度。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path


def 分位数(values: list[int], ratio: float) -> int:
    """返回无需额外依赖的最近秩分位数。"""

    ordered = sorted(values)
    return ordered[min(int(len(ordered) * ratio), len(ordered) - 1)]


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="审计分层记忆 Agent 的 SFT token 长度")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6144)
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    lengths = []
    turns = []
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
            token_ids = encoded.get("input_ids", []) if isinstance(encoded, Mapping) else encoded
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            lengths.append(len(token_ids))
            turns.append(sum(message["role"] == "assistant" for message in row["messages"]))
    report = {
        "samples": len(lengths),
        "limit": args.limit,
        "minimum": min(lengths),
        "p50": 分位数(lengths, 0.50),
        "p95": 分位数(lengths, 0.95),
        "maximum": max(lengths),
        "over_limit": sum(length > args.limit for length in lengths),
        "mean_agent_turns": sum(turns) / len(turns),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["over_limit"]:
        raise SystemExit("存在超过训练长度上限的专家轨迹")


if __name__ == "__main__":
    主程序()
