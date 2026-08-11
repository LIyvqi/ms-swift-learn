#!/usr/bin/env python3
"""检查已有 Agent 动态评测是否与本次协议完全一致。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def 文件摘要(path: Path) -> str:
    """计算输入数据 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="核对已有 Agent-R1 评测协议")
    parser.add_argument("result", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, required=True)
    parser.add_argument("--maximum-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if not args.result.is_file():
        raise SystemExit(1)
    try:
        data = json.loads(args.result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise SystemExit(1) from None
    config = data.get("evaluation_config", {})
    expected = {
        "adapter": args.adapter.resolve(),
        "dataset": args.dataset.resolve(),
        "sample_offset": args.sample_offset,
        "maximum_samples": args.maximum_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "enable_thinking": args.enable_thinking,
        "dataset_sha256": 文件摘要(args.dataset),
    }
    actual_paths = {
        "adapter": Path(str(data.get("adapter", ""))).resolve(),
        "dataset": Path(str(data.get("dataset", ""))).resolve(),
    }
    matches = actual_paths == {key: expected[key] for key in actual_paths} and all(
        config.get(key) == value
        for key, value in expected.items()
        if key not in {"adapter", "dataset"}
    )
    if not matches:
        raise SystemExit(1)
    print(f"已有评测协议匹配：{args.result}")


if __name__ == "__main__":
    主程序()
