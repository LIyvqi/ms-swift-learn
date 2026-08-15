#!/usr/bin/env python3
"""审计 Direct/CoT 数据经过 ms-swift 模板后的前缀、损失掩码与长度。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median
from typing import Any

思考模式 = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def 读取数据(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并在格式错误时给出明确行号。"""

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path} 第 {line_number} 行缺少非空 messages")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path} 第 {line_number} 行最后一项不是 assistant")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} 没有可审计样本")
    return rows


def 分位数(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * ratio), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 Qwen3.5 双思考模式 SFT 数据")
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        default=[
            Path("datasets/gsm8k_1k/direct_train.jsonl"),
            Path("datasets/gsm8k_1k/cot_train.jsonl"),
            Path("datasets/gsm8k_1k/mixed_train.jsonl"),
        ],
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3.5-0.8B-Base"),
    )
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    from swift.model import get_processor
    from swift.template import get_template

    processor = get_processor(str(args.model))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    template = get_template(
        processor,
        max_length=args.max_length,
        add_non_thinking_prefix=True,
        loss_scale="default+ignore_empty_think",
    )
    template.mode = "train"
    template.init_processor(processor)

    reports = []
    failed = []
    for path in args.datasets:
        lengths = []
        raw_direct = 0
        raw_cot = 0
        input_empty_think = 0
        supervised_empty_think = 0
        supervised_nonempty_think = 0
        for index, row in enumerate(读取数据(path), start=1):
            assistant = str(row["messages"][-1].get("content", ""))
            is_cot = bool(
                (match := 思考模式.search(assistant)) and match.group(1).strip()
            )
            raw_cot += int(is_cot)
            raw_direct += int(not is_cot)

            encoded = template.encode(row, return_length=True)
            input_ids = encoded["input_ids"]
            labels = encoded["labels"]
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            supervised_ids = [
                token for token, label in zip(input_ids, labels) if label != -100
            ]
            supervised_text = tokenizer.decode(
                supervised_ids, skip_special_tokens=False
            )
            lengths.append(len(input_ids))
            input_empty_think += int(bool(re.search(r"<think>\s*</think>", input_text)))
            supervised_empty_think += int(
                bool(re.search(r"<think>\s*</think>", supervised_text))
            )
            supervised_nonempty_think += int(
                any(content.strip() for content in 思考模式.findall(supervised_text))
            )

            if is_cot and not any(
                content.strip() for content in 思考模式.findall(supervised_text)
            ):
                failed.append(f"{path}:{index} 的显式 CoT 没有进入监督损失")
            if not is_cot and re.search(r"<think>\s*</think>", supervised_text):
                failed.append(f"{path}:{index} 的空思考前缀仍在计算损失")

        reports.append(
            {
                "文件": str(path),
                "样本数": len(lengths),
                "原始Direct样本": raw_direct,
                "原始显式CoT样本": raw_cot,
                "模板输入含空思考前缀": input_empty_think,
                "监督目标含空思考前缀": supervised_empty_think,
                "监督目标含非空思考": supervised_nonempty_think,
                "最小token": min(lengths),
                "中位token": median(lengths),
                "P95 token": 分位数(lengths, 0.95),
                "最大token": max(lengths),
                "超过长度上限": sum(length > args.max_length for length in lengths),
            }
        )

    print(
        json.dumps(
            {
                "模型": str(args.model),
                "长度上限": args.max_length,
                "add_non_thinking_prefix": True,
                "loss_scale": "default+ignore_empty_think",
                "数据集": reports,
                "失败项": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failed:
        raise SystemExit("thinking 数据审计失败")


if __name__ == "__main__":
    main()
