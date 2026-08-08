#!/usr/bin/env python3
"""统计 GSM8K 推理结果的答案正确率、格式正确率和平均输出长度。"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else ""


def normalize_number(text: str) -> Decimal | None:
    cleaned = text.replace(",", "").replace("$", "").strip()
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def is_correct(response: str, expected: str) -> bool:
    predicted = extract_boxed(response)
    predicted_number = normalize_number(predicted)
    expected_number = normalize_number(expected)
    if predicted_number is not None and expected_number is not None:
        return abs(predicted_number - expected_number) <= Decimal("1e-8")
    return bool(predicted and predicted == expected.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 ms-swift 生成的 GSM8K JSONL 结果")
    parser.add_argument("result", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.result.open(encoding="utf-8")]
    if not rows:
        raise RuntimeError("推理结果为空")
    expected_answers = [
        row.get("final_answer") or extract_boxed(row.get("labels", "")) for row in rows
    ]
    correct = sum(
        is_correct(row["response"], expected)
        for row, expected in zip(rows, expected_answers)
    )
    formatted = sum(bool(extract_boxed(row["response"])) for row in rows)
    lengths = [len(row["response"]) for row in rows]
    groups = {}
    for row, expected in zip(rows, expected_answers):
        style = "CoT" if "<think>" in row.get("labels", "") else "Direct"
        group = groups.setdefault(style, {"样本数": 0, "答案正确数": 0})
        group["样本数"] += 1
        group["答案正确数"] += int(is_correct(row["response"], expected))
    for group in groups.values():
        group["答案正确率"] = group["答案正确数"] / group["样本数"]
    summary = {
        "文件": str(args.result),
        "样本数": len(rows),
        "答案正确数": correct,
        "答案正确率": correct / len(rows),
        "格式正确率": formatted / len(rows),
        "平均输出字符数": sum(lengths) / len(lengths),
        "最大输出字符数": max(lengths),
        "按风格分组": groups,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
