#!/usr/bin/env python3
"""统计 ms-swift completions.jsonl 中显式思考的真实覆盖情况。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

思考模式 = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
严格模式 = re.compile(
    r"^\s*<think>\s*.+?\s*</think>\s*\\boxed\{[^{}]+\}\s*$", re.DOTALL
)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 GRPO rollout 的思考块")
    parser.add_argument("path", type=Path, help="completions.jsonl 文件路径")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="保留完全重复的日志行；默认去除保存检查点时重复写入的末步记录",
    )
    args = parser.parse_args()

    total = opened = closed = unclosed = empty = nonempty = strict = boxed = (
        reason_chars
    ) = 0
    seen_records: set[str] = set()
    duplicate_records = 0
    with args.path.open(encoding="utf-8") as handle:
        for line in handle:
            normalized_line = line.strip()
            if not normalized_line:
                continue
            if not args.keep_duplicates and normalized_line in seen_records:
                duplicate_records += 1
                continue
            seen_records.add(normalized_line)
            record = json.loads(line)
            completions = record.get("completion") or record.get("completions") or []
            if isinstance(completions, str):
                completions = [completions]
            for completion in completions:
                total += 1
                has_opening = "<think>" in completion
                opened += int(has_opening)
                matches = 思考模式.findall(completion)
                if matches:
                    closed += 1
                    reason = next(
                        (item.strip() for item in reversed(matches) if item.strip()), ""
                    )
                    if reason:
                        nonempty += 1
                        reason_chars += len(reason)
                    else:
                        empty += 1
                elif has_opening:
                    unclosed += 1
                strict += int(bool(严格模式.fullmatch(completion)))
                boxed += int(bool(re.search(r"\\boxed\{[^{}]+\}", completion)))

    def ratio(value: int) -> str:
        return f"{value / total:.2%}" if total else "0.00%"

    print(f"生成总数：{total}")
    print(f"忽略的完全重复记录：{duplicate_records}")
    print(f"出现思考开始标签：{opened}（{ratio(opened)}）")
    print(f"闭合思考块：{closed}（{ratio(closed)}）")
    print(f"未闭合思考块：{unclosed}（{ratio(unclosed)}）")
    print(f"空思考块：{empty}（{ratio(empty)}）")
    print(f"非空思考块：{nonempty}（{ratio(nonempty)}）")
    print(f"严格格式：{strict}（{ratio(strict)}）")
    print(f"包含框选答案：{boxed}（{ratio(boxed)}）")
    print(
        f"非空思考平均字符数：{reason_chars / nonempty:.1f}"
        if nonempty
        else "非空思考平均字符数：0.0"
    )


if __name__ == "__main__":
    main()
