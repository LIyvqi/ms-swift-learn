#!/usr/bin/env python3
"""统计四分类生成结果的准确率、宏平均召回率和格式正确率。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


标签 = ("政治", "财经", "体育", "计算机")
标签模式 = "|".join(map(re.escape, 标签))
格式模式 = rf"^\s*(?:<think>\s*</think>\s*)?\\boxed\{{(?:{标签模式})\}}\s*$"


def 提取标签(text: str) -> str:
    boxed = re.findall(rf"\\boxed\{{({标签模式})\}}", text)
    if boxed:
        return boxed[-1]
    matches = re.findall(标签模式, text)
    return matches[-1] if matches else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="评测复旦新闻四分类推理结果")
    parser.add_argument("result", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.result.open(encoding="utf-8")]
    if not rows:
        raise RuntimeError("推理结果为空")

    confusion: Counter[tuple[str, str]] = Counter()
    formatted = 0
    lengths = []
    for row in rows:
        expected = row.get("label") or 提取标签(row.get("labels", ""))
        response = row["response"]
        predicted = 提取标签(response)
        confusion[(expected, predicted)] += 1
        formatted += int(bool(re.fullmatch(格式模式, response, re.DOTALL)))
        lengths.append(len(response))

    per_class = {}
    recalls = []
    correct = 0
    for label in 标签:
        total = sum(count for (expected, _), count in confusion.items() if expected == label)
        hit = confusion[(label, label)]
        recall = hit / total if total else 0.0
        recalls.append(recall)
        correct += hit
        per_class[label] = {"样本数": total, "正确数": hit, "召回率": recall}

    summary = {
        "文件": str(args.result),
        "样本数": len(rows),
        "正确数": correct,
        "准确率": correct / len(rows),
        "宏平均召回率": sum(recalls) / len(recalls),
        "格式正确率": formatted / len(rows),
        "平均输出字符数": sum(lengths) / len(lengths),
        "各类别": per_class,
        "混淆矩阵": {
            expected: {predicted or "未识别": confusion[(expected, predicted)] for predicted in (*标签, "")}
            for expected in 标签
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
