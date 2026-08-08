#!/usr/bin/env python3
"""统计 CoT 新闻分类的标签、结构、一致性和人工证据覆盖指标。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# 复用训练奖励的同一套解析逻辑，防止训练和离线评测口径不一致。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from course.plugins.cot_classification_common import 允许标签, 解析回答, 证据分隔符


def 提取期望标签(row: dict) -> str:
    """优先读取顶层标签，兼容 ms-swift 把标准答案写入 labels 的结果。"""
    if row.get("label"):
        return row["label"]
    return 解析回答(row.get("labels", "")).标签


def main() -> None:
    parser = argparse.ArgumentParser(description="评测复旦新闻 CoT 四分类推理结果")
    parser.add_argument("result", type=Path)
    parser.add_argument("--reference", type=Path, help="可选的原始验证 JSONL，用于恢复被推理器裁掉的证据字段")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.result.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise RuntimeError("推理结果为空")
    if args.reference:
        references = [
            json.loads(line) for line in args.reference.open(encoding="utf-8") if line.strip()
        ]
        if len(references) != len(rows):
            raise ValueError("推理结果与参考数据的样本数不一致")
        for row, reference in zip(rows, references):
            row.setdefault("label", reference.get("label", ""))
            row.setdefault("evidence_terms", reference.get("evidence_terms", ""))

    confusion: Counter[tuple[str, str]] = Counter()
    strict_count = 0
    nonempty_count = 0
    consistent_count = 0
    reason_lengths = []
    evidence_scores = []
    all_evidence_count = 0

    for row in rows:
        expected = 提取期望标签(row)
        parsed = 解析回答(row["response"])
        confusion[(expected, parsed.标签)] += 1
        strict_count += int(parsed.严格格式 and 15 <= len(parsed.推理) <= 220)
        nonempty_count += int(bool(parsed.推理))
        consistent_count += int(bool(parsed.标签) and parsed.标签 in parsed.推理)
        reason_lengths.append(len(parsed.推理))

        packed_terms = row.get("evidence_terms", "")
        if packed_terms:
            terms = [term for term in packed_terms.split(证据分隔符) if term]
            coverage = sum(term in parsed.推理 for term in terms) / len(terms)
            evidence_scores.append(coverage)
            all_evidence_count += int(coverage == 1.0)

    per_class = {}
    recalls = []
    correct = 0
    for label in 允许标签:
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
        "标签准确率": correct / len(rows),
        "宏平均召回率": sum(recalls) / len(recalls),
        "严格 CoT 格式率": strict_count / len(rows),
        "非空推理率": nonempty_count / len(rows),
        "推理结论一致率": consistent_count / len(rows),
        "平均推理字符数": sum(reason_lengths) / len(reason_lengths),
        "证据评测样本数": len(evidence_scores),
        "平均证据覆盖率": sum(evidence_scores) / len(evidence_scores) if evidence_scores else None,
        "三个证据全部覆盖率": all_evidence_count / len(evidence_scores) if evidence_scores else None,
        "各类别": per_class,
        "混淆矩阵": {
            expected: {
                predicted or "未识别": confusion[(expected, predicted)]
                for predicted in (*允许标签, "")
            }
            for expected in 允许标签
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
