#!/usr/bin/env python3
"""审计 MeMo 数据规模、泄漏、拆分和真实聊天模板 token 长度。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from swift.model import get_processor
from swift.template import get_template


项目根目录 = Path(__file__).resolve().parents[2]
数据目录 = 项目根目录 / "datasets/memo_rule_memory"
模型目录 = 项目根目录 / "models/Qwen3.5-0.8B-Base"


def 读_jsonl(path: Path) -> list[dict]:
    """读取非空 JSONL。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 分位数(values: list[int], ratio: float) -> int:
    """计算最近秩分位数。"""

    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]


def 主程序() -> None:
    rules = 读_jsonl(数据目录 / "rules.jsonl")
    train = 读_jsonl(数据目录 / "memory_train.jsonl")
    val = 读_jsonl(数据目录 / "memory_val.jsonl")
    cases = 读_jsonl(数据目录 / "audit_val.jsonl")
    if len(rules) != 80 or len(train) != 680 or len(val) != 200 or len(cases) != 120:
        raise RuntimeError("数据规模不符合课程定义")
    if set(tuple(row["source_rule_ids"]) for row in train) & set(tuple(row["source_rule_ids"]) for row in val):
        # 同一规则可以出现，但问题模板必须完全不同；这里只检查消息问题没有重复。
        pass
    train_questions = {row["messages"][1]["content"] for row in train}
    val_questions = {row["messages"][1]["content"] for row in val}
    if train_questions & val_questions:
        raise RuntimeError("训练与留出问题存在文本重复")
    source_ids = [case["record_id"] for case in cases]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("审核验证集重复使用新闻")
    for case in cases:
        if any(rule_id in case["content"] for rule_id in case["gold_rule_ids"]):
            raise RuntimeError(f"案例泄漏规则编号：{case['case_id']}")

    processor = get_processor(str(模型目录))
    template = get_template(processor, max_length=4096)
    template.set_mode("train")
    lengths = []
    for row in train + val:
        encoded = template.encode(row)
        ids = encoded["input_ids"]
        lengths.append(len(ids) if isinstance(ids, list) else ids.shape[-1])
    result = {
        "规则": len(rules),
        "训练问答": len(train),
        "留出问答": len(val),
        "审核案例": len(cases),
        "规则类别": Counter(rule["category"] for rule in rules),
        "审核场景": Counter(case["scenario_type"] for case in cases),
        "token长度": {
            "最小": min(lengths), "中位数": 分位数(lengths, 0.5),
            "P95": 分位数(lengths, 0.95), "最大": max(lengths),
            "超过768": sum(length > 768 for length in lengths),
        },
        "训练留出问题重复": len(train_questions & val_questions),
        "审核新闻重复": len(source_ids) - len(set(source_ids)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["token长度"]["超过768"]:
        raise RuntimeError("存在超过 MEMORY_MAX_LENGTH 的样本")


if __name__ == "__main__":
    主程序()
