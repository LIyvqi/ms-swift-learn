#!/usr/bin/env python3
"""审计 2000 条分层样本、知识边界、训练协议和实际 token 长度。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from manage_cases import 校验案例
from manage_rules import 校验规则
from taxonomy import 专家, 样本总数, 类别, 划分规模


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
数据目录 = 课程目录 / "data"


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL。"""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 文件摘要(path: Path) -> str:
    """计算文件 SHA256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def 分位数(values: list[int], ratio: float) -> int:
    """计算离散最近秩分位数。"""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio + 0.999999) - 1))
    return ordered[index]


def 主程序() -> None:
    """执行所有可自动证明的数据约束。"""

    canonical = 读取_jsonl(数据目录 / "beavertails_2000.jsonl")
    if len(canonical) != 样本总数:
        raise RuntimeError(f"规范样本应为 {样本总数} 条")
    ids = [row["record_id"] for row in canonical]
    if len(ids) != len(set(ids)):
        raise RuntimeError("规范样本 record_id 不唯一")
    actual_splits = Counter(row["split"] for row in canonical)
    if dict(actual_splits) != dict(划分规模):
        raise RuntimeError(f"划分规模错误：{dict(actual_splits)}")
    for row in canonical:
        flags = [category for category in 类别 if row["category_flags"][category]]
        if flags != row["categories"]:
            raise RuntimeError(f"类别顺序或 flags 不一致：{row['record_id']}")

    train_ids = {row["record_id"] for row in canonical if row["split"] == "train"}
    cases = 读取_jsonl(数据目录 / "knowledge/cases.jsonl")
    base_cases = [row for row in cases if row["source_split"] == "train"]
    reviewed_cases = [row for row in cases if row["source_split"] == "reviewed"]
    if {row["record_id"] for row in base_cases} != train_ids or len(base_cases) != len(train_ids):
        raise RuntimeError("案例库的基础部分必须恰好由 1600 条训练样本构成")
    if len(cases) != len(base_cases) + len(reviewed_cases):
        raise RuntimeError("案例库只允许 train 或人工复核 reviewed 来源")
    case_validation = 校验案例(cases)
    if not case_validation["valid"]:
        raise RuntimeError("案例库校验失败：" + json.dumps(case_validation["errors"], ensure_ascii=False))
    rules = 读取_jsonl(数据目录 / "knowledge/rules.jsonl")
    rule_validation = 校验规则(rules)
    if not rule_validation["valid"]:
        raise RuntimeError("规则库校验失败：" + json.dumps(rule_validation["errors"], ensure_ascii=False))

    contexts = 读取_jsonl(数据目录 / "evaluation_contexts.jsonl")
    evaluation_inputs = 读取_jsonl(数据目录 / "evaluation_inputs.jsonl")
    test_ids = {row["record_id"] for row in canonical if row["split"] == "test"}
    clean_inputs = [row for row in evaluation_inputs if row["evaluation_split"] == "clean"]
    challenge_inputs = [row for row in evaluation_inputs if row["evaluation_split"] == "obfuscated"]
    if len(clean_inputs) != 200 or {row["record_id"] for row in clean_inputs} != test_ids:
        raise RuntimeError("清洁评测集必须与 200 条规范测试集完全一致")
    if len(challenge_inputs) != 100:
        raise RuntimeError("表面扰动泛化挑战集必须为 100 条")
    if any(row["source_record_id"] not in test_ids for row in challenge_inputs):
        raise RuntimeError("泛化挑战集混入非测试来源")
    clean_by_id = {row["record_id"]: row for row in clean_inputs}
    if any(
        (row["prompt"], row["response"])
        == (
            clean_by_id[row["source_record_id"]]["prompt"],
            clean_by_id[row["source_record_id"]]["response"],
        )
        for row in challenge_inputs
    ):
        raise RuntimeError("泛化挑战集存在未实际发生表面扰动的样本")
    evaluation_ids = {row["record_id"] for row in evaluation_inputs}
    if len(evaluation_ids) != len(evaluation_inputs):
        raise RuntimeError("评测输入 record_id 不唯一")
    if len(contexts) != len(evaluation_inputs) * 4:
        raise RuntimeError("每条评测输入必须具有四种冻结检索上下文")
    if any(item["record_id"] not in evaluation_ids for item in contexts):
        raise RuntimeError("冻结评测上下文混入未知评测样本")
    if any(item["record_id"] in {case["record_id"] for case in item["cases"]} for item in contexts):
        raise RuntimeError("评测上下文发生自案例泄漏")

    manifest = json.loads((数据目录 / "manifest.json").read_text(encoding="utf-8"))
    broken_hashes = []
    for relative, metadata in manifest["files"].items():
        path = 课程目录 / relative
        if not path.is_file() or 文件摘要(path) != metadata["sha256"]:
            broken_hashes.append(relative)
    if broken_hashes:
        raise RuntimeError(f"以下数据文件与 manifest 不一致：{broken_hashes}")
    maximum_deviation = float(manifest["maximum_category_rate_deviation_percentage_points"])
    if maximum_deviation > 0.20:
        raise RuntimeError(f"类别边际分布最大偏差 {maximum_deviation:.4f} 个百分点，超过 0.20")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(项目根目录 / "models/Qwen3.5-0.8B-Base")
    view_summary = {}
    for path in sorted((数据目录 / "views").glob("*.jsonl")):
        rows = 读取_jsonl(path)
        lengths = []
        for row in rows:
            encoded = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False
            )
            # Qwen3.5 的多模态 tokenizer 返回 BatchEncoding；普通 tokenizer 才直接返回 ID 列表。
            input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            lengths.append(len(input_ids))
        if not lengths:
            raise RuntimeError(f"空训练视图：{path}")
        view_summary[path.name] = {
            "rows": len(rows),
            "minimum_tokens": min(lengths),
            "mean_tokens": mean(lengths),
            "p50_tokens": 分位数(lengths, 0.50),
            "p95_tokens": 分位数(lengths, 0.95),
            "maximum_tokens": max(lengths),
            "over_1536": sum(length > 1536 for length in lengths),
            "over_1536_rate": sum(length > 1536 for length in lengths) / len(lengths),
        }

    summary = {
        "canonical_rows": len(canonical),
        "splits": dict(actual_splits),
        "case_library_rows": len(cases),
        "reviewed_case_rows": len(reviewed_cases),
        "case_validation": case_validation,
        "rule_validation": rule_validation,
        "evaluation_context_rows": len(contexts),
        "evaluation_input_rows": len(evaluation_inputs),
        "clean_evaluation_rows": len(clean_inputs),
        "obfuscated_evaluation_rows": len(challenge_inputs),
        "maximum_category_rate_deviation_percentage_points": maximum_deviation,
        "views": view_summary,
    }
    (数据目录 / "audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
