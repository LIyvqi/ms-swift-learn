#!/usr/bin/env python3
"""按稳定规则 ID 生成跨规则、Case 和知识库的人工变更影响报告。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

网关模块 = import_module("course.31_hierarchical_memory_agent.memory_gateway")
分层记忆网关 = 网关模块.分层记忆网关


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析人工修改一条规则的回归影响范围")
    parser.add_argument("--rule-id", required=True, help="稳定规则 ID，例如 BT-001")
    parser.add_argument(
        "--registry",
        type=Path,
        default=项目根目录 / "datasets/hierarchical_memory_audit/source_registry.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=项目根目录
        / "outputs/31_hierarchical_memory_agent/change_impact.md",
    )
    parser.add_argument("--maximum-case-ids", type=int, default=30)
    return parser.parse_args()


def 唯一_case(records: list[Any]) -> dict[str, Any]:
    """把多标签投影还原为底层事实 Case。"""

    result = {}
    for record in records:
        result.setdefault(str(record.metadata["record_id"]), record)
    return result


def 主程序() -> None:
    """只输出身份和统计，不把可能敏感的 Case 原文复制进报告。"""

    args = 解析参数()
    gateway = 分层记忆网关(args.registry)
    rules = [
        record
        for record in gateway.connectors["rule_store"].records
        if str(record.metadata["rule_id"]) == args.rule_id
    ]
    if len(rules) != 1:
        raise ValueError(f"active 规则 {args.rule_id} 应唯一，当前数量为 {len(rules)}")
    rule = rules[0]
    category = rule.categories[0]
    route = rule.path[-3]

    case_records = gateway.connectors["case_store"].records
    direct_cases = 唯一_case(
        [record for record in case_records if category in record.categories]
    )
    cooccurrence = Counter()
    for record in direct_cases.values():
        for other in record.categories:
            if other != category:
                cooccurrence[other] += 1

    neighbor_rules = [
        record
        for record in gateway.connectors["rule_store"].records
        if record.memory_id != rule.memory_id and record.path[-3] == route
    ]
    neighbor_categories = {item.categories[0] for item in neighbor_rules}
    confusing_categories = set(cooccurrence) | neighbor_categories
    hard_negative_cases = 唯一_case(
        [
            record
            for record in case_records
            if category not in record.categories
            and bool(set(record.categories) & confusing_categories)
        ]
    )
    safe_cases = 唯一_case(
        [
            record
            for record in case_records
            if bool(record.content.get("is_safe"))
        ]
    )
    knowledge = [
        record
        for record in gateway.connectors["knowledge_store"].records
        if category in record.categories
    ]

    maximum = max(1, args.maximum_case_ids)
    lines = [
        f"# 规则 {args.rule_id} 变更影响报告",
        "",
        "本报告由稳定身份和已审批/训练标签确定性生成，不读取模型预测，也不复制 Case 原文。",
        "",
        "## 规则",
        "",
        f"- 规则记忆：`{rule.memory_id}`",
        f"- 类别：`{category}`",
        f"- 路由：`{route}`",
        f"- 来源：`{rule.metadata['source']}`",
        "",
        "## 直接关联知识",
        "",
    ]
    lines.extend(
        f"- `{record.memory_id}`：{record.title}" for record in knowledge
    )
    lines.extend(
        [
            "",
            "## 回归 Case 规模",
            "",
            f"- 直接正例：{len(direct_cases)}",
            f"- 相邻类别 hard negative：{len(hard_negative_cases)}",
            f"- SAFE 对照池：{len(safe_cases)}",
            "",
            "## 多标签共现类别",
            "",
        ]
    )
    if cooccurrence:
        lines.extend(
            f"- `{name}`：共同出现 {count} 次"
            for name, count in cooccurrence.most_common()
        )
    else:
        lines.append("- 当前没有多标签共现类别。")
    lines.extend(["", "## 同路由相邻规则", ""])
    lines.extend(
        f"- `{record.memory_id}`：{record.title}" for record in neighbor_rules
    )
    lines.extend(["", "## 建议抽查的直接 Case ID", ""])
    lines.extend(f"- `{case_id}`" for case_id in sorted(direct_cases)[:maximum])
    lines.extend(["", "## 建议抽查的 hard negative Case ID", ""])
    lines.extend(
        f"- `{case_id}`" for case_id in sorted(hard_negative_cases)[:maximum]
    )
    lines.extend(
        [
            "",
            "## 合并前检查",
            "",
            "- 对直接正例重新计算类别召回，防止规则收紧后漏放。",
            "- 对相邻类别和 SAFE 池检查过杀，尤其关注新增条件与例外。",
            "- 验证所有引用仍指向 active 规则修订和 approved 知识。",
            "- 人工确认后才重建索引和训练视图；模型建议不能直接合并。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "rule_id": args.rule_id,
                "category": category,
                "direct_cases": len(direct_cases),
                "hard_negative_cases": len(hard_negative_cases),
                "safe_cases": len(safe_cases),
                "knowledge_documents": len(knowledge),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    主程序()
