#!/usr/bin/env python3
"""校验、查询和原子升级版本化审核规则。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from retrieval import 读取_jsonl
from taxonomy import 专家, 类别到专家


课程目录 = Path(__file__).resolve().parent
默认规则文件 = 课程目录 / "data/knowledge/rules.jsonl"
默认审计日志 = 课程目录 / "data/knowledge/rule_changes.jsonl"
必需字段 = {
    "rule_id",
    "version",
    "status",
    "route",
    "category",
    "name_zh",
    "definition_zh",
    "definition_en",
    "inclusions",
    "exceptions",
    "severity",
    "priority",
    "source",
    "effective_time",
}


def 解析参数() -> argparse.Namespace:
    """定义三个规则管理子命令。"""

    parser = argparse.ArgumentParser(description="管理 Macaron 内容审核规则库")
    parser.add_argument("--rules", type=Path, default=默认规则文件)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="检查字段、版本和分类映射")
    list_parser = subparsers.add_parser("list", help="列出规则")
    list_parser.add_argument("--route", choices=list(专家))
    list_parser.add_argument("--status", choices=("active", "deprecated"))
    upsert_parser = subparsers.add_parser("upsert", help="从一个 JSON 对象升级现有规则")
    upsert_parser.add_argument("--input", type=Path, required=True)
    upsert_parser.add_argument("--reason", required=True)
    upsert_parser.add_argument("--audit-log", type=Path, default=默认审计日志)
    return parser.parse_args()


def 校验规则(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """执行能阻止错误路由和版本覆盖的结构校验。"""

    errors: list[str] = []
    identities = Counter()
    active_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        missing = 必需字段 - row.keys()
        if missing:
            errors.append(f"第 {index} 条缺少字段：{sorted(missing)}")
            continue
        identity = (str(row["rule_id"]), int(row["version"]))
        identities[identity] += 1
        if row["status"] not in {"active", "deprecated"}:
            errors.append(f"{identity} 状态非法：{row['status']}")
        if not isinstance(row["version"], int) or row["version"] < 1:
            errors.append(f"{identity} version 必须是正整数")
        if row["route"] not in 专家:
            errors.append(f"{identity} 未知专家：{row['route']}")
        expected_route = 类别到专家.get(row["category"])
        if expected_route is None:
            errors.append(f"{identity} 类别尚未登记进 taxonomy.py：{row['category']}")
        elif row["route"] != expected_route:
            errors.append(f"{identity} 路由应为 {expected_route}，实际 {row['route']}")
        if not isinstance(row["inclusions"], list) or not row["inclusions"]:
            errors.append(f"{identity} inclusions 必须是非空列表")
        if not isinstance(row["exceptions"], list) or not row["exceptions"]:
            errors.append(f"{identity} exceptions 必须是非空列表")
        if not 1 <= int(row["severity"]) <= 5:
            errors.append(f"{identity} severity 必须在 1 到 5")
        if row["status"] == "active":
            active_by_rule[row["rule_id"]].append(row)
            active_by_category[row["category"]].append(row)
    for identity, count in identities.items():
        if count > 1:
            errors.append(f"重复规则版本：{identity}")
    for rule_id, active in active_by_rule.items():
        if len(active) > 1:
            errors.append(f"{rule_id} 同时存在 {len(active)} 个 active 版本")
    for category, active in active_by_category.items():
        if len(active) > 1:
            errors.append(f"类别 {category} 同时被 {len(active)} 条 active 规则管理")
    missing_categories = sorted(set(类别到专家) - set(active_by_category))
    if missing_categories:
        errors.append(f"以下类别没有 active 规则：{missing_categories}")
    return {
        "valid": not errors,
        "rules": len(rows),
        "active_rules": sum(len(items) for items in active_by_rule.values()),
        "deprecated_rules": sum(row.get("status") == "deprecated" for row in rows),
        "errors": errors,
    }


def 原子写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """先写同目录临时文件并 fsync，再原子替换正式规则库。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def 升级规则(
    rows: list[dict[str, Any]], patch: dict[str, Any], reason: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """弃用当前 active 版本并创建只增加版本号的新记录。"""

    rule_id = str(patch.get("rule_id", ""))
    active = [row for row in rows if row.get("rule_id") == rule_id and row.get("status") == "active"]
    if len(active) != 1:
        raise ValueError(f"rule_id={rule_id!r} 必须恰好有一个 active 版本")
    old = active[0]
    immutable = {"rule_id", "category", "route"}
    for field in immutable:
        if field in patch and patch[field] != old[field]:
            raise ValueError(f"不能通过 upsert 修改 {field}；新增分类需先更新 taxonomy.py")
    forbidden = {"version", "status"} & patch.keys()
    if forbidden:
        raise ValueError(f"版本和状态由工具管理，输入不得包含：{sorted(forbidden)}")
    new = {**old, **patch}
    new["version"] = max(int(row["version"]) for row in rows if row["rule_id"] == rule_id) + 1
    new["status"] = "active"
    new["effective_time"] = dt.datetime.now(dt.timezone.utc).date().isoformat()
    updated = []
    for row in rows:
        if row is old:
            updated.append({**row, "status": "deprecated"})
        else:
            updated.append(row)
    updated.append(new)
    event = {
        "changed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rule_id": rule_id,
        "old_version": old["version"],
        "new_version": new["version"],
        "reason": reason,
        "changed_fields": sorted(patch.keys()),
    }
    return updated, event


def 主程序() -> None:
    """执行指定的规则管理操作。"""

    args = 解析参数()
    rows = 读取_jsonl(args.rules)
    if args.command == "validate":
        result = 校验规则(rows)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["valid"] else 1)
    if args.command == "list":
        selected = [
            row for row in rows
            if (args.route is None or row["route"] == args.route)
            and (args.status is None or row["status"] == args.status)
        ]
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return
    patch = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(patch, dict):
        raise TypeError("--input 必须包含一个 JSON 对象")
    updated, event = 升级规则(rows, patch, args.reason)
    validation = 校验规则(updated)
    if not validation["valid"]:
        raise ValueError("升级后的规则库校验失败：" + json.dumps(validation["errors"], ensure_ascii=False))
    原子写_jsonl(args.rules, updated)
    args.audit_log.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"updated": event, "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
