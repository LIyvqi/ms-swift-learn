#!/usr/bin/env python3
"""校验和原子追加人工复核案例，形成可持续增长的 Case 库。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from retrieval import 读取_jsonl
from taxonomy import 类别, 风险路由


课程目录 = Path(__file__).resolve().parent
默认案例文件 = 课程目录 / "data/knowledge/cases.jsonl"


def 解析参数() -> argparse.Namespace:
    """定义案例库管理命令。"""

    parser = argparse.ArgumentParser(description="管理 Macaron 内容审核案例库")
    parser.add_argument("--cases", type=Path, default=默认案例文件)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--reviewed-only", action="store_true")
    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def 校验案例(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """检查案例 ID、标签、路由和人工复核元数据。"""

    errors = []
    seen = set()
    seen_content = set()
    required = {
        "record_id", "prompt", "response", "is_safe", "categories", "routes",
        "source_split", "source", "reviewed_by", "reviewed_at",
    }
    for index, row in enumerate(rows, start=1):
        missing = required - row.keys()
        if missing:
            errors.append(f"第 {index} 条缺少字段：{sorted(missing)}")
            continue
        if row["record_id"] in seen:
            errors.append(f"重复 record_id：{row['record_id']}")
        seen.add(row["record_id"])
        content_hash = hashlib.sha256((str(row["prompt"]) + "\0" + str(row["response"])).encode()).hexdigest()
        if content_hash in seen_content:
            errors.append(f"{row['record_id']} 与已有案例的 prompt + response 完全重复")
        seen_content.add(content_hash)
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            errors.append(f"{row['record_id']} prompt 为空")
        if not isinstance(row["response"], str) or not row["response"].strip():
            errors.append(f"{row['record_id']} response 为空")
        unknown = set(row["categories"]) - set(类别)
        if unknown:
            errors.append(f"{row['record_id']} 未知类别：{sorted(unknown)}")
        expected_routes = 风险路由(row["categories"])
        if not row["is_safe"] and not expected_routes:
            expected_routes = ["L4"]
        if row["routes"] != expected_routes:
            errors.append(f"{row['record_id']} 路由应为 {expected_routes}，实际 {row['routes']}")
        if row["is_safe"] and row["categories"]:
            errors.append(f"{row['record_id']} SAFE 案例不应含风险类别")
        if row["source_split"] not in {"train", "reviewed"}:
            errors.append(f"{row['record_id']} source_split 非法")
    return {
        "valid": not errors,
        "cases": len(rows),
        "reviewed_cases": sum(row.get("source_split") == "reviewed" for row in rows),
        "errors": errors,
    }


def 原子写(path: Path, rows: list[dict[str, Any]]) -> None:
    """在同目录写临时文件后原子替换。"""

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


def 规范人工案例(payload: dict[str, Any]) -> dict[str, Any]:
    """补齐确定性 ID、路由和来源字段。"""

    required = {"prompt", "response", "is_safe", "categories", "source", "reviewed_by", "reviewed_at"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"人工案例缺少字段：{sorted(missing)}")
    digest = hashlib.sha256((payload["prompt"] + "\0" + payload["response"]).encode()).hexdigest()
    categories = [category for category in 类别 if category in set(payload["categories"])]
    routes = 风险路由(categories)
    if not payload["is_safe"] and not routes:
        routes = ["L4"]
    return {
        "record_id": payload.get("record_id", f"reviewed-{digest[:20]}"),
        "prompt": payload["prompt"].strip(),
        "response": payload["response"].strip(),
        "is_safe": bool(payload["is_safe"]),
        "categories": categories,
        "routes": routes,
        "source_split": "reviewed",
        "source": payload["source"],
        "reviewed_by": payload["reviewed_by"],
        "reviewed_at": payload["reviewed_at"],
    }


def 主程序() -> None:
    """执行案例校验、查看或追加。"""

    args = 解析参数()
    rows = 读取_jsonl(args.cases)
    if args.command == "validate":
        result = 校验案例(rows)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["valid"] else 1)
    if args.command == "list":
        selected = [row for row in rows if not args.reviewed_only or row["source_split"] == "reviewed"]
        print(json.dumps(selected[: args.limit], ensure_ascii=False, indent=2))
        return
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("--input 必须包含一个 JSON 对象")
    new_case = 规范人工案例(payload)
    if any(row["record_id"] == new_case["record_id"] for row in rows):
        raise ValueError(f"案例已经存在：{new_case['record_id']}")
    updated = [*rows, new_case]
    validation = 校验案例(updated)
    if not validation["valid"]:
        raise ValueError("追加后校验失败：" + json.dumps(validation["errors"], ensure_ascii=False))
    原子写(args.cases, updated)
    print(json.dumps({"added": new_case, "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
