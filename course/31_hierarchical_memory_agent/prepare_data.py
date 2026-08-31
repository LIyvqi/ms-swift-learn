#!/usr/bin/env python3
"""构建三个独立记忆后端、专家轨迹和 ms-swift GYM 训练数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

网关模块 = import_module("course.31_hierarchical_memory_agent.memory_gateway")
环境模块 = import_module("course.31_hierarchical_memory_agent.agent_environment")
分层记忆网关 = 网关模块.分层记忆网关
分层记忆审核环境 = 环境模块.分层记忆审核环境

默认原始数据 = 项目根目录 / "course/30_macaron_mol_audit/data/beavertails_2000.jsonl"
默认原始规则 = 项目根目录 / "course/30_macaron_mol_audit/data/knowledge/rules.jsonl"
默认原始案例 = 项目根目录 / "course/30_macaron_mol_audit/data/knowledge/cases.jsonl"
默认输出目录 = 项目根目录 / "datasets/hierarchical_memory_audit"


def 解析参数() -> argparse.Namespace:
    """允许替换原始库位置，但默认复用已审计的第 30 课数据。"""

    parser = argparse.ArgumentParser(description="准备多源分层记忆审核 Agent 数据")
    parser.add_argument("--raw-data", type=Path, default=默认原始数据)
    parser.add_argument("--raw-rules", type=Path, default=默认原始规则)
    parser.add_argument("--raw-cases", type=Path, default=默认原始案例)
    parser.add_argument("--output-dir", type=Path, default=默认输出目录)
    return parser.parse_args()


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 记录。"""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 写入_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """以稳定字段顺序写入紧凑 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 文件_sha256(path: Path) -> str:
    """计算生成文件摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 构建规则库(rules: list[dict[str, Any]], output_dir: Path) -> Path:
    """生成独立 JSONL 规则后端，不修改原始第 30 课规则。"""

    path = output_dir / "rules.jsonl"
    写入_jsonl(path, rules)
    return path


def 构建案例库(
    cases: list[dict[str, Any]], rules: list[dict[str, Any]], output_dir: Path
) -> Path:
    """把每条事实 Case 投影到全部标签路径，验证和测试样本不会进入。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "cases.sqlite3"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".cases.", suffix=".sqlite3", dir=output_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        connection = sqlite3.connect(temporary_path)
        try:
            connection.execute(
                "CREATE TABLE memory_items ("
                "memory_id TEXT PRIMARY KEY, path_json TEXT NOT NULL, title TEXT NOT NULL, "
                "search_text TEXT NOT NULL, content_json TEXT NOT NULL, "
                "categories_json TEXT NOT NULL, metadata_json TEXT NOT NULL)"
            )
            connection.execute("CREATE INDEX idx_memory_title ON memory_items(title)")
            rows = []
            route_by_category = {
                str(rule["category"]): str(rule["route"])
                for rule in rules
                if rule.get("status") == "active"
            }
            for case in sorted(cases, key=lambda row: row["record_id"]):
                categories = list(case["categories"])
                verdict = "SAFE" if case["is_safe"] else "UNSAFE"
                content = {
                    "prompt": case["prompt"],
                    "response": case["response"],
                    "is_safe": case["is_safe"],
                    "categories": categories,
                }
                metadata = {
                    "record_id": case["record_id"],
                    "source_split": case["source_split"],
                    "source": case["source"],
                    "reviewed_by": case["reviewed_by"],
                    "reviewed_at": case["reviewed_at"],
                }
                projections = categories or ["safe"]
                for projection_category in projections:
                    projection_route = route_by_category.get(projection_category, "L0")
                    hierarchy = [
                        "内容审核案例",
                        "通用平台",
                        projection_route,
                        projection_category,
                        verdict,
                    ]
                    projection_metadata = {
                        **metadata,
                        "projection_category": projection_category,
                    }
                    rows.append(
                        (
                            f"case:{case['record_id']}@{projection_category}",
                            json.dumps(hierarchy, ensure_ascii=False, separators=(",", ":")),
                            f"已复核{verdict}案例 {case['record_id']}",
                            f"{case['prompt']}\n{case['response']}",
                            json.dumps(content, ensure_ascii=False, separators=(",", ":")),
                            json.dumps(categories, ensure_ascii=False, separators=(",", ":")),
                            json.dumps(projection_metadata, ensure_ascii=False, separators=(",", ":")),
                        )
                    )
            connection.executemany(
                "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?)", rows
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary_path, final_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return final_path


def 构建知识库(rules: list[dict[str, Any]], output_dir: Path) -> Path:
    """按真实目录生成定义与边界知识，演示五到六层文档导航。"""

    root = output_dir / "knowledge"
    root.mkdir(parents=True, exist_ok=True)
    expected_files = set()
    for rule in rules:
        category = str(rule["category"])
        route = str(rule["route"])
        documents = (
            (
                "风险概念",
                "定义",
                f"{rule['name_zh']}的概念说明",
                f"{rule['definition_zh']} 英文检索说明：{rule['definition_en']}。",
                rule["inclusions"],
            ),
            (
                "边界判断",
                "例外",
                f"{rule['name_zh']}的包含条件与例外",
                "通常包含："
                + "；".join(rule["inclusions"])
                + "。需要排除："
                + "；".join(rule["exceptions"])
                + "。",
                [*rule["inclusions"], *rule["exceptions"]],
            ),
        )
        for family, document_type, title, body, aliases in documents:
            path_parts = ["内容审核知识", family, "通用平台", route, category, document_type]
            file_path = root.joinpath(*path_parts[:-1], f"{path_parts[-1]}.json")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "memory_id": f"knowledge:{category}:{document_type}",
                "title": title,
                "body": body,
                "aliases": aliases,
                "categories": [category],
                "path": path_parts,
                "metadata": {
                    "source": rule["source"],
                    "rule_id": rule["rule_id"],
                    "version": rule["version"],
                    "status": "approved",
                },
            }
            file_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            expected_files.add(file_path.resolve())
    for existing in root.rglob("*.json"):
        if existing.resolve() not in expected_files:
            existing.unlink()
    return root


def 构建源目录(
    rules: list[dict[str, Any]], output_dir: Path
) -> Path:
    """只登记独立库能力和位置，不把内容复制进中央目录。"""

    categories = [str(rule["category"]) for rule in rules]
    names = [str(rule["name_zh"]) for rule in rules]
    registry = {
        "schema_version": 1,
        "sources": [
            {
                "source_id": "rule_store",
                "name": "版本化审核规则库",
                "description": "保存当前政策、成立条件、例外、优先级和版本。",
                "connector": "jsonl_rules",
                "location": "rules.jsonl",
                "hierarchy_schema": ["政策域", "适用范围", "路由", "类别", "版本"],
                "minimum_search_depth": 5,
                "search_terms": ["policy", "rule", "conditions", "exceptions", *categories, *names],
            },
            {
                "source_id": "case_store",
                "name": "已标注审核案例库",
                "description": "保存训练集和未来人工复核的相似边界案例。",
                "connector": "sqlite_cases",
                "location": "cases.sqlite3",
                "hierarchy_schema": ["案例域", "业务线", "路由", "类别", "结论"],
                "minimum_search_depth": 5,
                "search_terms": ["case", "precedent", "safe", "unsafe", *categories, *names],
            },
            {
                "source_id": "knowledge_store",
                "name": "目录型内容安全知识库",
                "description": "保存风险概念、黑话别名、背景解释和边界知识。",
                "connector": "directory_knowledge",
                "location": "knowledge",
                "hierarchy_schema": ["知识域", "知识族", "业务线", "路由", "类别", "文档类型"],
                "minimum_search_depth": 5,
                "search_terms": ["knowledge", "aliases", "background", "meaning", *categories, *names],
            },
        ],
    }
    path = output_dir / "source_registry.json"
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def 环境配置(row: dict[str, Any], categories: list[str]) -> dict[str, Any]:
    """金标签只进入环境配置，不进入首轮用户观察。"""

    return {
        "name": "course_hierarchical_memory_audit",
        "record_id": row["record_id"],
        "prompt": row["prompt"],
        "response": row["response"],
        "is_safe": row["is_safe"],
        "categories": row["categories"],
        "allowed_categories": categories,
        "registry_path": "datasets/hierarchical_memory_audit/source_registry.json",
        "max_steps": 5,
    }


def 构建_rl_row(row: dict[str, Any], categories: list[str]) -> dict[str, Any]:
    """GRPO 数据只保留占位 prompt 和环境配置。"""

    return {
        "messages": [{"role": "user", "content": "环境将在 rollout 开始时注入审核任务。"}],
        "task": "decision",
        "record_id": row["record_id"],
        "is_safe": row["is_safe"],
        "categories": row["categories"],
        "env_config": 环境配置(row, categories),
    }


def 构建_sft_row(
    row: dict[str, Any], categories: list[str], gateway: 分层记忆网关
) -> tuple[dict[str, Any], dict[str, Any]]:
    """逐步回放确定性专家，确保训练轨迹与在线环境完全一致。"""

    config = 环境配置(row, categories)
    environment = 分层记忆审核环境(gateway, config)
    observation, _, system_prompt = environment.reset()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation},
    ]
    last_info: dict[str, Any] = {}
    while not environment.done:
        action = environment.expert_action()
        messages.append({"role": "assistant", "content": action})
        next_observation, _, done, last_info = environment.step(action)
        if not done:
            messages.append({"role": "user", "content": next_observation})
    if not last_info.get("trace") or last_info["trace"][-1]["event"] != "finish":
        raise RuntimeError(f"专家轨迹没有正常结束：{row['record_id']}")
    return (
        {
            "messages": messages,
            "task": "decision",
            "record_id": row["record_id"],
            "is_safe": row["is_safe"],
            "categories": row["categories"],
        },
        last_info,
    )


def 构建状态转移_sft_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """为检索轨迹选择一个观察后状态，避免重新改变首轮动作先验。"""

    assistant_indices = [
        index
        for index, message in enumerate(row["messages"])
        if message.get("role") == "assistant"
    ]
    if len(assistant_indices) <= 1:
        # 直接 finish 已由完整轨迹阶段监督，再训练会让小模型全部直接结束。
        return None
    # 首轮 locate 已由完整轨迹充分监督；修复集只学习收到环境观察后的推进。
    candidate_indices = assistant_indices[1:]
    digest = hashlib.sha256(str(row["record_id"]).encode("utf-8")).digest()
    target_index = candidate_indices[int.from_bytes(digest[:8], "big") % len(candidate_indices)]
    messages = deepcopy(row["messages"][: target_index + 1])
    action = messages[-1]["content"]
    if '"tool":"search"' in action:
        target_action = "search"
    elif '"tool":"finish"' in action:
        target_action = "finish"
    elif '"tool":"locate"' in action:
        target_action = "locate"
    else:
        raise RuntimeError(f"无法识别状态转移动作：{row['record_id']}")
    return {
        "messages": messages,
        "task": row["task"],
        "record_id": row["record_id"],
        "is_safe": row["is_safe"],
        "categories": row["categories"],
        "target_action": target_action,
    }


def 冒烟子集(rows: list[dict[str, Any]], maximum: int = 24) -> list[dict[str, Any]]:
    """稳定覆盖 SAFE、UNSAFE 和尽可能多的风险类别。"""

    selected = []
    seen_categories = set()
    for row in rows:
        new_categories = set(row.get("categories", [])) - seen_categories
        needs_safe = row.get("is_safe") and not any(item.get("is_safe") for item in selected)
        needs_unsafe = not row.get("is_safe") and not any(
            not item.get("is_safe") for item in selected
        )
        if new_categories or needs_safe or needs_unsafe:
            selected.append(row)
            seen_categories.update(row.get("categories", []))
        if len(selected) >= maximum:
            break
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= maximum:
            break
    return selected


def 主程序() -> None:
    """生成独立后端、训练视图、校验和与泄漏审计。"""

    args = 解析参数()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = 读取_jsonl(args.raw_data)
    rules = 读取_jsonl(args.raw_rules)
    cases = 读取_jsonl(args.raw_cases)
    categories = [str(rule["category"]) for rule in rules if rule.get("status") == "active"]
    if len(categories) != len(set(categories)):
        raise RuntimeError("active 规则类别必须唯一")
    if any(case.get("source_split") not in {"train", "reviewed"} for case in cases):
        raise RuntimeError("Case 库只能包含训练或人工复核来源")

    构建规则库(rules, args.output_dir)
    构建案例库(cases, rules, args.output_dir)
    构建知识库(rules, args.output_dir)
    registry_path = 构建源目录(rules, args.output_dir)
    candidate_path = args.output_dir / "candidate_cases.jsonl"
    if not candidate_path.exists():
        candidate_path.touch()

    gateway = 分层记忆网关(registry_path)
    split_rows = {
        split: [row for row in raw_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    ids = {split: {row["record_id"] for row in rows} for split, rows in split_rows.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise RuntimeError("训练、验证和测试 record_id 发生交叉")
    indexed_case_ids = {
        str(record.metadata["record_id"])
        for record in gateway.connectors["case_store"].records
    }
    if indexed_case_ids != {case["record_id"] for case in cases}:
        raise RuntimeError("SQLite Case 索引与来源案例不一致")
    if indexed_case_ids & (ids["validation"] | ids["test"]):
        raise RuntimeError("验证或测试样本泄漏进 Case 库")

    trajectory_stats = Counter()
    generated: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        sft_rows = []
        for row in split_rows[split]:
            sft_row, info = 构建_sft_row(row, categories, gateway)
            sft_rows.append(sft_row)
            trace = info["trace"]
            trajectory_stats[f"{split}:turns:{len(trace)}"] += 1
            trajectory_stats[f"{split}:direct"] += int(
                bool(trace) and trace[0]["event"] == "finish"
            )
            trajectory_stats[f"{split}:searches"] += sum(
                step["event"] == "search" for step in trace
            )
        generated[f"sft_{split}"] = sft_rows
        state_rows = [构建状态转移_sft_row(row) for row in sft_rows]
        generated[f"sft_state_{split}"] = [row for row in state_rows if row is not None]
        generated[f"rl_{split}"] = [构建_rl_row(row, categories) for row in split_rows[split]]
    generated["rl_test"] = [构建_rl_row(row, categories) for row in split_rows["test"]]

    for name, rows in generated.items():
        写入_jsonl(args.output_dir / f"{name}.jsonl", rows)
    写入_jsonl(
        args.output_dir / "sft_smoke.jsonl",
        冒烟子集(generated["sft_train"]),
    )
    写入_jsonl(
        args.output_dir / "sft_state_smoke.jsonl",
        冒烟子集(generated["sft_state_train"]),
    )
    写入_jsonl(
        args.output_dir / "rl_smoke.jsonl",
        冒烟子集(generated["rl_train"]),
    )

    tracked_files = [
        path
        for path in args.output_dir.rglob("*")
        if path.is_file()
        and path.name not in {"cases.sqlite3", "candidate_cases.jsonl", "manifest.json"}
    ]
    manifest = {
        "schema_version": 1,
        "source_rows": len(raw_rows),
        "splits": {split: len(rows) for split, rows in split_rows.items()},
        "allowed_categories": categories,
        "case_store": {
            "approved_or_training_cases": len(cases),
            "candidate_cases": sum(1 for line in candidate_path.open(encoding="utf-8") if line.strip()),
            "validation_test_indexed": 0,
        },
        "gateway": gateway.摘要(),
        "trajectory_stats": dict(sorted(trajectory_stats.items())),
        "generated_rows": {name: len(rows) for name, rows in generated.items()},
        "checksums": {
            str(path.relative_to(args.output_dir)): 文件_sha256(path)
            for path in sorted(tracked_files)
        },
        "leakage_guard": "Case 后端只含 train/reviewed；validation/test 从不进入可检索库；candidate 默认不可检索",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
