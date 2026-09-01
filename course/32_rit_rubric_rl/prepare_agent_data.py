#!/usr/bin/env python3
"""构建极简规则库、案例库以及 ms-swift 多轮 SFT/GYM 数据。"""

from __future__ import annotations

import argparse
import hashlib
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
环境模块 = import_module("course.32_rit_rubric_rl.agent_environment")
记忆模块 = import_module("course.32_rit_rubric_rl.agent_memory")
极简RiT审核环境 = 环境模块.极简RiT审核环境
极简审核记忆 = 记忆模块.极简审核记忆

默认原始数据 = 项目根目录 / "course/30_macaron_mol_audit/data/beavertails_2000.jsonl"
默认原始规则 = 项目根目录 / "datasets/hierarchical_memory_audit/rules.jsonl"
默认输出目录 = 项目根目录 / "datasets/rit_audit_agent"


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备极简 RiT 安全审核 Agent 数据")
    parser.add_argument("--raw-data", type=Path, default=默认原始数据)
    parser.add_argument("--raw-rules", type=Path, default=默认原始规则)
    parser.add_argument("--output-dir", type=Path, default=默认输出目录)
    parser.add_argument(
        "--human-cases",
        type=Path,
        help="可选的独立人工复核 Case JSONL；approved 记录会并入案例库",
    )
    parser.add_argument("--safe-cases", type=int, default=48)
    parser.add_argument("--cases-per-category", type=int, default=8)
    return parser.parse_args()


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 写入_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """稳定写入紧凑 JSONL，控制课程仓库体积。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 文件摘要(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 构建规则库(raw_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留当前 active 版本及审核真正需要的条件、例外和来源。"""

    fields = (
        "rule_id",
        "version",
        "status",
        "category",
        "name_zh",
        "definition_zh",
        "definition_en",
        "inclusions",
        "exceptions",
        "priority",
        "source",
        "effective_time",
    )
    rules = [
        {field: row[field] for field in fields if field in row}
        for row in raw_rules
        if row.get("status") == "active"
    ]
    if len(rules) != 14 or len({row["category"] for row in rules}) != 14:
        raise ValueError("极简规则库必须恰好覆盖 14 个 active 类别")
    return rules


def 构建案例库(
    train_rows: list[dict[str, Any]],
    categories: list[str],
    safe_limit: int,
    category_limit: int,
) -> list[dict[str, Any]]:
    """从 train 均衡抽取已确认 Case，绝不纳入 validation 或 test。"""

    selected: dict[str, dict[str, Any]] = {}
    safe_count = 0
    per_category: Counter[str] = Counter()
    for row in train_rows:
        if row["is_safe"]:
            if safe_count >= safe_limit:
                continue
            safe_count += 1
            selected[row["record_id"]] = row
            continue
        useful = [
            category
            for category in row["categories"]
            if per_category[category] < category_limit
        ]
        if not useful:
            continue
        selected[row["record_id"]] = row
        for category in row["categories"]:
            if category in categories and per_category[category] < category_limit:
                per_category[category] += 1
        if safe_count >= safe_limit and all(
            per_category[category] >= category_limit for category in categories
        ):
            break
    missing = {
        category: category_limit - per_category[category]
        for category in categories
        if per_category[category] < category_limit
    }
    if safe_count < safe_limit or missing:
        raise ValueError(f"案例配额不足：safe={safe_count}/{safe_limit}，类别={missing}")
    cases = []
    for row in selected.values():
        verdict = "SAFE" if row["is_safe"] else "UNSAFE"
        cases.append(
            {
                "case_id": f"case:{row['record_id']}",
                "record_id": row["record_id"],
                "prompt": row["prompt"],
                "response": row["response"],
                "is_safe": bool(row["is_safe"]),
                "categories": list(row["categories"]),
                "review_note": f"人工确认结论为 {verdict}；类别以数据集复核标注为准。",
                "review_status": "approved",
                "source_split": "train",
                "source": "BeaverTails 固定训练划分",
            }
        )
    return cases


def 构建人工案例库(
    rows: list[dict[str, Any]], categories: set[str], raw_ids: set[str]
) -> list[dict[str, Any]]:
    """规范化独立人工反馈，并防止复用原始 train/validation/test 样本。"""

    required = {
        "case_id",
        "record_id",
        "prompt",
        "response",
        "is_safe",
        "categories",
        "review_note",
        "review_status",
    }
    approved = []
    for row in rows:
        if row.get("review_status") != "approved":
            continue
        missing = required - set(row)
        if missing:
            raise ValueError(f"人工 Case 缺少字段：{sorted(missing)}")
        if str(row["record_id"]) in raw_ids:
            raise ValueError("独立人工 Case 的 record_id 不能复用原始数据样本")
        if not isinstance(row["is_safe"], bool):
            raise ValueError("人工 Case 的 is_safe 必须是 JSON 布尔值")
        if not isinstance(row["categories"], list) or not all(
            isinstance(label, str) for label in row["categories"]
        ):
            raise ValueError("人工 Case 的 categories 必须是字符串列表")
        labels = list(row["categories"])
        if set(labels) - categories:
            raise ValueError(f"人工 Case 存在未知类别：{sorted(set(labels) - categories)}")
        if row["is_safe"] != (len(labels) == 0):
            raise ValueError("人工 Case 的 is_safe 与 categories 不一致")
        approved.append(
            {
                "case_id": str(row["case_id"]),
                "record_id": str(row["record_id"]),
                "prompt": str(row["prompt"]),
                "response": str(row["response"]),
                "is_safe": row["is_safe"],
                "categories": labels,
                "review_note": str(row["review_note"]),
                "review_status": "approved",
                "source_split": "human",
                "source": str(row.get("source", "独立人工复核队列")),
            }
        )
    return approved


def 需要工具(row: dict[str, Any]) -> list[str]:
    """生成可解释的混合策略：明确安全拒答可直出，其余按风险复杂度查库。"""

    response = str(row["response"]).casefold()
    direct_markers = (
        "i can't help",
        "i cannot help",
        "cannot assist",
        "seek professional help",
        "contact emergency",
        "safe way",
    )
    if row["is_safe"] and any(marker in response for marker in direct_markers):
        return []
    if row["is_safe"]:
        return ["search_case"]
    if len(row["categories"]) > 1:
        return ["search_rule", "search_case"]
    return ["search_rule"]


def 环境配置(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """金标签与专家路线只进入环境配置，reset 时不会展示给模型。"""

    try:
        relative = output_dir.resolve().relative_to(项目根目录.resolve())
    except ValueError:
        relative = output_dir.resolve()
    prefix = str(relative)
    return {
        "name": "course_rit_audit_agent",
        "record_id": row["record_id"],
        "prompt": row["prompt"],
        "response": row["response"],
        "is_safe": bool(row["is_safe"]),
        "categories": list(row["categories"]),
        "required_tools": 需要工具(row),
        "rules_path": f"{prefix}/rules.jsonl",
        "cases_path": f"{prefix}/cases.jsonl",
        "max_steps": 3,
    }


def 构建_rl_row(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """GYM 数据只含占位消息和隐藏环境配置。"""

    return {
        "messages": [{"role": "user", "content": "环境会在 rollout 开始时注入审核任务。"}],
        "task": "rit_audit_agent",
        "record_id": row["record_id"],
        "source_split": row["split"],
        "env_config": 环境配置(row, output_dir),
    }


def 构建_sft_row(
    row: dict[str, Any], output_dir: Path, memory: 极简审核记忆
) -> tuple[dict[str, Any], dict[str, Any]]:
    """逐步回放专家动作，确保离线轨迹与在线环境使用同一套代码。"""

    config = 环境配置(row, output_dir)
    environment = 极简RiT审核环境(memory, config)
    observation, _, system = environment.reset()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": observation},
    ]
    info: dict[str, Any] = {}
    while not environment.done:
        action = environment.expert_action()
        messages.append({"role": "assistant", "content": action})
        next_observation, _, done, info = environment.step(action)
        if not done:
            messages.append({"role": "user", "content": next_observation})
    if info.get("event") != "finish" or info.get("metrics", {}).get("gated_reward") != 1.0:
        raise RuntimeError(f"专家轨迹没有满分结束：{row['record_id']}，{info}")
    return (
        {
            "messages": messages,
            "task": "rit_audit_agent",
            "record_id": row["record_id"],
            "source_split": row["split"],
            "expert_route": config["required_tools"] + ["finish"],
        },
        info,
    )


def 平衡冒烟(rows: list[dict[str, Any]], raw_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """固定选择 16 条 SAFE 和 16 条 UNSAFE。"""

    safe = [row for row in rows if raw_by_id[row["record_id"]]["is_safe"]][:16]
    unsafe = [row for row in rows if not raw_by_id[row["record_id"]]["is_safe"]][:16]
    return [item for pair in zip(safe, unsafe) for item in pair]


def 主程序() -> None:
    args = 解析参数()
    raw_rows = 读取_jsonl(args.raw_data)
    counts = Counter(row["split"] for row in raw_rows)
    if counts != {"train": 1600, "validation": 200, "test": 200}:
        raise ValueError(f"原始划分异常：{dict(counts)}")
    if len({row["record_id"] for row in raw_rows}) != len(raw_rows):
        raise ValueError("原始 record_id 不唯一")
    rules = 构建规则库(读取_jsonl(args.raw_rules))
    categories = [row["category"] for row in rules]
    train_rows = [row for row in raw_rows if row["split"] == "train"]
    base_cases = 构建案例库(
        train_rows, categories, args.safe_cases, args.cases_per_category
    )
    human_cases = 构建人工案例库(
        读取_jsonl(args.human_cases) if args.human_cases else [],
        set(categories),
        {str(row["record_id"]) for row in raw_rows},
    )
    cases = base_cases + human_cases
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("案例库 case_id 不唯一")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    写入_jsonl(args.output_dir / "rules.jsonl", rules)
    写入_jsonl(args.output_dir / "cases.jsonl", cases)
    memory = 极简审核记忆(
        args.output_dir / "rules.jsonl", args.output_dir / "cases.jsonl"
    )
    by_split = {
        split: [row for row in raw_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    sft_train_with_info = [
        构建_sft_row(row, args.output_dir, memory) for row in by_split["train"]
    ]
    sft_validation_with_info = [
        构建_sft_row(row, args.output_dir, memory)
        for row in by_split["validation"]
    ]
    sft_train = [row for row, _ in sft_train_with_info]
    sft_validation = [row for row, _ in sft_validation_with_info]
    rl_train = [构建_rl_row(row, args.output_dir) for row in by_split["train"]]
    rl_validation = [构建_rl_row(row, args.output_dir) for row in by_split["validation"]]
    rl_test = [构建_rl_row(row, args.output_dir) for row in by_split["test"]]
    raw_by_id = {row["record_id"]: row for row in raw_rows}
    outputs = {
        "sft_train.jsonl": sft_train,
        "sft_validation.jsonl": sft_validation,
        "sft_smoke.jsonl": 平衡冒烟(sft_train, raw_by_id),
        "rl_train.jsonl": rl_train,
        "rl_validation.jsonl": rl_validation,
        "rl_test.jsonl": rl_test,
        "rl_smoke.jsonl": 平衡冒烟(rl_train, raw_by_id),
    }
    for name, rows in outputs.items():
        写入_jsonl(args.output_dir / name, rows)
    train_ids = {row["record_id"] for row in by_split["train"]}
    validation_ids = {row["record_id"] for row in by_split["validation"]}
    test_ids = {row["record_id"] for row in by_split["test"]}
    base_case_ids = {row["record_id"] for row in base_cases}
    human_case_ids = {row["record_id"] for row in human_cases}
    if (
        base_case_ids - train_ids
        or base_case_ids & validation_ids
        or base_case_ids & test_ids
        or human_case_ids & (train_ids | validation_ids | test_ids)
    ):
        raise ValueError("案例库发生验证或测试泄漏")
    route_counts = Counter(
        " → ".join(row["expert_route"]) for row in sft_train
    )
    checksum_names = ["rules.jsonl", "cases.jsonl", *sorted(outputs)]
    readme = args.output_dir / "README.md"
    if readme.exists():
        checksum_names.append("README.md")
    manifest = {
        "schema_version": 1,
        "course": "极简 RiT 安全审核 Agent",
        "source_split_sizes": dict(counts),
        "rule_count": len(rules),
        "case_count": len(cases),
        "base_case_count": len(base_cases),
        "human_case_count": len(human_cases),
        "case_source_split": "train_and_optional_independent_human",
        "generated_rows": {name: len(rows) for name, rows in outputs.items()},
        "train_route_counts": dict(sorted(route_counts.items())),
        "checksums": {
            name: 文件摘要(args.output_dir / name) for name in checksum_names
        },
        "leakage_guard": "validation/test 不进入规则库、案例库或训练；训练 Case 检索还会排除当前 record_id",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
