#!/usr/bin/env python3
"""离线评测多源目录定位、子树检索、成本和专家环境闭环。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path
from statistics import mean
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

网关模块 = import_module("course.31_hierarchical_memory_agent.memory_gateway")
环境模块 = import_module("course.31_hierarchical_memory_agent.agent_environment")
分层记忆网关 = 网关模块.分层记忆网关
分层记忆审核环境 = 环境模块.分层记忆审核环境


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取评测样本。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 支持召回(rows: list[dict[str, Any]], config: dict[str, Any]) -> float:
    """违规样本检查类别覆盖，安全样本检查是否找到安全 Case。"""

    categories = set(config["categories"])
    if categories:
        retrieved = {category for row in rows for category in row.get("categories", [])}
        return len(retrieved & categories) / len(categories)
    return float(
        any(
            row.get("memory_type") == "case"
            and bool(row.get("content", {}).get("is_safe"))
            for row in rows
        )
    )


def 目录支持召回(rows: list[dict[str, Any]], config: dict[str, Any]) -> float:
    """检查定位阶段是否暴露了可支持当前结论的目录。"""

    categories = set(config["categories"])
    if categories:
        visible = {category for row in rows for category in row.get("categories", [])}
        return len(visible & categories) / len(categories)
    return float(any(row["source_id"] == "case_store" for row in rows))


def 搜索目录集合(
    gateway: Any,
    locations: list[dict[str, Any]],
    query: str,
    maximum_paths: int,
) -> tuple[list[dict[str, Any]], int]:
    """依次搜索前若干目录并按 memory_id 去重。"""

    records: dict[str, dict[str, Any]] = {}
    candidate_records = 0
    for location in locations[:maximum_paths]:
        candidate_records += int(location["record_count"])
        for row in gateway.搜索(
            location["source_id"], location["path"], query, top_k=6
        ):
            records.setdefault(row["memory_id"], row)
    return list(records.values()), candidate_records


def 回放专家(gateway: Any, row: dict[str, Any]) -> dict[str, Any]:
    """确认离线专家使用的正是在线环境，而不是旁路生成文本。"""

    environment = 分层记忆审核环境(gateway, row["env_config"])
    environment.reset()
    total_reward = 0.0
    info: dict[str, Any] = {}
    while not environment.done:
        action = environment.expert_action()
        _, reward, _, info = environment.step(action)
        total_reward += reward
    return {
        "finished": bool(info.get("trace")) and info["trace"][-1]["event"] == "finish",
        "safety_accuracy": info.get("metrics", {}).get("safety_accuracy", 0.0),
        "category_f1": info.get("metrics", {}).get("category_f1", 0.0),
        "memory_recall": info.get("metrics", {}).get("memory_category_recall", 0.0),
        "turns": len(info.get("trace", [])),
        "total_reward": total_reward,
    }


def 主程序() -> None:
    """比较全库搜索、Top-1 分层搜索和 Top-3 联邦搜索。"""

    parser = argparse.ArgumentParser(description="评测分层联邦记忆流水线")
    parser.add_argument(
        "--registry",
        type=Path,
        default=项目根目录 / "datasets/hierarchical_memory_audit/source_registry.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=项目根目录 / "datasets/hierarchical_memory_audit/rl_test.jsonl",
    )
    parser.add_argument("--maximum-samples", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=项目根目录 / "outputs/31_hierarchical_memory_agent/offline_metrics.json",
    )
    args = parser.parse_args()

    gateway = 分层记忆网关(args.registry)
    rows = 读取_jsonl(args.dataset)
    if args.maximum_samples > 0:
        rows = rows[: args.maximum_samples]
    metrics: dict[str, list[float]] = {
        "locate_recall_at_6": [],
        "top1_support_recall": [],
        "top3_support_recall": [],
        "flat_rule_recall": [],
        "flat_case_recall": [],
        "flat_knowledge_recall": [],
        "top1_candidate_fraction": [],
        "top3_candidate_fraction": [],
        "locate_latency_ms": [],
        "top3_search_latency_ms": [],
    }
    total_records = len(gateway.by_id)
    for row in rows:
        config = row["env_config"]
        query = f"{config['prompt']}\n{config['response']}"
        started = time.perf_counter()
        locations = gateway.定位(query, top_k=6)
        metrics["locate_latency_ms"].append((time.perf_counter() - started) * 1000)
        metrics["locate_recall_at_6"].append(目录支持召回(locations, config))

        started = time.perf_counter()
        top1, top1_candidates = 搜索目录集合(gateway, locations, query, 1)
        top3, top3_candidates = 搜索目录集合(gateway, locations, query, 3)
        metrics["top3_search_latency_ms"].append((time.perf_counter() - started) * 1000)
        metrics["top1_support_recall"].append(支持召回(top1, config))
        metrics["top3_support_recall"].append(支持召回(top3, config))
        metrics["top1_candidate_fraction"].append(top1_candidates / total_records)
        metrics["top3_candidate_fraction"].append(top3_candidates / total_records)

        root_paths = {
            "rule_store": "内容审核政策",
            "case_store": "内容审核案例",
            "knowledge_store": "内容审核知识",
        }
        for source_id, metric_name in (
            ("rule_store", "flat_rule_recall"),
            ("case_store", "flat_case_recall"),
            ("knowledge_store", "flat_knowledge_recall"),
        ):
            flat = gateway.搜索(source_id, root_paths[source_id], query, top_k=6)
            metrics[metric_name].append(支持召回(flat, config))

    expert_rows = [回放专家(gateway, row) for row in rows]
    summary = {
        "samples": len(rows),
        "gateway": gateway.摘要(),
        "retrieval": {name: mean(values) if values else 0.0 for name, values in metrics.items()},
        "expert_environment": {
            "completion_rate": mean(row["finished"] for row in expert_rows),
            "safety_accuracy": mean(row["safety_accuracy"] for row in expert_rows),
            "category_f1": mean(row["category_f1"] for row in expert_rows),
            "memory_recall": mean(row["memory_recall"] for row in expert_rows),
            "mean_turns": mean(row["turns"] for row in expert_rows),
            "mean_total_reward": mean(row["total_reward"] for row in expert_rows),
        },
        "说明": (
            "检索指标是确定性基础设施评测；专家环境为金标签生成轨迹的协议上界，"
            "两者都不是训练后模型效果。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
