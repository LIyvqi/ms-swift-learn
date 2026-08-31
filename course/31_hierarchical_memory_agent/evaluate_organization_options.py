#!/usr/bin/env python3
"""比较当前分层路由与确定性硬关系图路由，不依赖 LLM 生成图谱。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from statistics import mean
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

网关模块 = import_module("course.31_hierarchical_memory_agent.memory_gateway")
分层记忆网关 = 网关模块.分层记忆网关


根路径 = {
    "rule_store": "内容审核政策",
    "case_store": "内容审核案例",
    "knowledge_store": "内容审核知识",
}


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 记录。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 实体身份(row: dict[str, Any]) -> str:
    """多标签 Case 的多个检索投影共享同一个事实身份。"""

    return str(row.get("metadata", {}).get("record_id", row["memory_id"]))


def 支持召回(rows: list[dict[str, Any]], config: dict[str, Any]) -> float:
    """违规样本按类别覆盖计分，安全样本要求命中已标注 SAFE Case。"""

    expected = set(config["categories"])
    if expected:
        visible = {category for row in rows for category in row.get("categories", [])}
        return len(visible & expected) / len(expected)
    return float(
        any(
            row.get("memory_type") == "case"
            and bool(row.get("content", {}).get("is_safe"))
            for row in rows
        )
    )


def 搜索路径集合(
    gateway: Any, locations: list[dict[str, Any]], query: str
) -> tuple[list[dict[str, Any]], int]:
    """搜索候选路径并按底层事实身份去重。"""

    records: dict[str, dict[str, Any]] = {}
    candidates = 0
    for location in locations:
        candidates += int(location["record_count"])
        for row in gateway.搜索(
            location["source_id"], location["path"], query, top_k=6
        ):
            records.setdefault(实体身份(row), row)
    return list(records.values()), candidates


def 节点查找表(gateway: Any) -> dict[tuple[str, str], Any]:
    """建立来源和路径到现有目录节点的稳定映射。"""

    return {
        (source_id, "/".join(node.path)): node
        for source_id, connector in gateway.connectors.items()
        for node in connector.nodes
    }


def 硬图定位(
    gateway: Any,
    query: str,
    node_lookup: dict[tuple[str, str], Any],
    top_k: int = 3,
    seed_k: int = 12,
) -> list[dict[str, Any]]:
    """以已审核类别边连接三个库，再用跨来源支持度重排深层路径。"""

    # 这里的图边只来自规则类别、知识类别和已标注 Case 类别，不抽取测试金标签。
    category_support: Counter[str] = Counter()
    path_direct: Counter[tuple[str, str]] = Counter()
    path_categories: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_seed_entities: set[tuple[str, str]] = set()
    for source_id, root in 根路径.items():
        rows = gateway.搜索(source_id, root, query, top_k=seed_k)
        for rank, row in enumerate(rows, 1):
            entity_key = (source_id, 实体身份(row))
            if entity_key in seen_seed_entities:
                continue
            seen_seed_entities.add(entity_key)
            weight = 1.0 / (60 + rank)
            categories = list(dict.fromkeys(row.get("categories", [])))
            category_weight = weight / max(len(categories), 1)
            for category in categories:
                category_support[str(category)] += category_weight
            key = (source_id, str(row["path"]))
            path_direct[key] += weight
            path_categories[key].update(str(item) for item in categories)

    # 类别节点相当于稳定枢纽：一个来源命中概念后，可把另外两个库的同类路径带入候选。
    for (source_id, path), node in node_lookup.items():
        minimum_depth = int(gateway.connectors[source_id].source.get("minimum_search_depth", 1))
        if len(node.path) < minimum_depth:
            continue
        categories = set(node.categories)
        support = sum(category_support[category] for category in categories)
        if support <= 0:
            continue
        path_categories[(source_id, path)].update(categories)
        path_direct[(source_id, path)] += 1.8 * support

    ranked = []
    for key, score in path_direct.items():
        node = node_lookup.get(key)
        if node is None:
            continue
        source_id, path = key
        categories = path_categories[key]
        cross_source = sum(category_support[category] for category in categories)
        ranked.append(
            {
                "source_id": source_id,
                "path": path,
                "record_count": len(node.record_indices),
                "categories": sorted(categories),
                "score": score + 0.8 * cross_source,
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["score"],
            row["record_count"],
            row["source_id"],
            row["path"],
        )
    )

    # 同一来源和类别的重复文档类型只保留最高项，把名额留给其他关系路径。
    selected = []
    seen_signatures = set()
    for row in ranked:
        signature = (row["source_id"], tuple(row["categories"]))
        if signature in seen_signatures:
            continue
        selected.append(row)
        seen_signatures.add(signature)
        if len(selected) >= top_k:
            break
    return selected


def 受控共识图定位(
    gateway: Any,
    query: str,
    node_lookup: dict[tuple[str, str], Any],
    seed_k: int = 12,
) -> list[dict[str, Any]]:
    """限制每个库只返回一个路径，用跨来源一致性而非图扩散决定目录。"""

    source_category_best: dict[str, Counter[str]] = defaultdict(Counter)
    direct_best: dict[tuple[str, str], float] = defaultdict(float)
    path_categories: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source_id, root in 根路径.items():
        for rank, row in enumerate(
            gateway.搜索(source_id, root, query, top_k=seed_k), 1
        ):
            weight = 1.0 / (60 + rank)
            categories = list(dict.fromkeys(row.get("categories", [])))
            for category in categories:
                source_category_best[source_id][str(category)] = max(
                    source_category_best[source_id][str(category)],
                    weight / max(len(categories), 1),
                )
            key = (source_id, str(row["path"]))
            direct_best[key] = max(direct_best[key], weight)
            path_categories[key].update(str(item) for item in categories)

    # 一个来源内重复命中同类不会无限加分，只有跨来源共同命中才形成更强共识。
    category_consensus = Counter()
    for per_source in source_category_best.values():
        category_consensus.update(per_source)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (source_id, path), node in node_lookup.items():
        minimum_depth = int(gateway.connectors[source_id].source.get("minimum_search_depth", 1))
        if len(node.path) < minimum_depth:
            continue
        categories = set(node.categories)
        direct = direct_best.get((source_id, path), 0.0)
        consensus = sum(category_consensus[category] for category in categories)
        if direct <= 0 and consensus <= 0:
            continue
        by_source[source_id].append(
            {
                "source_id": source_id,
                "path": path,
                "record_count": len(node.record_indices),
                "categories": sorted(categories),
                "score": direct + 1.25 * consensus,
            }
        )
    selected = []
    for source_id in 根路径:
        candidates = by_source[source_id]
        candidates.sort(
            key=lambda row: (-row["score"], row["record_count"], row["path"])
        )
        if candidates:
            selected.append(candidates[0])
    selected.sort(key=lambda row: (-row["score"], row["source_id"]))
    return selected


def 主程序() -> None:
    """在固定测试集上比较两种路由，不修改在线网关。"""

    parser = argparse.ArgumentParser(description="比较分层目录与硬关系图路由")
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
        default=项目根目录
        / "outputs/31_hierarchical_memory_agent/organization_options.json",
    )
    args = parser.parse_args()

    gateway = 分层记忆网关(args.registry)
    rows = 读取_jsonl(args.dataset)
    if args.maximum_samples > 0:
        rows = rows[: args.maximum_samples]
    lookup = 节点查找表(gateway)
    total_records = len(gateway.by_id)
    measurements: dict[str, list[float]] = defaultdict(list)
    examples = []
    for row in rows:
        config = row["env_config"]
        query = f"{config['prompt']}\n{config['response']}"

        started = time.perf_counter()
        hierarchy_locations = gateway.定位(query, top_k=3)
        hierarchy_rows, hierarchy_candidates = 搜索路径集合(
            gateway, hierarchy_locations, query
        )
        measurements["hierarchy_support_recall"].append(
            支持召回(hierarchy_rows, config)
        )
        measurements["hierarchy_candidate_fraction"].append(
            hierarchy_candidates / total_records
        )
        measurements["hierarchy_latency_ms"].append(
            (time.perf_counter() - started) * 1000
        )

        started = time.perf_counter()
        graph_locations = 硬图定位(gateway, query, lookup, top_k=3)
        graph_rows, graph_candidates = 搜索路径集合(gateway, graph_locations, query)
        graph_recall = 支持召回(graph_rows, config)
        measurements["hard_graph_support_recall"].append(graph_recall)
        measurements["hard_graph_candidate_fraction"].append(
            graph_candidates / total_records
        )
        measurements["hard_graph_latency_ms"].append(
            (time.perf_counter() - started) * 1000
        )

        started = time.perf_counter()
        consensus_locations = 受控共识图定位(gateway, query, lookup)
        consensus_rows, consensus_candidates = 搜索路径集合(
            gateway, consensus_locations, query
        )
        measurements["consensus_graph_support_recall"].append(
            支持召回(consensus_rows, config)
        )
        measurements["consensus_graph_candidate_fraction"].append(
            consensus_candidates / total_records
        )
        measurements["consensus_graph_latency_ms"].append(
            (time.perf_counter() - started) * 1000
        )
        if len(examples) < 12 and graph_recall > 支持召回(hierarchy_rows, config):
            examples.append(
                {
                    "record_id": row["record_id"],
                    "gold_categories": config["categories"],
                    "hierarchy_paths": [
                        [item["source_id"], item["path"]] for item in hierarchy_locations
                    ],
                    "hard_graph_paths": [
                        [item["source_id"], item["path"]] for item in graph_locations
                    ],
                }
            )

    summary = {name: mean(values) for name, values in sorted(measurements.items())}
    result = {
        "samples": len(rows),
        "summary": summary,
        "improved_examples": examples,
        "说明": (
            "硬图只使用库内已有类别和目录关系，不读取测试金标签；它是关系索引消融，"
            "不是完整 GraphRAG，也不会改写业务真值。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
