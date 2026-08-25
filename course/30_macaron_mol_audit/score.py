#!/usr/bin/env python3
"""汇总单体、多 LoRA、路由、RAG 和持续新增专家的真实生成结果。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from taxonomy import 专家, 类别


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
检索模式 = ("none", "rules", "cases", "full")


def 解析参数() -> argparse.Namespace:
    """定义评分输入输出。"""

    parser = argparse.ArgumentParser(description="评分 Macaron 多 LoRA 内容审核实验")
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=项目根目录 / "outputs/30_macaron_mol_audit/evaluation/generations",
    )
    parser.add_argument("--data-dir", type=Path, default=课程目录 / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=项目根目录 / "outputs/30_macaron_mol_audit/evaluation",
    )
    return parser.parse_args()


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL。"""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 写_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """写出逐样本流水线轨迹。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 二分类指标(gold: list[bool], predicted: list[bool]) -> dict[str, float]:
    """把 UNSAFE 作为正类计算处置指标。"""

    gold_unsafe = [not value for value in gold]
    predicted_unsafe = [not value for value in predicted]
    tp = sum(left and right for left, right in zip(gold_unsafe, predicted_unsafe))
    fp = sum(not left and right for left, right in zip(gold_unsafe, predicted_unsafe))
    fn = sum(left and not right for left, right in zip(gold_unsafe, predicted_unsafe))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": sum(left == right for left, right in zip(gold, predicted)) / max(1, len(gold)),
        "unsafe_precision": precision,
        "unsafe_recall": recall,
        "unsafe_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def 多标签指标(
    gold: list[set[str]], predicted: list[set[str]], selected_categories: Iterable[str] = 类别
) -> dict[str, Any]:
    """计算精确匹配、微平均和宏平均 F1。"""

    selected = tuple(selected_categories)
    selected_set = set(selected)
    gold = [labels & selected_set for labels in gold]
    predicted = [labels & selected_set for labels in predicted]
    tp = sum(len(left & right) for left, right in zip(gold, predicted))
    fp = sum(len(right - left) for left, right in zip(gold, predicted))
    fn = sum(len(left - right) for left, right in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    per_category = {}
    for category in selected:
        category_tp = sum(category in left and category in right for left, right in zip(gold, predicted))
        category_fp = sum(category not in left and category in right for left, right in zip(gold, predicted))
        category_fn = sum(category in left and category not in right for left, right in zip(gold, predicted))
        denominator = 2 * category_tp + category_fp + category_fn
        per_category[category] = 2 * category_tp / denominator if denominator else 0.0
    return {
        "exact_match": sum(left == right for left, right in zip(gold, predicted)) / max(1, len(gold)),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "macro_f1": mean(per_category.values()) if per_category else 0.0,
        "per_category_f1": per_category,
    }


def 加载生成(generation_dir: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    """加载六个 LoRA 的生成并拒绝重复键。"""

    index = {}
    metadata = {}
    for target in ("baseline", "router", "l1", "l2", "l3", "l4"):
        path = generation_dir / f"{target}.jsonl"
        rows = 读取_jsonl(path)
        for row in rows:
            key = (target, row["mode"], row["record_id"])
            if key in index:
                raise RuntimeError(f"重复生成键：{key}")
            index[key] = row
        metadata[target] = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return index, metadata


def 路由指标(rows: list[dict[str, Any]], index: dict[tuple[str, str, str], dict[str, Any]], mode: str) -> dict[str, Any]:
    """独立评价 L0 的安全判断和 route recall。"""

    predicted_safe = []
    primary_hits = []
    route_recall = []
    exact = []
    valid = []
    for row in rows:
        trace = index[("router", mode, row["record_id"])]
        routes = trace["routes"] or []
        predicted_safe.append(trace["decision"] == "SAFE")
        valid.append(bool(trace["valid_format"]))
        if row["routes"]:
            primary_hits.append(bool(routes) and routes[0] == row["routes"][0])
            route_recall.append(len(set(routes) & set(row["routes"])) / len(set(row["routes"])))
            exact.append(routes == row["routes"][:2])
    return {
        **二分类指标([row["is_safe"] for row in rows], predicted_safe),
        "primary_route_accuracy": mean(primary_hits) if primary_hits else 0.0,
        "route_recall_at_2": mean(route_recall) if route_recall else 0.0,
        "route_exact_at_2": mean(exact) if exact else 0.0,
        "format_rate": mean(valid),
    }


def 单体指标(rows: list[dict[str, Any]], index: dict[tuple[str, str, str], dict[str, Any]], mode: str) -> dict[str, Any]:
    """评价一个 LoRA 同时输出 14 类的对照模型。"""

    traces = [index[("baseline", mode, row["record_id"])] for row in rows]
    predicted_labels = [set(trace["labels"] or []) for trace in traces]
    predicted_safe = [trace["decision"] == "SAFE" for trace in traces]
    return {
        **二分类指标([row["is_safe"] for row in rows], predicted_safe),
        **多标签指标([set(row["categories"]) for row in rows], predicted_labels),
        "format_rate": mean(bool(trace["valid_format"]) for trace in traces),
        "mean_lora_calls": 1.0,
    }


def 多专家流水线(
    rows: list[dict[str, Any]],
    index: dict[tuple[str, str, str], dict[str, Any]],
    mode: str,
    policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """按硬路由、Top-2 或全专家组合 L0 与专家结果。"""

    traces = []
    for row in rows:
        router = index[("router", mode, row["record_id"])]
        if policy == "all_experts":
            called = list(专家)
        elif router["decision"] != "UNSAFE" or not router["routes"]:
            called = []
        elif policy == "hard":
            called = router["routes"][:1]
        elif policy == "top2":
            called = router["routes"][:2]
        else:
            raise ValueError(f"未知专家组合策略：{policy}")
        predicted_labels: set[str] = set()
        expert_valid = []
        expert_outputs = {}
        for route in called:
            expert = index[(route.lower(), mode, row["record_id"])]
            labels = set(expert["labels"] or [])
            predicted_labels.update(labels)
            expert_valid.append(bool(expert["valid_format"]))
            expert_outputs[route] = {
                "decision": expert["decision"],
                "labels": sorted(labels),
                "valid_format": expert["valid_format"],
            }
        predicted_safe = (
            not predicted_labels if policy == "all_experts" else router["decision"] == "SAFE"
        )
        traces.append(
            {
                "record_id": row["record_id"],
                "source_record_id": row.get("source_record_id", row["record_id"]),
                "evaluation_split": row.get("evaluation_split", "clean"),
                "mode": mode,
                "policy": policy,
                "gold_is_safe": row["is_safe"],
                "gold_categories": row["categories"],
                "gold_routes": row["routes"],
                "router_decision": router["decision"],
                "router_routes": router["routes"] or [],
                "called_experts": called,
                "expert_outputs": expert_outputs,
                "predicted_is_safe": predicted_safe,
                "predicted_categories": [category for category in 类别 if category in predicted_labels],
                "valid_format": bool(router["valid_format"]) and all(expert_valid),
            }
        )
    metrics = {
        **二分类指标([row["is_safe"] for row in rows], [trace["predicted_is_safe"] for trace in traces]),
        **多标签指标(
            [set(row["categories"]) for row in rows],
            [set(trace["predicted_categories"]) for trace in traces],
        ),
        "format_rate": mean(trace["valid_format"] for trace in traces),
        "mean_lora_calls": mean(1 + len(trace["called_experts"]) for trace in traces),
    }
    return metrics, traces


def 检索指标(rows: list[dict[str, Any]], data_dir: Path) -> dict[str, Any]:
    """独立检查正确规则和同类案例是否真的进入 Top-K。"""

    row_index = {row["record_id"]: row for row in rows}
    contexts = [
        item for item in 读取_jsonl(data_dir / "evaluation_contexts.jsonl")
        if item["record_id"] in row_index
    ]
    result = {}
    for mode in 检索模式:
        mode_rows = [item for item in contexts if item["mode"] == mode]
        rule_recalls, case_recalls = [], []
        for item in mode_rows:
            gold = set(row_index[item["record_id"]]["categories"])
            if not gold:
                continue
            rule_labels = {rule["category"] for rule in item["rules"]}
            case_labels = {category for case in item["cases"] for category in case["categories"]}
            if item["rules"]:
                rule_recalls.append(len(gold & rule_labels) / len(gold))
            if item["cases"]:
                case_recalls.append(len(gold & case_labels) / len(gold))
        result[mode] = {
            "rule_recall_at_3": mean(rule_recalls) if rule_recalls else None,
            "case_label_recall_at_3": mean(case_recalls) if case_recalls else None,
        }
    return result


def 评估方案集(
    rows: list[dict[str, Any]],
    index: dict[tuple[str, str, str], dict[str, Any]],
    data_dir: Path,
) -> dict[str, Any]:
    """在四种检索模式上同时评估单体、路由和三种多专家组合。"""

    baseline = {}
    router = {}
    mol: dict[str, dict[str, Any]] = defaultdict(dict)
    traces = []
    for mode in 检索模式:
        baseline[mode] = 单体指标(rows, index, mode)
        router[mode] = 路由指标(rows, index, mode)
        for policy in ("hard", "top2", "all_experts"):
            metrics, pipeline_traces = 多专家流水线(rows, index, mode, policy)
            mol[mode][policy] = metrics
            traces.extend(pipeline_traces)
    return {
        "samples": len(rows),
        "baseline": baseline,
        "router": router,
        "mol": dict(mol),
        "retrieval": 检索指标(rows, data_dir),
        "traces": traces,
    }


def 持续新增专家指标(
    rows: list[dict[str, Any]], index: dict[tuple[str, str, str], dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    """把 L4 视为部署后新增专家，检查旧专家输出完全不变及新增类别收益。"""

    old_categories = tuple(category for route in ("L1", "L2", "L3") for category in 专家[route]["类别"])
    new_categories = tuple(专家["L4"]["类别"])
    before, after = [], []
    for row in rows:
        old_predictions = set()
        for route in ("L1", "L2", "L3"):
            old_predictions.update(index[(route.lower(), "full", row["record_id"])]["labels"] or [])
        new_predictions = set(index[("l4", "full", row["record_id"])]["labels"] or [])
        before.append(old_predictions)
        after.append(old_predictions | new_predictions)
    gold = [set(row["categories"]) for row in rows]
    old_before = 多标签指标(gold, before, old_categories)
    old_after = 多标签指标(gold, after, old_categories)
    new_before = 多标签指标(gold, before, new_categories)
    new_after = 多标签指标(gold, after, new_categories)
    return {
        "scenario": "部署时已有 L1-L3，随后只增加 L4；不重训也不覆盖旧 LoRA",
        "old_adapter_sha256": {route: metadata[route.lower()]["adapter_sha256"] for route in ("L1", "L2", "L3")},
        "old_prediction_projection_consistency": mean(
            (left & set(old_categories)) == (right & set(old_categories))
            for left, right in zip(before, after)
        ),
        "old_categories_before": old_before,
        "old_categories_after": old_after,
        "new_l4_categories_before": new_before,
        "new_l4_categories_after": new_after,
    }


def 百分比(value: float | None) -> str:
    """渲染百分比。"""

    return "—" if value is None else f"{value * 100:.2f}%"


def 渲染报告(summary: dict[str, Any]) -> str:
    """生成适合直接引用到课程 RESULTS 的简明 Markdown。"""

    lines = [
        "# Macaron 风格多 LoRA 内容审核实测汇总",
        "",
        f"固定测试集：{summary['test_samples']} 条；基础模型：Qwen3.5-0.8B-Base。",
        "",
        "## 端到端结果",
        "",
        "| 检索 | 方案 | 安全判断准确率 | 多标签 Micro-F1 | 多标签 Macro-F1 | 精确匹配 | 格式率 | 平均 LoRA 调用 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in 检索模式:
        baseline = summary["baseline"][mode]
        lines.append(
            f"| {mode} | 单体 14 类 LoRA | {百分比(baseline['accuracy'])} | {百分比(baseline['micro_f1'])} | "
            f"{百分比(baseline['macro_f1'])} | {百分比(baseline['exact_match'])} | {百分比(baseline['format_rate'])} | 1.00 |"
        )
        for policy, name in (("hard", "MoL 硬路由"), ("top2", "MoL Top-2"), ("all_experts", "四专家并行")):
            metrics = summary["mol"][mode][policy]
            lines.append(
                f"| {mode} | {name} | {百分比(metrics['accuracy'])} | {百分比(metrics['micro_f1'])} | "
                f"{百分比(metrics['macro_f1'])} | {百分比(metrics['exact_match'])} | {百分比(metrics['format_rate'])} | "
                f"{metrics['mean_lora_calls']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## 路由与检索",
            "",
            "| 检索 | L0 安全准确率 | 主路由准确率 | Route Recall@2 | 规则 Recall@3 | 案例标签 Recall@3 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in 检索模式:
        router = summary["router"][mode]
        retrieval = summary["retrieval"][mode]
        lines.append(
            f"| {mode} | {百分比(router['accuracy'])} | {百分比(router['primary_route_accuracy'])} | "
            f"{百分比(router['route_recall_at_2'])} | {百分比(retrieval['rule_recall_at_3'])} | "
            f"{百分比(retrieval['case_label_recall_at_3'])} |"
        )
    if summary["generalization"]["samples"]:
        lines.extend(
            [
                "",
                "## 表面扰动泛化挑战",
                "",
                f"挑战集：{summary['generalization']['samples']} 条；“原始子集”与挑战集一一对应，标签不变。",
                "",
                "| 检索 | 方案 | 原始子集 Micro-F1 | 扰动集 Micro-F1 | 变化 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        paired = summary["generalization"]["paired_clean"]
        challenged = summary["generalization"]["obfuscated"]
        for mode in 检索模式:
            for family, policy, name in (
                ("baseline", None, "单体 14 类 LoRA"),
                ("mol", "hard", "MoL 硬路由"),
                ("mol", "top2", "MoL Top-2"),
            ):
                left = paired[family][mode] if policy is None else paired[family][mode][policy]
                right = challenged[family][mode] if policy is None else challenged[family][mode][policy]
                delta = right["micro_f1"] - left["micro_f1"]
                lines.append(
                    f"| {mode} | {name} | {百分比(left['micro_f1'])} | {百分比(right['micro_f1'])} | {delta * 100:+.2f} pp |"
                )
    continual = summary["continual_add_l4"]
    lines.extend(
        [
            "",
            "## 持续新增 L4",
            "",
            f"旧专家输出投影一致率：{百分比(continual['old_prediction_projection_consistency'])}。",
            f"旧领域 Macro-F1：新增前 {百分比(continual['old_categories_before']['macro_f1'])}，"
            f"新增后 {百分比(continual['old_categories_after']['macro_f1'])}。",
            f"L4 新领域 Macro-F1：新增前 {百分比(continual['new_l4_categories_before']['macro_f1'])}，"
            f"新增后 {百分比(continual['new_l4_categories_after']['macro_f1'])}。",
            "",
            "说明：新增专家实验能证明旧 LoRA 权重和旧领域输出未改变，但不能单独证明所有持续学习场景都没有遗忘。",
            "",
        ]
    )
    return "\n".join(lines)


def 主程序() -> None:
    """完成全部组合评分并写出 JSON、JSONL 和 Markdown。"""

    args = 解析参数()
    rows = 读取_jsonl(args.data_dir / "evaluation_inputs.jsonl")
    index, metadata = 加载生成(args.generation_dir)
    available_ids = {
        record_id for target, mode, record_id in index
        if target == "baseline" and mode == "none"
    }
    rows = [row for row in rows if row["record_id"] in available_ids]
    clean_rows = [row for row in rows if row["evaluation_split"] == "clean"]
    challenge_rows = [row for row in rows if row["evaluation_split"] == "obfuscated"]
    clean_index = {row["record_id"]: row for row in clean_rows}
    paired_clean_rows = [clean_index[row["source_record_id"]] for row in challenge_rows]
    expected_keys = {
        (target, mode, row["record_id"])
        for target in ("baseline", "router", "l1", "l2", "l3", "l4")
        for mode in 检索模式
        for row in rows
    }
    missing_keys = expected_keys - index.keys()
    if missing_keys:
        raise RuntimeError(f"六个 LoRA 的评测样本不一致，缺少 {len(missing_keys)} 个键")
    clean = 评估方案集(clean_rows, index, args.data_dir)
    paired_clean = 评估方案集(paired_clean_rows, index, args.data_dir) if paired_clean_rows else None
    obfuscated = 评估方案集(challenge_rows, index, args.data_dir) if challenge_rows else None
    baseline = clean["baseline"]
    router = clean["router"]
    mol = clean["mol"]
    summary = {
        "test_samples": len(clean_rows),
        "generation_metadata": metadata,
        "baseline": baseline,
        "router": router,
        "mol": dict(mol),
        "retrieval": clean["retrieval"],
        "continual_add_l4": 持续新增专家指标(clean_rows, index, metadata),
        "generalization": {
            "samples": len(challenge_rows),
            "construction": "从测试集联合分层抽取，只扰动长英文词的点号与大小写表面形式，语义和标签不变",
            "paired_clean": (
                {key: value for key, value in paired_clean.items() if key != "traces"}
                if paired_clean else None
            ),
            "obfuscated": (
                {key: value for key, value in obfuscated.items() if key != "traces"}
                if obfuscated else None
            ),
        },
        "rag_delta": {
            "baseline_micro_f1_full_minus_none": baseline["full"]["micro_f1"] - baseline["none"]["micro_f1"],
            "hard_micro_f1_full_minus_none": mol["full"]["hard"]["micro_f1"] - mol["none"]["hard"]["micro_f1"],
            "top2_micro_f1_full_minus_none": mol["full"]["top2"]["micro_f1"] - mol["none"]["top2"]["micro_f1"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    写_jsonl(
        args.output_dir / "pipeline_traces.jsonl",
        [*clean["traces"], *(obfuscated["traces"] if obfuscated else [])],
    )
    report = 渲染报告(summary)
    (args.output_dir / "comparison.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    主程序()
