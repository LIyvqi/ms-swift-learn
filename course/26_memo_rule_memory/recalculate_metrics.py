#!/usr/bin/env python3
"""从已保存的原始响应重新解析并计算指标，避免重复消耗推理时间。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memo_core import 写_jsonl, 汇总审核, 集合指标, 解析审核, 读_jsonl


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重新计算 MeMo 审核实验指标")
    parser.add_argument("directories", nargs="+", type=Path, help="包含逐条 JSONL 的实验目录")
    return parser.parse_args()


def 记忆指标(traces: list[dict[str, Any]]) -> dict[str, float]:
    """从已保存的 Memory 多阶段轨迹重算原始与规范化规则指标。"""

    normalized_scores = []
    raw_scores = []
    valid_values = []
    for trace in traces:
        item = trace.get("memory_trace")
        if not item:
            continue
        memories = item.get("stages") or [item.get("stage1", {})]
        normalized = list(dict.fromkeys(
            rule_id for memory in memories for rule_id in memory.get("rule_ids", [])
        ))
        raw = list(dict.fromkeys(
            rule_id
            for memory in memories
            for rule_id in memory.get("raw_rule_ids", memory.get("rule_ids", []))
        ))
        normalized_scores.append(集合指标(normalized, trace["gold_rule_ids"]))
        raw_scores.append(集合指标(raw, trace["gold_rule_ids"]))
        valid_values.extend(bool(memory.get("valid")) for memory in memories)
    if not normalized_scores:
        return {}
    count = len(normalized_scores)
    return {
        "memory_rule_precision": sum(score[0] for score in normalized_scores) / count,
        "memory_rule_recall": sum(score[1] for score in normalized_scores) / count,
        "memory_rule_f1": sum(score[2] for score in normalized_scores) / count,
        "memory_raw_rule_precision": sum(score[0] for score in raw_scores) / count,
        "memory_raw_rule_recall": sum(score[1] for score in raw_scores) / count,
        "memory_raw_rule_f1": sum(score[2] for score in raw_scores) / count,
        "memory_format_rate": sum(valid_values) / len(valid_values),
    }


def 重算目录(directory: Path) -> list[dict[str, Any]]:
    """重算目录内各方法，并用原子替换之外的普通覆盖保存可复现实验结果。"""

    summaries = []
    for path in sorted(directory.glob("*.jsonl")):
        traces = 读_jsonl(path)
        if not traces or "response" not in traces[0]:
            continue
        method = str(traces[0].get("method") or path.stem)
        for trace in traces:
            trace["prediction"] = 解析审核(trace["response"])
        summary = {"method": method, **汇总审核(traces), **记忆指标(traces)}
        写_jsonl(path, traces)
        (directory / f"{method}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    (directory / "comparison.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summaries


def 主程序() -> None:
    for directory in 解析参数().directories:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        重算目录(directory)


if __name__ == "__main__":
    主程序()
