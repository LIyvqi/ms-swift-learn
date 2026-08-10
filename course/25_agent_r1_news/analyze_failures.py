"""按检索、组合、协议、反思、决策和证据阶段归因动态评测失败。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def 主函数() -> None:
    parser = argparse.ArgumentParser(description="分析 Agent-R1 动态评测失败轨迹")
    parser.add_argument("evaluation", type=Path, help="evaluate_agent.py 生成的 JSON")
    parser.add_argument("--output", type=Path, help="可选的归因 JSON 输出路径")
    parser.add_argument("--maximum-examples", type=int, default=5)
    args = parser.parse_args()

    data = json.loads(args.evaluation.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    if not isinstance(traces, list):
        raise SystemExit("评测文件中的 traces 不是列表")

    applicable: dict[str, int] = defaultdict(int)
    failures: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def 记录(
        category: str,
        failed: bool,
        item: dict[str, Any],
        events: list[str],
        metrics: dict[str, Any],
    ) -> None:
        applicable[category] += 1
        if not failed:
            return
        failures[category] += 1
        if len(examples[category]) < max(0, args.maximum_examples):
            examples[category].append(
                {
                    "record_id": item.get("record_id"),
                    "task": item.get("task"),
                    "events": events,
                    "metrics": metrics,
                }
            )

    for raw_item in traces:
        if not isinstance(raw_item, dict):
            continue
        trace = raw_item.get("trace", [])
        if not isinstance(trace, list):
            trace = []
        metrics = raw_item.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        events = [
            str(step.get("event", ""))
            for step in trace
            if isinstance(step, dict)
        ]
        task = str(raw_item.get("task", ""))

        记录("未完成", not events or events[-1] != "finish", raw_item, events, metrics)
        记录(
            "出现无效动作",
            any(event.startswith("invalid") for event in events),
            raw_item,
            events,
            metrics,
        )
        记录(
            "最终协议未满分",
            float(metrics.get("protocol_score", 0.0)) < 0.999,
            raw_item,
            events,
            metrics,
        )
        记录(
            "显式思考缺失",
            float(metrics.get("thinking_score", 0.0)) < 0.999,
            raw_item,
            events,
            metrics,
        )
        记录(
            "检索失败",
            float(metrics.get("retrieval_recall", 0.0)) < 0.999,
            raw_item,
            events,
            metrics,
        )
        if task in {"compose", "decision"}:
            记录(
                "组合失败",
                float(metrics.get("composition_f1", 0.0)) < 0.999,
                raw_item,
                events,
                metrics,
            )
        if "reflect" in events:
            记录(
                "反思未改善",
                float(
                    metrics.get(
                        "reflection_best_gain", metrics.get("reflection_gain", 0.0)
                    )
                )
                <= 0.0,
                raw_item,
                events,
                metrics,
            )
        if task == "decision":
            记录(
                "决策失败",
                float(metrics.get("decision_accuracy", 0.0)) < 0.999,
                raw_item,
                events,
                metrics,
            )
            记录(
                "规则不合规",
                float(metrics.get("rule_compliance", 0.0)) < 0.999,
                raw_item,
                events,
                metrics,
            )
            记录(
                "证据不足",
                float(metrics.get("evidence_coverage", 0.0)) < 0.999,
                raw_item,
                events,
                metrics,
            )

    categories = sorted(applicable)
    result = {
        "evaluation": str(args.evaluation),
        "samples": len(traces),
        "categories": {
            category: {
                "applicable": applicable[category],
                "failures": failures[category],
                "failure_rate": failures[category] / applicable[category]
                if applicable[category]
                else 0.0,
                "examples": examples[category],
            }
            for category in categories
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    主函数()
