"""把多个动态 Agent 评测 JSON 汇总成便于写入课程报告的 Markdown 表格。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def 嵌套取值(data: dict[str, Any], *keys: str) -> float | None:
    """读取嵌套数值；旧评测文件缺少新字段时返回空值。"""

    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return float(current) if isinstance(current, int | float) else None


def 从轨迹补充智能体指标(data: dict[str, Any]) -> dict[str, float]:
    """为早期未保存 agent_summary 的结果重算协议类指标。"""

    traces = data.get("traces", [])
    if not isinstance(traces, list) or not traces:
        return {}
    action_count = 0
    invalid_count = 0
    completed_count = 0
    turns = []
    for item in traces:
        trace = item.get("trace", []) if isinstance(item, dict) else []
        if not isinstance(trace, list):
            trace = []
        action_count += len(trace)
        invalid_count += sum(
            str(step.get("event", "")).startswith("invalid")
            for step in trace
            if isinstance(step, dict)
        )
        completed_count += int(
            bool(trace)
            and isinstance(trace[-1], dict)
            and trace[-1].get("event") == "finish"
        )
        turns.append(len(trace))
    return {
        "completion_rate": completed_count / len(traces),
        "invalid_action_rate": invalid_count / action_count if action_count else 0.0,
        "mean_turns": mean(turns),
    }


def 格式化(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def 智能体值(
    agent: dict[str, Any], fallback: dict[str, float], key: str
) -> float | None:
    """优先读取新版汇总字段，缺失时使用从轨迹重算的值。"""

    value = agent.get(key, fallback.get(key))
    return float(value) if isinstance(value, int | float) else None


def 主函数() -> None:
    parser = argparse.ArgumentParser(description="比较多个 Agent-R1 动态评测结果")
    parser.add_argument("results", nargs="+", type=Path, help="评测 JSON 文件")
    parser.add_argument("--labels", nargs="*", help="与输入文件一一对应的短名称")
    parser.add_argument("--output", type=Path, help="可选的 Markdown 输出文件")
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.results):
        raise SystemExit("--labels 数量必须与评测文件数量相同")

    headers = [
        "模型",
        "完成率",
        "无效动作率",
        "平均轮数",
        "检索 F1",
        "组合 F1",
        "决策准确率",
        "决策 Macro-F1",
        "决策规则 F1",
        "证据覆盖率",
        "决策 schema",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for index, path in enumerate(args.results):
        data = json.loads(path.read_text(encoding="utf-8"))
        fallback = 从轨迹补充智能体指标(data)
        agent = data.get("agent_summary", {})
        if not isinstance(agent, dict):
            agent = {}

        label = args.labels[index] if args.labels is not None else path.stem
        values = [
            label,
            格式化(智能体值(agent, fallback, "completion_rate")),
            格式化(智能体值(agent, fallback, "invalid_action_rate")),
            格式化(智能体值(agent, fallback, "mean_turns")),
            格式化(嵌套取值(data, "summary", "retrieve", "retrieval_f1")),
            格式化(嵌套取值(data, "summary", "compose", "composition_f1")),
            格式化(嵌套取值(data, "summary", "decision", "decision_accuracy")),
            格式化(智能体值(agent, fallback, "decision_macro_f1")),
            格式化(嵌套取值(data, "summary", "decision", "composition_f1")),
            格式化(嵌套取值(data, "summary", "decision", "evidence_coverage")),
            格式化(嵌套取值(data, "summary", "decision", "task_schema_score")),
        ]
        lines.append("| " + " | ".join(values) + " |")

    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    主函数()
