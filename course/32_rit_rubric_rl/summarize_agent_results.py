#!/usr/bin/env python3
"""从被 Git 忽略的完整轨迹中生成可复核的 Agent 对照表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
默认输出目录 = 项目根目录 / "outputs/32_rit_rubric_rl/agent"
默认实验 = (
    ("SFT + 两库", "sft_test_with_memory.json"),
    ("SFT + 空库消融", "sft_test_without_memory.json"),
    ("ORM-GRPO + 两库", "orm_test_with_memory.json"),
    ("RiT-GRPO + 两库", "rit_test_with_memory.json"),
)


def 百分比(value: Any) -> str:
    """把零到一指标转成两位百分比。"""

    return f"{100 * float(value):.2f}%"


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="汇总 RiT 审核 Agent 的真实生成结果")
    parser.add_argument("--input-dir", type=Path, default=默认输出目录)
    args = parser.parse_args()

    rows = []
    for label, filename in 默认实验:
        path = args.input_dir / filename
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        summary = result["summary"]
        traces = result["traces"]
        predicted_safe = sum(row.get("predicted_is_safe") is True for row in traces)
        rows.append(
            {
                "实验": label,
                "完成率": 百分比(summary["completion_rate"]),
                "安全准确率": 百分比(summary["safety_accuracy"]),
                "Exact": 百分比(summary["response_reward"]),
                "样本 F1": 百分比(summary["category_f1"]),
                "Micro-F1": 百分比(summary["category_micro_f1"]),
                "过程分": 百分比(summary["process_reward"]),
                "规则落地": 百分比(summary["rule_grounding"]),
                "案例落地": 百分比(summary["case_grounding"]),
                "预测 SAFE": 百分比(predicted_safe / max(len(traces), 1)),
                "耗时": f"{float(summary['elapsed_seconds']):.1f} 秒",
            }
        )
    if not rows:
        raise SystemExit("没有找到任何 Agent 评测结果")
    columns = list(rows[0])
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[column]) for column in columns) + " |")


if __name__ == "__main__":
    主程序()
