#!/usr/bin/env python3
"""用可控反例验证 RiT 硬门控不会奖励错误答案或空思考。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))
核心 = importlib.import_module("course.32_rit_rubric_rl.rit_core")


def 读取(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 评分(row: dict[str, Any], completion: str, gate: str) -> dict[str, Any]:
    detail = 核心.计算本地RiT奖励(
        completion,
        row["prompt_text"],
        row["response_text"],
        row["gold_is_safe"],
        row["gold_categories"],
        alpha=1.0,
        gate=gate,
    )
    return detail.转字典()


def 构造反例(row: dict[str, Any]) -> dict[str, str]:
    """从同一条专家轨迹构造只改变一个因素的奖励单元实验。"""

    gold = row["messages"][-1]["content"]
    answer = gold[gold.rfind("<answer>") :]
    shallow = "<think>直接判断。</think>\n" + answer
    hallucinated = gold.replace(
        核心.提取标签(核心.提取思考(gold), "evidence")[0],
        "这是一段输入中从未出现的虚构证据",
        1,
    )
    no_boundary = gold.replace(
        f"<boundary>{核心.提取标签(核心.提取思考(gold), 'boundary')[0]}</boundary>\n",
        "",
        1,
    )
    if row["gold_is_safe"] == "false":
        wrong = gold.replace(
            "<decision>UNSAFE</decision>", "<decision>SAFE</decision>", 1
        )
        wrong = wrong[: wrong.rfind("<answer>")] + (
            '<answer>{"is_safe":true,"categories":[]}</answer>'
        )
    else:
        wrong = gold.replace(
            "<decision>SAFE</decision>", "<decision>UNSAFE</decision>", 1
        )
        wrong = wrong[: wrong.rfind("<answer>")] + (
            '<answer>{"is_safe":false,"categories":["animal_abuse"]}</answer>'
        )
    return {
        "专家正确思考与正确答案": gold,
        "空思考但答案正确": shallow,
        "证据幻觉但答案正确": hallucinated,
        "缺少边界核对但答案正确": no_boundary,
        "结构完整但答案错误": wrong,
    }


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="审计 RiT 硬门控奖励设计")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=项目根目录 / "datasets/rit_audit/sft_train.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=项目根目录 / "outputs/32_rit_rubric_rl/reward_audit.json",
    )
    args = parser.parse_args()
    rows = 读取(args.dataset)
    unsafe = next(row for row in rows if row["gold_is_safe"] == "false")
    cases = 构造反例(unsafe)
    controlled = {
        name: {
            gate: 评分(unsafe, completion, gate)
            for gate in ("min", "none", "max", "conditional")
        }
        for name, completion in cases.items()
    }

    expert_details = [
        核心.计算本地RiT奖励(
            row["messages"][-1]["content"],
            row["prompt_text"],
            row["response_text"],
            row["gold_is_safe"],
            row["gold_categories"],
        )
        for row in rows
    ]
    alpha_grid = {
        str(alpha): {
            "正确答案_思考0.5": 核心.融合奖励(0.5, 1.0, alpha=alpha, gate="min"),
            "错误答案_思考1.0": 核心.融合奖励(1.0, 0.0, alpha=alpha, gate="min"),
        }
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    result = {
        "dataset": str(args.dataset),
        "samples": len(rows),
        "expert_mean_response": mean(item.响应奖励 for item in expert_details),
        "expert_mean_thinking": mean(item.思考奖励 for item in expert_details),
        "expert_mean_gated": mean(item.最终奖励 for item in expert_details),
        "controlled_counterexamples": controlled,
        "alpha_grid": alpha_grid,
        "interpretation": (
            "min gate 使错误答案的最终奖励恒为 0；答案正确时，思考缺陷会按 alpha 降低奖励。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        "samples": result["samples"],
        "expert_means": {
            "response": result["expert_mean_response"],
            "thinking": result["expert_mean_thinking"],
            "gated": result["expert_mean_gated"],
        },
        "controlled": {
            name: {
                gate: values["最终奖励"] for gate, values in gates.items()
            }
            for name, gates in controlled.items()
        },
        "alpha_grid": alpha_grid,
        "output": str(args.output),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
