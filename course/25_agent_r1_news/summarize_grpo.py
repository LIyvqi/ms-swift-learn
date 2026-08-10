"""汇总 ms-swift GRPO 日志的整体与最近窗口指标。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def 读取日志(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """分别返回全部训练 step 与真正执行 rollout、带奖励的 step。"""

    训练记录 = []
    奖励记录 = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "global_step/max_steps" not in row or "loss" not in row:
                continue
            训练记录.append(row)
            if "reward" in row:
                奖励记录.append(row)
    return 训练记录, 奖励记录


def 有限平均(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values.append(float(value))
    return mean(values) if values else None


def 训练健康(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总裁剪前 loss/梯度尖峰，并把最大值关联到具体全局 step。"""

    result: dict[str, Any] = {}
    for key, display_name, threshold in (
        ("loss", "loss", 1.0),
        ("grad_norm", "grad_norm", 1000.0),
    ):
        finite = []
        nonfinite = 0
        for row in rows:
            value = row.get(key)
            if not isinstance(value, int | float):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                finite.append((numeric, str(row.get("global_step/max_steps", ""))))
            else:
                nonfinite += 1
        maximum = max(finite, default=None, key=lambda item: item[0])
        result[f"最大_{display_name}"] = (
            {"value": maximum[0], "global_step": maximum[1]} if maximum else None
        )
        result[f"{display_name}_非有限_step数"] = nonfinite
        result[f"{display_name}_大于_{threshold:g}_step数"] = sum(
            value > threshold for value, _ in finite
        )
    return result


def 汇总(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    基础字段 = [
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "completions/mean_length",
        "completions/clipped_ratio",
        "num_turns",
        "kl",
        "step_time",
    ]
    result = {key: 有限平均(rows, key) for key in 基础字段}
    result["峰值逻辑显存_GiB"] = max(float(row.get("memory(GiB)", 0.0)) for row in rows)
    奖励字段 = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("rewards/") and key.endswith("/mean")
        }
    )
    result["分项奖励平均"] = {
        key.removeprefix("rewards/").removesuffix("/mean"): 有限平均(rows, key)
        for key in 奖励字段
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 Agent-R1 GRPO 训练日志")
    parser.add_argument("日志", type=Path)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    训练记录, 奖励记录 = 读取日志(args.日志)
    if not 训练记录:
        raise SystemExit("日志中还没有 GRPO 训练记录")
    if not 奖励记录:
        raise SystemExit("日志中还没有带奖励的 rollout 记录")
    window = max(1, args.window)
    result = {
        "日志": str(args.日志),
        "当前步": 训练记录[-1]["global_step/max_steps"],
        "训练_step_数": len(训练记录),
        "rollout_批次数": len(奖励记录),
        "端到端平均步时_秒": 有限平均(训练记录, "step_time"),
        "训练健康": 训练健康(训练记录),
        "全部_rollout": 汇总(奖励记录),
        f"最近_{min(window, len(奖励记录))}_个_rollout": 汇总(奖励记录[-window:]),
        "最新奖励记录": 奖励记录[-1],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
