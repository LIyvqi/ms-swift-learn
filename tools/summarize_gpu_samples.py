#!/usr/bin/env python3
"""按流水线阶段汇总 ROCm 外部采样的物理显存和 GPU 利用率。"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

步骤模式 = re.compile(r"^\[(?P<time>[^]]+)]\s+(?P<name>.+)$")


def 分位数(values: list[float], ratio: float) -> float:
    """计算离散最近秩分位数。"""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio + 0.999999) - 1))
    return ordered[index]


def 读取步骤(path: Path) -> list[tuple[float, str]]:
    """解析 runner 写入的 ISO-8601 阶段边界。"""

    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = 步骤模式.fullmatch(line.strip())
        if match:
            steps.append(
                (
                    datetime.fromisoformat(match.group("time")).timestamp(),
                    match.group("name"),
                )
            )
    return steps


def 读取采样(path: Path) -> list[dict]:
    """忽略失败的采样行，只汇总同时含显存和利用率的记录。"""

    rows = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if "物理显存GiB" in row and "GPU利用率" in row:
            rows.append(row)
    return rows


def 汇总阶段(samples: list[dict], steps: list[tuple[float, str]]) -> list[dict]:
    """用相邻阶段开始时间形成左闭右开区间。"""

    if not samples:
        raise RuntimeError("没有有效 GPU 采样")
    if not steps:
        raise RuntimeError("没有有效阶段时间戳")
    sample_end = max(float(row["时间戳"]) for row in samples) + 1e-6
    results = []
    for index, (start, name) in enumerate(steps):
        end = steps[index + 1][0] if index + 1 < len(steps) else sample_end
        rows = [row for row in samples if start <= float(row["时间戳"]) < end]
        if not rows:
            continue
        memory = [float(row["物理显存GiB"]) for row in rows]
        utilization = [float(row["GPU利用率"]) for row in rows]
        results.append(
            {
                "阶段": name,
                "采样数": len(rows),
                "持续秒数": max(
                    0.0, float(rows[-1]["时间戳"]) - float(rows[0]["时间戳"])
                ),
                "峰值物理显存GiB": max(memory),
                "平均物理显存GiB": sum(memory) / len(memory),
                "平均GPU利用率": sum(utilization) / len(utilization),
                "P95_GPU利用率": 分位数(utilization, 0.95),
            }
        )
    return results


def 渲染_markdown(results: list[dict]) -> str:
    """生成人可读的阶段效率表。"""

    lines = [
        "# 多模态流水线物理显存与利用率",
        "",
        "| 阶段 | 持续时间（秒） | 采样数 | 峰值显存（GiB） | 平均显存（GiB） | 平均 GPU 利用率 | P95 GPU 利用率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| {阶段} | {持续秒数:.1f} | {采样数} | {峰值物理显存GiB:.2f} | "
            "{平均物理显存GiB:.2f} | {平均GPU利用率:.2f}% | {P95_GPU利用率:.2f}% |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "说明：这是 `rocm-smi` 的按秒外部采样。极短瞬时峰值可能落在采样间隔之间，容量验收仍需同时检查是否 OOM 和框架内峰值。",
            "",
        ]
    )
    return "\n".join(lines)


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="汇总多模态流水线 GPU 采样")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    results = 汇总阶段(读取采样(args.samples), 读取步骤(args.steps))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(渲染_markdown(results), encoding="utf-8")
    print(f"GPU 阶段汇总已写入：{args.output_md}")


if __name__ == "__main__":
    主程序()
