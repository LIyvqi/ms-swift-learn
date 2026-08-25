#!/usr/bin/env python3
"""按第 30 课阶段汇总 ROCm 外部显存和利用率采样。"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


步骤格式 = re.compile(r"^\[(?P<time>[^]]+)]\s+(?P<name>.+)$")


def 解析参数() -> argparse.Namespace:
    """定义监控文件和输出目录。"""

    parser = argparse.ArgumentParser(description="汇总 Macaron 课程 GPU 采样")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def 分位数(values: list[float], ratio: float) -> float:
    """计算离散最近秩分位数。"""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio + 0.999999) - 1))
    return ordered[index]


def 主程序() -> None:
    """用相邻阶段开始时间形成区间并生成双格式汇总。"""

    args = 解析参数()
    samples = []
    for line in args.samples.open(encoding="utf-8"):
        row = json.loads(line)
        if "物理显存GiB" in row and "GPU利用率" in row:
            samples.append(row)
    steps = []
    for line in args.steps.read_text(encoding="utf-8").splitlines():
        match = 步骤格式.fullmatch(line.strip())
        if match:
            steps.append((datetime.fromisoformat(match.group("time")).timestamp(), match.group("name")))
    if not samples or not steps:
        raise RuntimeError("缺少有效 GPU 采样或阶段记录")
    sample_end = max(float(row["时间戳"]) for row in samples) + 1e-6
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    grouped_duration: dict[str, float] = defaultdict(float)
    for index, (start, name) in enumerate(steps):
        end = steps[index + 1][0] if index + 1 < len(steps) else sample_end
        rows = [row for row in samples if start <= float(row["时间戳"]) < end]
        if not rows or name.startswith("流水线失败"):
            continue
        grouped_rows[name].extend(rows)
        grouped_duration[name] += max(0.0, float(rows[-1]["时间戳"]) - float(rows[0]["时间戳"]))
    results = []
    for name, rows in grouped_rows.items():
        memory = [float(row["物理显存GiB"]) for row in rows]
        utilization = [float(row["GPU利用率"]) for row in rows]
        results.append(
            {
                "阶段": name,
                "持续秒数": grouped_duration[name],
                "采样数": len(rows),
                "峰值物理显存GiB": max(memory),
                "平均物理显存GiB": sum(memory) / len(memory),
                "平均GPU利用率": sum(utilization) / len(utilization),
                "P95_GPU利用率": 分位数(utilization, 0.95),
            }
        )
    lines = [
        "# Macaron 多 LoRA 课程 GPU 使用汇总",
        "",
        "| 阶段 | 持续秒数 | 采样数 | 峰值显存 GiB | 平均显存 GiB | 平均 GPU 利用率 | P95 利用率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['阶段']} | {row['持续秒数']:.1f} | {row['采样数']} | "
            f"{row['峰值物理显存GiB']:.2f} | {row['平均物理显存GiB']:.2f} | "
            f"{row['平均GPU利用率']:.2f}% | {row['P95_GPU利用率']:.2f}% |"
        )
    lines.extend(["", "说明：指标来自 `rocm-smi` 每秒外部采样，极短瞬时峰值可能未命中。", ""])
    args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"GPU 汇总已写入：{args.output_md}")


if __name__ == "__main__":
    主程序()
