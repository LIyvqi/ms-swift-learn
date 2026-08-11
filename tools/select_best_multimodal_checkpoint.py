#!/usr/bin/env python3
"""从最近一次多模态训练中选择验证损失最低的检查点。"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def 读取验证记录(日志路径: Path) -> list[tuple[float, int]]:
    """返回日志中的“验证损失、全局步数”，并忽略不完整记录。"""

    记录: list[tuple[float, int]] = []
    for 原始行 in 日志路径.read_text(encoding="utf-8").splitlines():
        try:
            行 = json.loads(原始行)
        except json.JSONDecodeError:
            continue
        if "eval_loss" not in 行:
            continue
        步数字段 = str(行.get("global_step/max_steps", ""))
        匹配 = re.match(r"^(\d+)/", 步数字段)
        if 匹配 is None:
            continue
        记录.append((float(行["eval_loss"]), int(匹配.group(1))))
    return 记录


def 选择检查点(输出根目录: Path, 运行目录模式: str) -> Path:
    """先锁定最近一次运行，再在该运行内按验证损失选择检查点。"""

    候选运行 = [Path(路径) for 路径 in glob.glob(str(输出根目录 / 运行目录模式))]
    候选运行 = [路径 for 路径 in 候选运行 if 路径.is_dir()]
    if not 候选运行:
        raise FileNotFoundError(f"找不到运行目录：{运行目录模式}")

    最近运行 = max(候选运行, key=lambda 路径: 路径.stat().st_mtime)
    日志路径 = 最近运行 / "logging.jsonl"
    if 日志路径.is_file():
        for _, 步数 in sorted(读取验证记录(日志路径)):
            检查点 = 最近运行 / f"checkpoint-{步数}"
            if 检查点.is_dir():
                return 检查点

    检查点列表 = [路径 for 路径 in 最近运行.glob("checkpoint-*") if 路径.is_dir()]
    if not 检查点列表:
        raise FileNotFoundError(f"运行目录内没有检查点：{最近运行}")
    return max(检查点列表, key=lambda 路径: 路径.stat().st_mtime)


def main() -> None:
    解析器 = argparse.ArgumentParser(description="选择最近一次多模态运行的最佳检查点")
    解析器.add_argument("--output-root", type=Path, required=True, help="outputs 根目录")
    解析器.add_argument("--run-glob", required=True, help="相对于 outputs 的运行目录模式")
    参数 = 解析器.parse_args()
    print(选择检查点(参数.output_root.resolve(), 参数.run_glob).resolve())


if __name__ == "__main__":
    main()
