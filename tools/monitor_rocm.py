#!/usr/bin/env python3
"""按固定间隔采样 ROCm 物理显存和利用率，供训练吞吐分析使用。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def 进程存在(pid: int) -> bool:
    """使用零信号判断被监控进程是否仍存在。"""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def 采样() -> dict:
    """读取第一张 ROCm GPU 的 JSON 指标。"""

    completed = subprocess.run(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    card = payload[min(payload)]
    total = int(card["VRAM Total Memory (B)"])
    used = int(card["VRAM Total Used Memory (B)"])
    return {
        "时间": datetime.now(UTC).isoformat(),
        "时间戳": time.time(),
        "GPU利用率": float(card["GPU use (%)"]),
        "物理显存字节": used,
        "物理显存GiB": used / 1024**3,
        "物理显存比例": used / total,
        "物理显存总量字节": total,
    }


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="持续采样 ROCm GPU 状态")
    parser.add_argument("--pid", type=int, required=True, help="该进程退出时停止采样")
    parser.add_argument("--output", type=Path, required=True, help="JSONL 输出路径")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔秒数")
    parser.add_argument(
        "--maximum-samples", type=int, default=0, help="测试用最大采样数，0 表示不限"
    )
    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("interval 必须大于 0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # 每次正式 runner 都生成一份自洽的新记录，避免失败重跑后混入旧时间段。
    with args.output.open("w", encoding="utf-8") as handle:
        while 进程存在(args.pid):
            try:
                row = 采样()
            except (
                OSError,
                subprocess.SubprocessError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as error:
                row = {
                    "时间": datetime.now(UTC).isoformat(),
                    "时间戳": time.time(),
                    "采样错误": f"{type(error).__name__}: {error}",
                }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            if args.maximum_samples and count >= args.maximum_samples:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    主程序()
