"""按不重叠的 step 区间汇总多次断点续训产生的 GRPO 日志。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from summarize_grpo import 有限平均, 汇总, 读取日志


@dataclass(frozen=True)
class 日志分段:
    """一份日志中真正进入最终训练轨迹的闭区间。"""

    路径: Path
    起始步: int
    结束步: int


def 解析全局步(row: dict[str, Any]) -> int:
    """从 ms-swift 的 ``当前步/总步数`` 字段中取出当前步。"""

    value = str(row.get("global_step/max_steps", ""))
    try:
        return int(value.split("/", 1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法解析全局步：{value!r}") from exc


def 解析分段(value: str) -> 日志分段:
    """解析 ``日志路径:起始步:结束步``。"""

    try:
        path_text, start_text, end_text = value.rsplit(":", 2)
        segment = 日志分段(Path(path_text), int(start_text), int(end_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "分段格式必须是 日志路径:起始步:结束步"
        ) from exc
    if segment.起始步 < 1 or segment.结束步 < segment.起始步:
        raise argparse.ArgumentTypeError("分段步数必须满足 1 <= 起始步 <= 结束步")
    return segment


def 读取分段(segment: 日志分段) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """读取一个闭区间，并验证区间中的训练 step 连续且完整。"""

    training_rows, reward_rows = 读取日志(segment.路径)
    selected_training = [
        row
        for row in training_rows
        if segment.起始步 <= 解析全局步(row) <= segment.结束步
    ]
    selected_rewards = [
        row
        for row in reward_rows
        if segment.起始步 <= 解析全局步(row) <= segment.结束步
    ]
    actual_steps = [解析全局步(row) for row in selected_training]
    expected_steps = list(range(segment.起始步, segment.结束步 + 1))
    if actual_steps != expected_steps:
        actual = f"{actual_steps[0]}～{actual_steps[-1]}" if actual_steps else "空"
        missing = sorted(set(expected_steps) - set(actual_steps))
        duplicate_count = len(actual_steps) - len(set(actual_steps))
        preview = missing[:10]
        raise ValueError(
            f"{segment.路径} 的请求区间 {segment.起始步}～{segment.结束步} "
            f"不完整；实际范围 {actual}，缺失步示例 {preview}，重复 {duplicate_count} 步"
        )
    return selected_training, selected_rewards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="拼接多次断点续训日志，排除失败运行和重复 step"
    )
    parser.add_argument(
        "--segment",
        action="append",
        required=True,
        type=解析分段,
        help="可重复传入，格式为 日志路径:起始步:结束步",
    )
    parser.add_argument("--window", type=int, default=100, help="最近 rollout 窗口")
    parser.add_argument(
        "--bucket-rollouts",
        type=int,
        default=0,
        help="按固定 rollout 数输出连续窗口，0 表示关闭",
    )
    args = parser.parse_args()

    segments = sorted(args.segment, key=lambda item: item.起始步)
    for previous, current in pairwise(segments):
        if current.起始步 != previous.结束步 + 1:
            raise SystemExit(
                f"相邻分段不连续：{previous.结束步} 后面是 {current.起始步}"
            )

    all_training: list[dict[str, Any]] = []
    all_rewards: list[dict[str, Any]] = []
    segment_summaries = []
    for segment in segments:
        training_rows, reward_rows = 读取分段(segment)
        if not reward_rows:
            raise SystemExit(f"分段没有带奖励的 rollout：{segment.路径}")
        all_training.extend(training_rows)
        all_rewards.extend(reward_rows)
        segment_summaries.append(
            {
                "日志": str(segment.路径),
                "步区间": f"{segment.起始步}～{segment.结束步}",
                "训练_step_数": len(training_rows),
                "rollout_批次数": len(reward_rows),
                "rollout_汇总": 汇总(reward_rows),
            }
        )

    window = max(1, args.window)
    result = {
        "正式轨迹步区间": f"{segments[0].起始步}～{segments[-1].结束步}",
        "训练_step_数": len(all_training),
        "rollout_批次数": len(all_rewards),
        "端到端平均步时_秒": 有限平均(all_training, "step_time"),
        "全部_rollout": 汇总(all_rewards),
        f"最近_{min(window, len(all_rewards))}_个_rollout": 汇总(all_rewards[-window:]),
        "分段": segment_summaries,
    }
    bucket_size = max(0, args.bucket_rollouts)
    if bucket_size:
        buckets = []
        for start in range(0, len(all_rewards), bucket_size):
            rows = all_rewards[start : start + bucket_size]
            buckets.append(
                {
                    "步区间": f"{解析全局步(rows[0])}～{解析全局步(rows[-1])}",
                    "rollout_批次数": len(rows),
                    "指标": 汇总(rows),
                }
            )
        result[f"连续窗口_每{bucket_size}个rollout"] = buckets
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
