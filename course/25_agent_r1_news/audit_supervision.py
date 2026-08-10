"""审计 Qwen3.5/ms-swift 编码后显式思考是否真正进入 SFT 损失。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from statistics import mean
from typing import Any

项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

环境模块 = import_module("course.25_agent_r1_news.agent_system")
思考模式 = 环境模块.思考模式


def 选择样本(path: Path, maximum_per_task: int) -> list[dict[str, Any]]:
    """按任务等量读取前若干样本，避免只审计文件开头的单一任务。"""

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row["task"])
            if maximum_per_task > 0 and counts[task] >= maximum_per_task:
                continue
            selected.append(row)
            counts[task] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 Agent-R1 SFT 显式思考的损失掩码")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--maximum-per-task", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4608)
    args = parser.parse_args()
    if args.maximum_per_task < 0:
        parser.error("--maximum-per-task 不能小于 0；0 表示审计全部样本")

    from swift.model import get_processor
    from swift.template import get_template

    processor = get_processor(str(args.model))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    template = get_template(
        processor,
        enable_thinking=True,
        max_length=args.max_length,
    )
    template.mode = "train"
    template.init_processor(processor)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failures = []
    for row in 选择样本(args.dataset, args.maximum_per_task):
        encoded = template.encode(row, return_length=True)
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        supervised_ids = [
            token for token, label in zip(input_ids, labels) if label != -100
        ]
        supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
        assistant_messages = [
            message
            for message in row["messages"]
            if message.get("role") == "assistant"
        ]
        raw_thinking_count = sum(
            bool(思考模式.search(str(message.get("content", ""))))
            for message in assistant_messages
        )
        supervised_thinking_count = len(思考模式.findall(supervised_text))
        task = str(row["task"])
        metrics[task]["samples"].append(1.0)
        metrics[task]["input_tokens"].append(float(len(input_ids)))
        metrics[task]["supervised_tokens"].append(float(len(supervised_ids)))
        metrics[task]["assistant_targets"].append(float(len(assistant_messages)))
        metrics[task]["raw_thinking_targets"].append(float(raw_thinking_count))
        metrics[task]["supervised_thinking_targets"].append(
            float(supervised_thinking_count)
        )
        if (
            raw_thinking_count != len(assistant_messages)
            or supervised_thinking_count < len(assistant_messages)
        ):
            failures.append(str(row.get("record_id", "未知记录")))

    summary = {}
    for task, values in sorted(metrics.items()):
        input_tokens = sum(values["input_tokens"])
        supervised_tokens = sum(values["supervised_tokens"])
        summary[task] = {
            "samples": int(sum(values["samples"])),
            "assistant_targets": int(sum(values["assistant_targets"])),
            "raw_thinking_targets": int(sum(values["raw_thinking_targets"])),
            "supervised_thinking_targets": int(
                sum(values["supervised_thinking_targets"])
            ),
            "mean_input_tokens": mean(values["input_tokens"]),
            "mean_supervised_tokens": mean(values["supervised_tokens"]),
            "supervised_token_ratio": (
                supervised_tokens / input_tokens if input_tokens else 0.0
            ),
        }
    report = {
        "dataset": str(args.dataset),
        "enable_thinking": True,
        "max_length": args.max_length,
        "tasks": summary,
        "failed_record_ids": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("存在未进入监督损失的显式思考目标")


if __name__ == "__main__":
    main()
