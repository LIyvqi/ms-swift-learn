#!/usr/bin/env python3
"""审计每轮显式规划和动作是否真正进入 SFT 监督损失。"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))
环境模块 = import_module("course.31_hierarchical_memory_agent.agent_environment")
思考模式 = 环境模块.思考模式


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="审计分层记忆 Agent 的训练掩码")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--maximum-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=6144)
    args = parser.parse_args()

    from swift.model import get_processor
    from swift.template import get_template

    processor = get_processor(str(args.model))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    template = get_template(processor, enable_thinking=True, max_length=args.max_length)
    template.mode = "train"
    template.init_processor(processor)
    failures = []
    samples = 0
    assistant_targets = 0
    supervised_thinking = 0
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            encoded = template.encode(row, return_length=True)
            supervised_ids = [
                token
                for token, label in zip(encoded["input_ids"], encoded["labels"])
                if label != -100
            ]
            supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
            targets = [message for message in row["messages"] if message["role"] == "assistant"]
            current_supervised = len(思考模式.findall(supervised_text))
            assistant_targets += len(targets)
            supervised_thinking += current_supervised
            if current_supervised < len(targets):
                failures.append(row["record_id"])
            samples += 1
            if args.maximum_samples > 0 and samples >= args.maximum_samples:
                break
    report = {
        "samples": samples,
        "assistant_targets": assistant_targets,
        "supervised_thinking_targets": supervised_thinking,
        "failed_record_ids": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("存在未进入监督损失的显式规划")


if __name__ == "__main__":
    主程序()
