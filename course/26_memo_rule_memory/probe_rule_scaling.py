#!/usr/bin/env python3
"""用真实 Qwen 模板测量全规则 Prompt 与参数化 Memory 请求的规模增长。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swift.model import get_processor
from memo_core import Executive消息, 单轮记忆问题, 规则上下文, 记忆消息, 读_jsonl


项目根目录 = Path(__file__).resolve().parents[2]


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="规则规模 Prompt token 探针")
    parser.add_argument("--model", default=str(项目根目录 / "models/Qwen3.5-0.8B-Base"))
    parser.add_argument("--sizes", default="80,400,800,1600")
    parser.add_argument(
        "--output", type=Path,
        default=项目根目录 / "outputs/26_memo_rule_memory/scaling_probe.json",
    )
    return parser.parse_args()


def 模板长度(tokenizer, messages: list[dict[str, str]]) -> int:
    """返回真实聊天模板编码后的 token 数。"""

    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    return int(ids.shape[-1])


def 扩展规则(rules: list[dict], size: int) -> list[dict]:
    """循环复制规则并只改稳定 ID，用于隔离知识库规模变量。"""

    expanded = []
    for index in range(size):
        source = rules[index % len(rules)]
        clone = dict(source)
        clone["rule_id"] = f"S{index // len(rules):03d}-{source['rule_id']}"
        expanded.append(clone)
    return expanded


def 主程序() -> None:
    args = 解析参数()
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    rules = 读_jsonl(项目根目录 / "datasets/memo_rule_memory/rules.jsonl")
    case = 读_jsonl(项目根目录 / "datasets/memo_rule_memory/audit_val.jsonl")[0]
    processor = get_processor(args.model)
    tokenizer = processor.tokenizer
    memory_request = 记忆消息(单轮记忆问题(case))
    memory_tokens = 模板长度(tokenizer, memory_request)
    rows = []
    for size in sizes:
        context = 规则上下文(扩展规则(rules, size), compact=True)
        messages = Executive消息(case, context, "all_rules")
        rows.append({
            "rules": size,
            "all_rules_context_chars": len(context),
            "all_rules_prompt_tokens": 模板长度(tokenizer, messages),
            "memory_request_tokens": memory_tokens,
        })
    result = {"model": args.model, "measurements": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
