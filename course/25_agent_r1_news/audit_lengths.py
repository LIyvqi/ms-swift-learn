"""用模型聊天模板检查多轮 SFT 序列长度，避免静默删除超长样本。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path


def 分位数(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * ratio), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="审计多轮 SFT token 长度")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4608)
    parser.add_argument("--knowledge", type=Path, help="可选的规则库 JSONL")
    parser.add_argument("--online-limit", type=int, default=5120)
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    lengths_by_task: dict[str, list[int]] = defaultdict(list)
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            token_ids = (
                encoded.get("input_ids", [])
                if isinstance(encoded, Mapping)
                else encoded
            )
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            lengths_by_task[row["task"]].append(len(token_ids))

    result = {}
    for task, lengths in sorted(lengths_by_task.items()):
        result[task] = {
            "samples": len(lengths),
            "minimum": min(lengths),
            "p50": 分位数(lengths, 0.5),
            "p95": 分位数(lengths, 0.95),
            "maximum": max(lengths),
            "over_limit": sum(length > args.limit for length in lengths),
        }
    output = {"limit": args.limit, "tasks": result}
    if args.knowledge:
        公开字段 = (
            "rule_id",
            "canonical_id",
            "title",
            "text",
            "category",
            "conditions",
            "exceptions",
            "priority",
            "source",
        )
        with args.knowledge.open(encoding="utf-8") as handle:
            rules = [json.loads(line) for line in handle if line.strip()]
        public_rules = [
            {key: rule[key] for key in 公开字段 if key in rule} for rule in rules
        ]
        rendered = json.dumps(public_rules, ensure_ascii=False, indent=2)
        rule_tokens = tokenizer.encode(rendered)
        output["knowledge"] = {
            "physical_rules": len(rules),
            "public_json_characters": len(rendered),
            "public_json_tokens": len(rule_tokens),
            "online_limit": args.online_limit,
            "over_online_limit": max(0, len(rule_tokens) - args.online_limit),
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
