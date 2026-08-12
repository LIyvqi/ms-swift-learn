#!/usr/bin/env python3
"""用留出改写问题评测 Base 与各轮规则 Memory 检查点。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from swift import InferRequest, RequestConfig, TransformersEngine

from memo_core import 解析记忆, 读_jsonl, 集合指标, 记忆消息, 写_jsonl


项目根目录 = Path(__file__).resolve().parents[2]


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测规则 Memory 的留出问答")
    parser.add_argument("--model", type=Path, default=项目根目录 / "models/Qwen3.5-0.8B-Base")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path, default=项目根目录 / "datasets/memo_rule_memory/memory_val.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser.parse_args()


def 主程序() -> None:
    args = 解析参数()
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_impl": "eager",
        "device_map": "cuda:0",
        "max_batch_size": args.batch_size,
    }
    if args.adapter:
        kwargs["adapters"] = [str(args.adapter)]
    engine = TransformersEngine(str(args.model), **kwargs)
    generation = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0)
    rows = 读_jsonl(args.dataset)
    requests = [
        InferRequest(messages=记忆消息(next(message["content"] for message in row["messages"] if message["role"] == "user")))
        for row in rows
    ]
    started = time.perf_counter()
    responses = []
    for start in range(0, len(requests), args.batch_size):
        responses.extend(engine.infer(requests[start:start + args.batch_size], request_config=generation))
    elapsed = time.perf_counter() - started

    traces = []
    rule_scores = []
    decision_hits = []
    format_hits = []
    fact_hits = []
    for row, response in zip(rows, responses):
        text = response.choices[0].message.content or ""
        prediction = 解析记忆(text)
        gold = 解析记忆(row["messages"][-1]["content"])
        precision, recall, f1 = 集合指标(prediction["rule_ids"], gold["rule_ids"])
        rule_scores.append((precision, recall, f1))
        decision_hits.append(prediction["decision"] == gold["decision"])
        format_hits.append(prediction["valid"])
        gold_facts = " ".join(gold["facts"])
        predicted_facts = " ".join(prediction["facts"])
        anchors = [value for value in gold["rule_ids"]] + [gold["decision"]]
        fact_hits.append(sum(value in text + predicted_facts for value in anchors) / len(anchors))
        traces.append(
            {
                "qa_type": row["qa_type"],
                "source_rule_ids": row["source_rule_ids"],
                "question": requests[len(traces)].messages[-1]["content"],
                "gold": gold,
                "response": text,
                "prediction": prediction,
                "rule_precision": precision,
                "rule_recall": recall,
                "rule_f1": f1,
                "fact_anchor_coverage": fact_hits[-1],
                "gold_fact_chars": len(gold_facts),
            }
        )
    summary = {
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "samples": len(traces),
        "format_rate": sum(format_hits) / len(format_hits),
        "decision_accuracy": sum(decision_hits) / len(decision_hits),
        "rule_precision": sum(value[0] for value in rule_scores) / len(rule_scores),
        "rule_recall": sum(value[1] for value in rule_scores) / len(rule_scores),
        "rule_f1": sum(value[2] for value in rule_scores) / len(rule_scores),
        "fact_anchor_coverage": sum(fact_hits) / len(fact_hits),
        "elapsed_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    写_jsonl(args.output, traces)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
