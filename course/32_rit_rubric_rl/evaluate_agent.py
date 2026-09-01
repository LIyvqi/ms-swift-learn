#!/usr/bin/env python3
"""让真实 Qwen LoRA 逐轮调用两库，并统计隔离测试上的 Agent 指标。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from statistics import mean
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))
记忆模块 = import_module("course.32_rit_rubric_rl.agent_memory")
环境模块 = import_module("course.32_rit_rubric_rl.agent_environment")
极简审核记忆 = 记忆模块.极简审核记忆
极简RiT审核环境 = 环境模块.极简RiT审核环境


def 读取_jsonl(path: Path, maximum: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if maximum > 0 and len(rows) >= maximum:
                break
    return rows


def 文件摘要(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 汇总多标签(traces: list[dict[str, Any]], categories: list[str]) -> dict[str, float]:
    per_category = []
    micro_tp = micro_fp = micro_fn = 0
    for category in categories:
        tp = sum(category in row["predicted_categories"] and category in row["gold_categories"] for row in traces)
        fp = sum(category in row["predicted_categories"] and category not in row["gold_categories"] for row in traces)
        fn = sum(category not in row["predicted_categories"] and category in row["gold_categories"] for row in traces)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_category.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"category_macro_f1": mean(per_category), "category_micro_f1": micro}


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="评测极简 RiT 安全审核 Agent")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--model", type=Path, default=项目根目录 / "models/Qwen3.5-0.8B-Base"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=项目根目录 / "datasets/rit_audit_agent/rl_test.jsonl",
    )
    parser.add_argument(
        "--rules", type=Path, default=项目根目录 / "datasets/rit_audit_agent/rules.jsonl"
    )
    parser.add_argument(
        "--cases", type=Path, default=项目根目录 / "datasets/rit_audit_agent/cases.jsonl"
    )
    parser.add_argument("--maximum-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=项目根目录 / "outputs/32_rit_rubric_rl/agent/model_evaluation.json",
    )
    args = parser.parse_args()

    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine

    began = time.perf_counter()
    engine = TransformersEngine(
        str(args.model),
        adapters=[str(args.adapter)] if args.adapter else None,
        torch_dtype=torch.bfloat16,
        attn_impl="eager",
        device_map="cuda:0",
        max_batch_size=args.batch_size,
    )
    generation = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0)
    memory = 极简审核记忆(args.rules, args.cases)
    rows = 读取_jsonl(args.dataset, args.maximum_samples)
    states = []
    for row in rows:
        config = deepcopy(row["env_config"])
        config["memory_disabled"] = args.disable_memory
        environment = 极简RiT审核环境(memory, config)
        observation, _, system = environment.reset()
        states.append(
            {
                "row": row,
                "environment": environment,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": observation},
                ],
                "info": {},
            }
        )
    active = list(states)
    while active:
        for start in range(0, len(active), args.batch_size):
            chunk = active[start : start + args.batch_size]
            requests = [
                InferRequest(
                    messages=state["messages"],
                    chat_template_kwargs={"enable_thinking": False},
                )
                for state in chunk
            ]
            responses = engine.infer(requests, request_config=generation, use_tqdm=False)
            for state, response in zip(chunk, responses):
                completion = str(response.choices[0].message.content or "")
                state["messages"].append({"role": "assistant", "content": completion})
                observation, _, done, info = state["environment"].step(completion)
                state["info"] = info
                if not done:
                    state["messages"].append({"role": "user", "content": observation})
        active = [state for state in active if not state["environment"].done]
    elapsed = time.perf_counter() - began
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    traces = []
    metric_values: dict[str, list[float]] = defaultdict(list)
    rubric_values: dict[str, list[float]] = defaultdict(list)
    for state in states:
        info = state["info"]
        config = state["row"]["env_config"]
        trace = info.get("trace", [])
        metrics = info.get("metrics", {})
        final = info.get("final", {})
        for name, value in metrics.items():
            metric_values[name].append(float(value))
        for name, value in info.get("rubric_scores", {}).items():
            rubric_values[name].append(float(value))
        predicted_categories = list(final.get("categories", []))
        traces.append(
            {
                "record_id": state["row"]["record_id"],
                "gold_is_safe": bool(config["is_safe"]),
                "gold_categories": list(config["categories"]),
                "predicted_is_safe": final.get("is_safe"),
                "predicted_categories": predicted_categories,
                "completed": bool(trace) and trace[-1]["event"] == "finish",
                "turns": len(trace),
                "invalid_actions": sum(str(item["event"]).startswith("invalid") for item in trace),
                "search_rule_calls": sum(item["event"] == "search_rule" for item in trace),
                "search_case_calls": sum(item["event"] == "search_case" for item in trace),
                "metrics": metrics,
                "rubric_scores": info.get("rubric_scores", {}),
                "final": final,
                "messages": state["messages"],
                "trace": trace,
            }
        )
    summary = {
        "samples": len(traces),
        "memory_enabled": not args.disable_memory,
        "completion_rate": mean(row["completed"] for row in traces),
        "invalid_action_rate": sum(row["invalid_actions"] for row in traces) / max(sum(row["turns"] for row in traces), 1),
        "mean_turns": mean(row["turns"] for row in traces),
        "mean_search_rule_calls": mean(row["search_rule_calls"] for row in traces),
        "mean_search_case_calls": mean(row["search_case_calls"] for row in traces),
        **{name: mean(values) for name, values in sorted(metric_values.items())},
        **汇总多标签(traces, list(memory.类别名称())),
        "rubric_pass_rates": {name: mean(values) for name, values in sorted(rubric_values.items())},
        "elapsed_seconds": elapsed,
    }
    result = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "dataset_sha256": 文件摘要(args.dataset),
        "rules_sha256": 文件摘要(args.rules),
        "cases_sha256": 文件摘要(args.cases),
        "generation": {
            "thinking_enabled": False,
            "temperature": 0.0,
            "max_new_tokens_per_turn": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "summary": summary,
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
