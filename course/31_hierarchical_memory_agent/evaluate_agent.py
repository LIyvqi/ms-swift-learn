#!/usr/bin/env python3
"""让真实 Qwen LoRA 逐轮操作多源分层记忆环境并统计端到端指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path
from statistics import mean
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))
网关模块 = import_module("course.31_hierarchical_memory_agent.memory_gateway")
环境模块 = import_module("course.31_hierarchical_memory_agent.agent_environment")
分层记忆网关 = 网关模块.分层记忆网关
分层记忆审核环境 = 环境模块.分层记忆审核环境


def 读取_jsonl(path: Path, maximum: int) -> list[dict[str, Any]]:
    """读取固定顺序样本，零表示全部。"""

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if maximum > 0 and len(rows) >= maximum:
                break
    return rows


def 文件_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="评测真实分层记忆审核 Agent")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--model", type=Path, default=项目根目录 / "models/Qwen3.5-0.8B-Base"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=项目根目录 / "datasets/hierarchical_memory_audit/rl_test.jsonl",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=项目根目录 / "datasets/hierarchical_memory_audit/source_registry.json",
    )
    parser.add_argument("--maximum-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument(
        "--output",
        type=Path,
        default=项目根目录 / "outputs/31_hierarchical_memory_agent/model_evaluation.json",
    )
    args = parser.parse_args()

    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine

    adapter_list = [str(args.adapter)] if args.adapter else None
    engine = TransformersEngine(
        str(args.model),
        adapters=adapter_list,
        torch_dtype=torch.bfloat16,
        attn_impl="eager",
        device_map="cuda:0",
        max_batch_size=args.batch_size,
    )
    generation = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0)
    gateway = 分层记忆网关(args.registry)
    rows = 读取_jsonl(args.dataset, args.maximum_samples)
    states = []
    for row in rows:
        environment = 分层记忆审核环境(gateway, row["env_config"])
        observation, _, system_prompt = environment.reset()
        states.append(
            {
                "row": row,
                "environment": environment,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": observation},
                ],
                "last_info": {},
                "total_reward": 0.0,
            }
        )

    active = list(states)
    while active:
        for start in range(0, len(active), args.batch_size):
            current = active[start : start + args.batch_size]
            requests = [
                InferRequest(
                    messages=state["messages"],
                    chat_template_kwargs={"enable_thinking": True},
                )
                for state in current
            ]
            responses = engine.infer(requests, request_config=generation)
            for state, response in zip(current, responses):
                completion = response.choices[0].message.content or ""
                state["messages"].append({"role": "assistant", "content": completion})
                observation, reward, done, info = state["environment"].step(completion)
                state["last_info"] = info
                state["total_reward"] += reward
                if not done:
                    state["messages"].append({"role": "user", "content": observation})
        active = [state for state in active if not state["environment"].done]

    traces = []
    completed = []
    invalid_rates = []
    metric_values: dict[str, list[float]] = {}
    locate_counts = []
    search_counts = []
    for state in states:
        info = state["last_info"]
        trace = info.get("trace", [])
        completed.append(float(bool(trace) and trace[-1]["event"] == "finish"))
        invalid_rates.append(
            sum(str(step["event"]).startswith("invalid") for step in trace) / max(len(trace), 1)
        )
        locate_counts.append(sum(step["event"] == "locate" for step in trace))
        search_counts.append(sum(step["event"] == "search" for step in trace))
        for name, value in info.get("metrics", {}).items():
            metric_values.setdefault(name, []).append(float(value))
        traces.append(
            {
                "record_id": state["row"]["record_id"],
                "total_reward": state["total_reward"],
                "messages": state["messages"],
                "trace": trace,
                "metrics": info.get("metrics", {}),
                "final": info.get("final", {}),
            }
        )
    summary = {
        "completion_rate": mean(completed) if completed else 0.0,
        "invalid_action_rate": mean(invalid_rates) if invalid_rates else 0.0,
        "mean_turns": mean(len(row["trace"]) for row in traces) if traces else 0.0,
        "mean_locate_calls": mean(locate_counts) if locate_counts else 0.0,
        "mean_search_calls": mean(search_counts) if search_counts else 0.0,
        **{
            name: mean(values) if values else 0.0
            for name, values in sorted(metric_values.items())
        },
    }
    result = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "samples": len(rows),
        "dataset_sha256": 文件_sha256(args.dataset),
        "registry_sha256": 文件_sha256(args.registry),
        "summary": summary,
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"samples": len(rows), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
