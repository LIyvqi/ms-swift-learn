"""让真实模型与规则环境逐轮交互，并汇总端到端任务指标。"""

from __future__ import annotations

import argparse
import hashlib
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

知识模块 = import_module("course.25_agent_r1_news.knowledge_pipeline")
环境模块 = import_module("course.25_agent_r1_news.agent_system")
RuleKnowledgeBase = 知识模块.RuleKnowledgeBase
NewsPolicyEnvironment = 环境模块.NewsPolicyEnvironment


def 计算宏平均_f1(真实标签: list[str], 预测标签: list[str]) -> float:
    """按类别计算 F1 后取算术平均，未正常 finish 的预测自然计为漏判。"""

    类别 = sorted(set(真实标签))
    if not 类别:
        return 0.0
    各类_f1 = []
    for label in 类别:
        tp = sum(
            gold == label and prediction == label
            for gold, prediction in zip(真实标签, 预测标签)
        )
        fp = sum(
            gold != label and prediction == label
            for gold, prediction in zip(真实标签, 预测标签)
        )
        fn = sum(
            gold == label and prediction != label
            for gold, prediction in zip(真实标签, 预测标签)
        )
        denominator = 2 * tp + fp + fn
        各类_f1.append(2 * tp / denominator if denominator else 0.0)
    return mean(各类_f1)


def 读取_jsonl(
    path: Path, maximum_samples: int, sample_offset: int = 0
) -> list[dict[str, Any]]:
    """按固定偏移读取连续样本，便于把检查点选择集与最终留出集隔离。"""

    rows = []
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if seen < sample_offset:
                seen += 1
                continue
            rows.append(json.loads(line))
            if maximum_samples > 0 and len(rows) >= maximum_samples:
                break
    return rows


def 文件_sha256(path: Path) -> str:
    """记录评测输入的内容摘要，防止同名数据被静默替换。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="评测真实 Agent-R1 新闻模型")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=项目根目录 / "models/Qwen3.5-0.8B-Base",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=项目根目录 / "datasets/agent_r1_news/rl_smoke.jsonl",
    )
    parser.add_argument("--maximum-samples", type=int, default=12)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="跳过数据集开头的非空记录数，用于构造独立留出集",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=项目根目录 / "outputs/25_agent_r1_news/model_evaluation.json",
    )
    args = parser.parse_args()
    if args.maximum_samples < 0:
        parser.error("--maximum-samples 不能小于 0；0 表示读取偏移后的全部样本")
    if args.sample_offset < 0:
        parser.error("--sample-offset 不能小于 0")

    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine

    engine = TransformersEngine(
        str(args.model),
        adapters=[str(args.adapter)],
        torch_dtype=torch.bfloat16,
        attn_impl="eager",
        device_map="cuda:0",
        max_batch_size=args.batch_size,
    )
    generation = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0)
    knowledge = RuleKnowledgeBase.from_jsonl(
        项目根目录 / "datasets/agent_r1_news/knowledge_rules.jsonl"
    )

    rows = 读取_jsonl(args.dataset, args.maximum_samples, args.sample_offset)
    states = []
    for row in rows:
        env = NewsPolicyEnvironment(knowledge, row["env_config"])
        observation, _, system_prompt = env.reset()
        states.append(
            {
                "row": row,
                "env": env,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": observation},
                ],
                "last_info": {},
                "total_reward": 0.0,
            }
        )

    # 每一轮把多个独立环境的当前消息一次性送入模型，环境状态仍逐条更新。
    active = list(states)
    while active:
        for start in range(0, len(active), args.batch_size):
            current = active[start : start + args.batch_size]
            requests = [InferRequest(messages=state["messages"]) for state in current]
            responses = engine.infer(requests, request_config=generation)
            for state, response in zip(current, responses):
                completion = response.choices[0].message.content or ""
                state["messages"].append({"role": "assistant", "content": completion})
                next_observation, reward, done, info = state["env"].step(completion)
                state["total_reward"] += reward
                state["last_info"] = info
                if not done:
                    state["messages"].append(
                        {"role": "user", "content": next_observation}
                    )
        active = [state for state in active if not state["env"].done]

    traces = []
    metrics_by_task: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    真实标签 = []
    预测标签 = []
    按类正确: dict[str, list[float]] = defaultdict(list)
    总动作数 = 0
    无效动作数 = 0
    完成数 = 0
    思考分数 = []
    for state in states:
        row = state["row"]
        last_info = state["last_info"]
        task = row["task"]
        for name, value in last_info.get("metrics", {}).items():
            metrics_by_task[task][name].append(float(value))
        trace = last_info.get("trace", [])
        总动作数 += len(trace)
        无效动作数 += sum(
            str(step.get("event", "")).startswith("invalid") for step in trace
        )
        思考分数.extend(float(step.get("thinking_score", 0.0)) for step in trace)
        完成数 += int(bool(trace) and trace[-1].get("event") == "finish")
        if task == "decision":
            gold = str(row["label"])
            prediction = str(last_info.get("final", {}).get("decision", "__未完成__"))
            真实标签.append(gold)
            预测标签.append(prediction)
            按类正确[gold].append(float(gold == prediction))
        traces.append(
            {
                "record_id": row["record_id"],
                "task": task,
                "total_reward": state["total_reward"],
                "messages": state["messages"],
                "trace": trace,
                "metrics": last_info.get("metrics", {}),
            }
        )

    summary = {
        task: {name: mean(values) for name, values in task_metrics.items() if values}
        for task, task_metrics in metrics_by_task.items()
    }
    agent_summary = {
        "completion_rate": 完成数 / len(states) if states else 0.0,
        "invalid_action_rate": 无效动作数 / 总动作数 if 总动作数 else 0.0,
        "thinking_presence_rate": mean(思考分数) if 思考分数 else 0.0,
        "mean_turns": mean(len(item["trace"]) for item in traces) if traces else 0.0,
        "decision_macro_f1": 计算宏平均_f1(真实标签, 预测标签),
        "decision_accuracy_by_label": {
            label: mean(values) for label, values in sorted(按类正确.items())
        },
    }
    result = {
        "adapter": str(args.adapter),
        "model": str(args.model),
        "dataset": str(args.dataset),
        "samples": len(traces),
        "evaluation_config": {
            "sample_offset": args.sample_offset,
            "maximum_samples": args.maximum_samples,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "dataset_sha256": 文件_sha256(args.dataset),
            "knowledge_sha256": 文件_sha256(
                项目根目录 / "datasets/agent_r1_news/knowledge_rules.jsonl"
            ),
            "sample_sequence_sha256": hashlib.sha256(
                "\n".join(f"{row['record_id']}\t{row['task']}" for row in rows).encode(
                    "utf-8"
                )
            ).hexdigest(),
        },
        "summary": summary,
        "agent_summary": agent_summary,
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "samples": len(traces),
                "summary": summary,
                "agent_summary": agent_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
