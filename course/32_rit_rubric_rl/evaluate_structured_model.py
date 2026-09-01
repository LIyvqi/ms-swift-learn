#!/usr/bin/env python3
"""真实生成并评价无自由思维链的短结构化审核模型。"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))
核心 = importlib.import_module("course.32_rit_rubric_rl.rit_core")
结构化 = importlib.import_module("course.32_rit_rubric_rl.structured_core")


def 读取(path: Path, maximum: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if maximum > 0 and len(rows) >= maximum:
                break
    return rows


def 汇总多标签(traces: list[dict[str, Any]]) -> dict[str, float]:
    """汇总 14 类的 macro-F1 与样本池化 micro-F1。"""

    per_category = []
    micro_tp = micro_fp = micro_fn = 0
    for category in 核心.允许类别:
        tp = sum(
            category in trace["predicted_categories"]
            and category in trace["gold_categories"]
            for trace in traces
        )
        fp = sum(
            category in trace["predicted_categories"]
            and category not in trace["gold_categories"]
            for trace in traces
        )
        fn = sum(
            category not in trace["predicted_categories"]
            and category in trace["gold_categories"]
            for trace in traces
        )
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_category.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {"category_macro_f1": mean(per_category), "category_micro_f1": micro_f1}


def 主程序() -> None:
    parser = argparse.ArgumentParser(description="评测短结构化 RiT 内容审核模型")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=项目根目录 / "models/Qwen3.5-0.8B-Base",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=项目根目录 / "datasets/rit_audit/structured_rl_test.jsonl",
    )
    parser.add_argument("--maximum-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--gate", choices=("min", "none", "max", "conditional"), default="min"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=项目根目录
        / "outputs/32_rit_rubric_rl/structured_model_evaluation.json",
    )
    args = parser.parse_args()

    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine

    rows = 读取(args.dataset, args.maximum_samples)
    engine = TransformersEngine(
        str(args.model),
        adapters=[str(args.adapter)] if args.adapter else None,
        torch_dtype=torch.bfloat16,
        attn_impl="eager",
        device_map="cuda:0",
        max_batch_size=args.batch_size,
    )
    config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    completions = []
    began = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        requests = [
            InferRequest(
                messages=row["messages"],
                chat_template_kwargs={"enable_thinking": False},
            )
            for row in chunk
        ]
        responses = engine.infer(requests, request_config=config, use_tqdm=False)
        completions.extend(
            str(response.choices[0].message.content or "") for response in responses
        )
    elapsed = time.perf_counter() - began
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    traces = []
    rubric_totals: dict[str, list[float]] = defaultdict(list)
    for row, completion in zip(rows, completions):
        parsed = 结构化.解析结构化回答(completion)
        detail = 结构化.计算结构化RiT奖励(
            completion,
            row["prompt_text"],
            row["response_text"],
            row["gold_is_safe"],
            row["gold_categories"],
            alpha=args.alpha,
            gate=args.gate,
        )
        expected_safe = 核心.解析布尔(row["gold_is_safe"])
        expected_categories = 核心.解包类别(row["gold_categories"])
        for name, score in detail.rubric_scores.items():
            rubric_totals[name].append(score)
        traces.append(
            {
                "record_id": row["record_id"],
                "gold_is_safe": expected_safe,
                "gold_categories": list(expected_categories),
                "predicted_is_safe": parsed.安全,
                "predicted_categories": list(parsed.类别),
                "format_valid": parsed.格式有效,
                "exact_correct": detail.响应奖励,
                "safety_correct": float(parsed.安全 == expected_safe),
                "sample_category_f1": 核心.样本多标签_f1(
                    parsed.类别, expected_categories
                ),
                "structure_score": detail.思考奖励,
                "gated_score": detail.最终奖励,
                "completion_chars": len(completion),
                "rubric_scores": detail.rubric_scores,
                "completion": completion,
            }
        )

    summary = {
        "samples": len(traces),
        "exact_accuracy": mean(row["exact_correct"] for row in traces)
        if traces
        else 0.0,
        "safety_accuracy": mean(row["safety_correct"] for row in traces)
        if traces
        else 0.0,
        "mean_sample_category_f1": mean(
            row["sample_category_f1"] for row in traces
        )
        if traces
        else 0.0,
        **汇总多标签(traces),
        "format_valid_rate": mean(row["format_valid"] for row in traces)
        if traces
        else 0.0,
        "mean_structure_score": mean(row["structure_score"] for row in traces)
        if traces
        else 0.0,
        "mean_gated_score": mean(row["gated_score"] for row in traces)
        if traces
        else 0.0,
        "mean_completion_chars": mean(row["completion_chars"] for row in traces)
        if traces
        else 0.0,
        "rubric_pass_rates": {
            name: mean(values) for name, values in sorted(rubric_totals.items())
        },
        "elapsed_seconds": elapsed,
        "samples_per_second": len(traces) / elapsed if elapsed else 0.0,
    }
    result = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "generation": {
            "thinking_enabled": False,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "reward": {"alpha": args.alpha, "gate": args.gate},
        "summary": summary,
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
