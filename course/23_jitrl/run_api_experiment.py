#!/usr/bin/env python3
"""通过已部署的 OpenAI 兼容模型 API 运行 JitRL 对照实验。"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from api_policy import OpenAI动作策略, 支持的打分模式
from experiment_common import 汇总设置, 运行单组
from jitrl_core import 经验记忆
from protocol_env import 所有决策状态, 隐式协议环境

ROOT = Path(__file__).resolve().parents[2]


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用推理 API 的 JitRL 持续学习实验")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="JITRL_API_KEY")
    parser.add_argument("--api-model", default="Qwen3.5-0.8B-Base")
    parser.add_argument("--score-mode", choices=支持的打分模式, default="constrained_logprobs")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--betas", default="2,4,8")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--unseen-probability", type=float, default=0.05)
    parser.add_argument("--optimism-alpha", type=float, default=5.0)
    parser.add_argument("--task-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/23_jitrl_api")
    parser.add_argument("--resume-memory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = 解析参数()
    if args.episodes <= 0 or args.concurrency <= 0:
        raise ValueError("episodes 与 concurrency 必须大于 0")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    betas = [float(value) for value in args.betas.split(",") if value.strip()]
    if not seeds or not betas or any(beta <= 0 for beta in betas):
        raise ValueError("seeds 不能为空，betas 必须全部大于 0")
    if args.resume_memory and len(seeds) != 1:
        raise ValueError("恢复外部记忆时只允许设置一个 seed，以免语义不明确")

    api_key = os.environ.get(args.api_key_env, "EMPTY")
    policy = OpenAI动作策略(
        base_url=args.base_url,
        api_key=api_key,
        model=args.api_model,
        score_mode=args.score_mode,
        top_logprobs=args.top_logprobs,
        timeout=args.request_timeout,
    )
    available_models = policy.检查服务()
    if available_models and args.api_model not in available_models:
        raise RuntimeError(f"API 模型 {args.api_model!r} 不在服务列表中：{available_models}")

    started = time.perf_counter()
    all_states = 所有决策状态()
    print(
        f"通过 API 并发计算 {len(all_states)} 个状态，模式={args.score_mode}，"
        f"并发数={args.concurrency}……",
        flush=True,
    )
    logits_cache = policy.批量打分(all_states, args.concurrency)
    logit_ranges = [max(values) - min(values) for values in logits_cache.values()]

    output_run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_directory = args.output_dir / output_run_name
    initial_memory = 经验记忆.加载(args.resume_memory) if args.resume_memory else None
    groups = []
    for beta in [0.0, *betas]:
        setting_runs = []
        for seed in seeds:
            print(f"运行 beta={beta:g}，seed={seed}……", flush=True)
            run, memory = 运行单组(
                logits_cache,
                beta=beta,
                seed=seed,
                args=args,
                initial_memory=initial_memory if beta > 0 else None,
            )
            setting_runs.append(run)
            if beta > 0:
                memory.保存(run_directory / f"memory_beta{beta:g}_seed{seed}.jsonl")
        groups.append({"summary": 汇总设置(setting_runs), "runs": setting_runs})

    task_environment = 隐式协议环境(args.task_seed, seeds[0])
    result = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "api_model": args.api_model,
            "score_mode": args.score_mode,
            "available_models": available_models,
            "client_creates_optimizer": False,
            "client_calls_backward": False,
            "server_parameter_unchanged": "远程 API 客户端不可观察，需由部署端保证模型版本固定",
            "elapsed_seconds": time.perf_counter() - started,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "api_key_env"
        },
        "environment": {
            "correct_actions": task_environment.correct_actions,
            "num_api_scored_states": len(all_states),
        },
        "base_logit_statistics": {
            "mean_candidate_range": sum(logit_ranges) / len(logit_ranges),
            "max_candidate_range": max(logit_ranges),
            "min_candidate_range": min(logit_ranges),
        },
        "settings": groups,
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    result_path = run_directory / "result.json"
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    result_path.write_text(serialized, encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("latest_result.json").write_text(serialized, encoding="utf-8")

    print("\nAPI 实验汇总：")
    for group in groups:
        summary = group["summary"]
        name = "静态基线" if summary["beta"] == 0 else f"JitRL beta={summary['beta']:g}"
        print(
            f"{name:>16} | 总成功率 {summary['success_rate_mean']:.3f} | "
            f"前10局 {summary['first_10_success_rate_mean']:.3f} | "
            f"后10局 {summary['last_10_success_rate_mean']:.3f}"
        )
    print(f"RESULT_PATH={result_path}")


if __name__ == "__main__":
    main()
