#!/usr/bin/env python3
"""通过 OpenAI 兼容 API 运行 KCR-JitRL 三库协同消融实验。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from experiment_common import 汇总设置, 运行单组, 默认实验设置
from kcr_core import 支持库, 规则库

# 共用实验模块已把第 23 课加入模块路径。动态导入可避免格式化工具
# 把第 23 课适配器提前到路径初始化之前。
api_policy_module = importlib.import_module("api_policy")
protocol_env_module = importlib.import_module("protocol_env")
OpenAI动作策略 = api_policy_module.OpenAI动作策略
支持的打分模式 = api_policy_module.支持的打分模式
所有决策状态 = protocol_env_module.所有决策状态
隐式协议环境 = protocol_env_module.隐式协议环境

ROOT = Path(__file__).resolve().parents[2]
本课目录 = Path(__file__).resolve().parent


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用推理 API 的 KCR-JitRL 实验")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="KCR_JITRL_API_KEY")
    parser.add_argument("--api-model", default="Qwen3.5-0.8B-Base")
    parser.add_argument("--score-mode", choices=支持的打分模式, default="constrained_logprobs")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--support-library", type=Path, default=本课目录 / "data/support_library.jsonl")
    parser.add_argument("--rule-library", type=Path, default=本课目录 / "data/rule_library.jsonl")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta-case", type=float, default=8.0)
    parser.add_argument("--beta-support", type=float, default=3.0)
    parser.add_argument("--beta-rule", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--case-threshold", type=float, default=0.95)
    parser.add_argument("--support-threshold", type=float, default=0.95)
    parser.add_argument("--rule-threshold", type=float, default=0.95)
    parser.add_argument("--min-confidence-samples", type=int, default=4)
    parser.add_argument("--unseen-probability", type=float, default=0.05)
    parser.add_argument("--optimism-alpha", type=float, default=5.0)
    parser.add_argument("--condense-min-evidence", type=int, default=6)
    parser.add_argument("--condense-margin", type=float, default=0.5)
    parser.add_argument("--task-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/24_kcr_jitrl_api")
    return parser.parse_args()


def main() -> None:
    args = 解析参数()
    if args.episodes <= 0 or args.concurrency <= 0:
        raise ValueError("episodes 与 concurrency 必须大于 0")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("seeds 不能为空")

    api_key = os.environ.get(args.api_key_env, "EMPTY")
    policy = OpenAI动作策略(
        base_url=args.base_url,
        api_key=api_key,
        model=args.api_model,
        score_mode=args.score_mode,
        top_logprobs=args.top_logprobs,
        timeout=args.request_timeout,
    )
    available_models = [] if args.skip_model_check else policy.检查服务()
    if available_models and args.api_model not in available_models:
        raise RuntimeError(f"API 模型 {args.api_model!r} 不在服务列表中：{available_models}")

    all_states = 所有决策状态()
    if args.probe_only:
        scores = policy.打分(all_states[0])
        print(f"API_PROBE=PASS，候选对数分数={scores}")
        return

    started = time.perf_counter()
    print(
        f"通过 API 计算 {len(all_states)} 个状态，模式={args.score_mode}，并发={args.concurrency}……",
        flush=True,
    )
    logits_cache = policy.批量打分(all_states, args.concurrency)
    support = 支持库.加载(args.support_library)
    base_rules = 规则库.加载(args.rule_library)
    run_directory = args.output_dir / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    groups = []
    for setting in 默认实验设置():
        setting_runs = []
        for seed in seeds:
            print(f"运行 setting={setting.name}，seed={seed}……", flush=True)
            run, cases, rules = 运行单组(
                logits_cache,
                support,
                base_rules,
                setting,
                seed=seed,
                args=args,
            )
            setting_runs.append(run)
            if setting.enable_cases:
                cases.保存(run_directory / f"cases_{setting.name}_seed{seed}.jsonl")
            if setting.enable_rules:
                rules.保存(run_directory / f"rules_{setting.name}_seed{seed}.jsonl")
        groups.append({"summary": 汇总设置(setting_runs), "runs": setting_runs})

    environment = 隐式协议环境(args.task_seed, seeds[0])
    ranges = [max(values) - min(values) for values in logits_cache.values()]
    result = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "api_model": args.api_model,
            "score_mode": args.score_mode,
            "available_models": available_models,
            "client_creates_optimizer": False,
            "client_calls_backward": False,
            "server_parameter_unchanged": "远程客户端不可观察，由服务端保证模型版本固定",
            "elapsed_seconds": time.perf_counter() - started,
        },
        # 密钥本身及密钥环境变量名都不写进结果文件。
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "api_key_env"
        },
        "environment": {
            "correct_actions": environment.correct_actions,
            "num_api_scored_states": len(all_states),
            "reliable_support_coverage": ["入口校验"],
            "low_confidence_noise_probe": ["货物扫描错误口述记录"],
            "manual_rule_coverage": ["升降平台", "入口校验硬禁令"],
            "uncovered_by_correct_prior": ["货物扫描", "出口放行"],
        },
        "base_logit_statistics": {
            "mean_candidate_range": sum(ranges) / len(ranges),
            "max_candidate_range": max(ranges),
            "min_candidate_range": min(ranges),
        },
        "settings": groups,
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    result_path = run_directory / "result.json"
    result_path.write_text(serialized, encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("latest_result.json").write_text(serialized, encoding="utf-8")

    print("\nKCR-JitRL API 实验汇总：")
    for group in groups:
        summary = group["summary"]
        print(
            f"{summary['setting']:>18} | 总成功率 {summary['success_rate_mean']:.3f} | "
            f"前10局 {summary['first_10_success_rate_mean']:.3f} | "
            f"后10局 {summary['last_10_success_rate_mean']:.3f} | "
            f"浓缩规则 {summary['condensed_rule_count_mean']:.1f}"
        )
    print(f"RESULT_PATH={result_path}")


if __name__ == "__main__":
    main()
