#!/usr/bin/env python3
"""使用冻结的 Qwen3.5-0.8B-Base 运行 KCR-JitRL 消融实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from experiment_common import 汇总设置, 运行单组, 默认实验设置
from kcr_core import 支持库, 规则库
from swift import get_model_processor

ROOT = Path(__file__).resolve().parents[2]
本课目录 = Path(__file__).resolve().parent
原始课程目录 = 本课目录.parent / "23_jitrl"
if str(原始课程目录) not in sys.path:
    sys.path.insert(0, str(原始课程目录))

from protocol_env import 所有决策状态, 构造提示, 隐式协议环境


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KCR-JitRL 三库协同推理期持续学习实验")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-0.8B-Base")
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
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/24_kcr_jitrl")
    return parser.parse_args()


def 参数指纹(model) -> str:
    """抽样计算参数指纹，检测实验期间是否发生模型更新。"""

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(f"{name}|{tuple(parameter.shape)}|{parameter.dtype}".encode())
        flat = parameter.detach().reshape(-1)
        if flat.numel():
            indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
            digest.update(repr(flat[indices].float().cpu().tolist()).encode())
    return digest.hexdigest()


def 参数版本(model) -> dict[str, int]:
    """记录 PyTorch 参数的原地修改版本号。"""

    return {name: parameter._version for name, parameter in model.named_parameters()}


class 冻结动作策略:
    """加载本地 Base 模型并读取候选编号的真实单 token logits。"""

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型目录：{model_path}")
        self.model, processor = get_model_processor(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_impl="eager",
        )
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.digit_ids = []
        for digit in ("1", "2", "3"):
            ids = self.tokenizer.encode(digit, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"候选编号 {digit!r} 不是单 token：{ids}")
            self.digit_ids.append(ids[0])

    @torch.inference_mode()
    def 批量打分(self, states, batch_size: int) -> dict[tuple, list[float]]:
        """一次预计算全部有限状态，所有消融组共享同一冻结策略。"""

        cache: dict[tuple, list[float]] = {}
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for start in range(0, len(states), batch_size):
                batch = states[start:start + batch_size]
                prompts = [构造提示(state) for state in batch]
                inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                try:
                    outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
                except TypeError:
                    outputs = self.model(**inputs, use_cache=False)
                digit_logits = outputs.logits[:, -1, self.digit_ids].float().cpu()
                for state, logits in zip(batch, digit_logits.tolist()):
                    key = (state.phase_index, state.batch_tag, state.candidates)
                    cache[key] = logits
                del outputs, inputs, digit_logits
        finally:
            self.tokenizer.padding_side = old_padding_side
        return cache


def 执行实验(args: argparse.Namespace, logits_cache: dict[tuple, list[float]]) -> tuple[list[dict], Path]:
    """运行全部消融设置并保存案例、浓缩规则和逐回合结果。"""

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
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
    return groups, run_directory


def main() -> None:
    args = 解析参数()
    if args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("episodes 与 batch-size 必须大于 0")
    if args.min_confidence_samples <= 0 or args.condense_min_evidence <= 0:
        raise ValueError("置信度样本数与规则浓缩证据数必须大于 0")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("seeds 不能为空")

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    policy = 冻结动作策略(args.model)
    versions_before = 参数版本(policy.model)
    fingerprint_before = 参数指纹(policy.model)
    all_states = 所有决策状态()
    print(f"正在批量计算 {len(all_states)} 个状态的真实基础 logits……", flush=True)
    logits_cache = policy.批量打分(all_states, args.batch_size)
    groups, run_directory = 执行实验(args, logits_cache)

    fingerprint_after = 参数指纹(policy.model)
    parameter_unchanged = (
        versions_before == 参数版本(policy.model)
        and fingerprint_before == fingerprint_after
    )
    if not parameter_unchanged:
        raise RuntimeError("检测到模型参数发生变化，实验不再满足无梯度前提")

    logit_ranges = [max(values) - min(values) for values in logits_cache.values()]
    environment = 隐式协议环境(args.task_seed, seeds[0])
    result = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": str(args.model.resolve()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in policy.model.parameters() if parameter.requires_grad
            ),
            "optimizer_created": False,
            "backward_called": False,
            "parameter_unchanged": parameter_unchanged,
            "parameter_fingerprint_before": fingerprint_before,
            "parameter_fingerprint_after": fingerprint_after,
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "correct_actions": environment.correct_actions,
            "num_precomputed_states": len(all_states),
            "reliable_support_coverage": ["入口校验"],
            "low_confidence_noise_probe": ["货物扫描错误口述记录"],
            "manual_rule_coverage": ["升降平台", "入口校验硬禁令"],
            "uncovered_by_correct_prior": ["货物扫描", "出口放行"],
        },
        "base_logit_statistics": {
            "mean_candidate_range": sum(logit_ranges) / len(logit_ranges),
            "max_candidate_range": max(logit_ranges),
            "min_candidate_range": min(logit_ranges),
        },
        "settings": groups,
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    result_path = run_directory / "result.json"
    result_path.write_text(serialized, encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("latest_result.json").write_text(serialized, encoding="utf-8")

    print("\nKCR-JitRL 实验汇总：")
    for group in groups:
        summary = group["summary"]
        print(
            f"{summary['setting']:>18} | 总成功率 {summary['success_rate_mean']:.3f} | "
            f"前10局 {summary['first_10_success_rate_mean']:.3f} | "
            f"后10局 {summary['last_10_success_rate_mean']:.3f} | "
            f"浓缩规则 {summary['condensed_rule_count_mean']:.1f}"
        )
    print(f"PARAMETER_UNCHANGED={parameter_unchanged}")
    print(f"RESULT_PATH={result_path}")


if __name__ == "__main__":
    main()
