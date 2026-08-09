#!/usr/bin/env python3
"""用冻结的 Qwen3.5-0.8B-Base 运行静态策略与 JitRL 对照实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from experiment_common import 汇总设置, 运行单组
from jitrl_core import 经验记忆
from protocol_env import 所有决策状态, 构造提示, 隐式协议环境
from swift import get_model_processor

ROOT = Path(__file__).resolve().parents[2]


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JitRL 推理期持续学习教学实验")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-0.8B-Base")
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
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/23_jitrl")
    parser.add_argument("--resume-memory", type=Path)
    return parser.parse_args()


def 参数指纹(model) -> str:
    """抽取每个参数的首、中、尾值，配合版本号检测意外原地更新。"""

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(f"{name}|{tuple(parameter.shape)}|{parameter.dtype}".encode())
        flat = parameter.detach().reshape(-1)
        if flat.numel():
            indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
            values = flat[indices].float().cpu().tolist()
            digest.update(repr(values).encode())
    return digest.hexdigest()


def 参数版本(model) -> dict[str, int]:
    """PyTorch 原地修改参数时会递增版本号。"""

    return {name: parameter._version for name, parameter in model.named_parameters()}


class 冻结动作策略:
    """通过 ms-swift 加载模型，并读取单 token 候选编号的真实 logits。"""

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
        """一次预计算有限环境的全部基础 logits，后续实验不再重复前向。"""

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


def main() -> None:
    args = 解析参数()
    if args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("episodes 与 batch-size 必须大于 0")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    betas = [float(value) for value in args.betas.split(",") if value.strip()]
    if not seeds or not betas or any(beta <= 0 for beta in betas):
        raise ValueError("seeds 不能为空，betas 必须全部大于 0")
    if args.resume_memory and len(seeds) != 1:
        raise ValueError("恢复外部记忆时只允许设置一个 seed，以免语义不明确")

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    policy = 冻结动作策略(args.model)
    trainable_parameters = sum(parameter.numel() for parameter in policy.model.parameters() if parameter.requires_grad)
    versions_before = 参数版本(policy.model)
    fingerprint_before = 参数指纹(policy.model)

    all_states = 所有决策状态()
    print(f"正在批量计算 {len(all_states)} 个有限决策状态的真实基础 logits……", flush=True)
    logits_cache = policy.批量打分(all_states, args.batch_size)
    logit_ranges = [max(values) - min(values) for values in logits_cache.values()]

    initial_memory = 经验记忆.加载(args.resume_memory) if args.resume_memory else None
    run_groups = []
    output_run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_directory = args.output_dir / output_run_name
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
        run_groups.append({"summary": 汇总设置(setting_runs), "runs": setting_runs})

    versions_after = 参数版本(policy.model)
    fingerprint_after = 参数指纹(policy.model)
    parameter_unchanged = versions_before == versions_after and fingerprint_before == fingerprint_after
    if not parameter_unchanged:
        raise RuntimeError("检测到模型参数发生变化，实验不再满足 JitRL 前提")

    task_environment = 隐式协议环境(args.task_seed, seeds[0])
    result = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": str(args.model.resolve()),
            "swift_loader": "swift.get_model_processor",
            "torch_version": torch.__version__,
            "device": str(policy.device),
            "trainable_parameters": trainable_parameters,
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
            "actions": list(next(iter(all_states)).candidates),
            "correct_actions": task_environment.correct_actions,
            "num_precomputed_states": len(all_states),
        },
        "base_logit_statistics": {
            "mean_candidate_range": sum(logit_ranges) / len(logit_ranges),
            "max_candidate_range": max(logit_ranges),
            "min_candidate_range": min(logit_ranges),
        },
        "settings": run_groups,
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    result_path = run_directory / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = args.output_dir / "latest_result.json"
    latest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n实验汇总：")
    for group in run_groups:
        summary = group["summary"]
        name = "静态基线" if summary["beta"] == 0 else f"JitRL beta={summary['beta']:g}"
        print(
            f"{name:>16} | 总成功率 {summary['success_rate_mean']:.3f} | "
            f"前10局 {summary['first_10_success_rate_mean']:.3f} | "
            f"后10局 {summary['last_10_success_rate_mean']:.3f} | "
            f"平均奖励 {summary['mean_reward']:.3f}"
        )
    print(f"PARAMETER_UNCHANGED={parameter_unchanged}")
    print(f"RESULT_PATH={result_path}")


if __name__ == "__main__":
    main()
