#!/usr/bin/env python3
"""本地模型与远程 API 共用的 KCR-JitRL 多回合实验循环。"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from kcr_core import softmax, 修正动作_logits, 按概率采样, 支持库, 案例库, 规则库

本课目录 = Path(__file__).resolve().parent
原始课程目录 = 本课目录.parent / "23_jitrl"
if str(原始课程目录) not in sys.path:
    sys.path.insert(0, str(原始课程目录))

from protocol_env import 隐式协议环境


@dataclass(frozen=True)
class 实验设置:
    """一个可独立消融的 KCR-JitRL 策略设置。"""

    name: str
    enable_cases: bool
    enable_support: bool
    enable_rules: bool
    enable_condensation: bool
    use_confidence_gate: bool


def 默认实验设置() -> list[实验设置]:
    """返回覆盖三种来源、门控和规则浓缩的完整消融矩阵。"""

    return [
        实验设置("static", False, False, False, False, True),
        实验设置("case_only", True, False, False, False, True),
        实验设置("case_support", True, True, False, False, True),
        实验设置("case_rule", True, False, True, False, True),
        实验设置("kcr_no_condense", True, True, True, False, True),
        实验设置("kcr_no_gate", True, True, True, True, False),
        实验设置("kcr_full", True, True, True, True, True),
    ]


def 运行单组(
    logits_cache: dict[tuple, list[float]],
    support: 支持库,
    base_rules: 规则库,
    setting: 实验设置,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict, 案例库, 规则库]:
    """固定冻结策略和任务，只改变知识来源组合运行连续回合。"""

    environment = 隐式协议环境(args.task_seed, seed)
    policy_rng = random.Random(seed + 1_000_003)
    kcr_rng = random.Random(seed + 2_000_003)
    cases = 案例库()
    rules = base_rules.克隆()
    episodes = []
    condensed_updates = 0

    for episode in range(args.episodes):
        state = environment.reset(episode)
        states: list[str] = []
        actions: list[str] = []
        rewards: list[float] = []
        corrections = []
        success = False
        while state is not None:
            key = (state.phase_index, state.batch_tag, state.candidates)
            details = 修正动作_logits(
                state.retrieval_state,
                state.candidates,
                logits_cache[key],
                cases,
                support,
                rules,
                kcr_rng,
                enable_cases=setting.enable_cases,
                enable_support=setting.enable_support,
                enable_rules=setting.enable_rules,
                use_confidence_gate=setting.use_confidence_gate,
                beta_case=args.beta_case,
                beta_support=args.beta_support,
                beta_rule=args.beta_rule,
                top_k=args.top_k,
                case_threshold=args.case_threshold,
                support_threshold=args.support_threshold,
                rule_threshold=args.rule_threshold,
                unseen_probability=args.unseen_probability,
                optimism_alpha=args.optimism_alpha,
                min_confidence_samples=args.min_confidence_samples,
            )
            corrected = [details.corrected_logits[action] for action in state.candidates]
            probabilities = softmax(corrected, args.temperature)
            action = 按概率采样(state.candidates, probabilities, policy_rng)
            states.append(state.retrieval_state)
            actions.append(action)
            next_state, reward, done, info = environment.step(action)
            rewards.append(reward)
            corrections.append({
                "phase": state.phase_name,
                "action": action,
                "neighbor_count": details.neighbor_count,
                "case_confidence": details.case_signal.confidence,
                "support_confidence": details.support_signal.confidence,
                "rule_confidence": details.rule_signal.confidence,
                "support_ids": details.support_signal.matched_ids,
                "rule_ids": details.rule_signal.matched_ids,
                "hard_blocked_actions": details.hard_blocked_actions,
                "contributions": details.contributions,
                "probabilities": dict(zip(state.candidates, probabilities)),
            })
            success = bool(info["success"])
            state = next_state
            if done:
                break

        if setting.enable_cases:
            cases.添加轨迹(states, actions, rewards, args.gamma, episode)
        if setting.enable_condensation:
            condensed_updates += rules.从案例浓缩(
                cases,
                min_evidence=args.condense_min_evidence,
                margin=args.condense_margin,
            )
        episodes.append({
            "episode": episode,
            "success": success,
            "steps": len(actions),
            "reward_sum": sum(rewards),
            "actions": actions,
            "rewards": rewards,
            "corrections": corrections,
        })

    successes = [int(item["success"]) for item in episodes]
    window = min(10, len(episodes))
    condensed_rules = [entry for entry in rules.entries if entry.source == "案例浓缩"]
    summary = {
        "setting": setting.name,
        "seed": seed,
        "success_rate": sum(successes) / len(successes),
        "first_10_success_rate": sum(successes[:window]) / window,
        "last_10_success_rate": sum(successes[-window:]) / window,
        "first_success_episode": next((index for index, value in enumerate(successes) if value), None),
        "mean_steps": sum(item["steps"] for item in episodes) / len(episodes),
        "mean_reward": sum(item["reward_sum"] for item in episodes) / len(episodes),
        "case_entries": len(cases),
        "condensed_rule_count": len(condensed_rules),
        "condensed_rule_updates": condensed_updates,
        "setting_config": asdict(setting),
        "episodes": episodes,
    }
    return summary, cases, rules


def 汇总设置(runs: list[dict]) -> dict:
    """汇总同一个消融设置下的多随机种子结果。"""

    def mean(key: str) -> float:
        return sum(run[key] for run in runs) / len(runs)

    return {
        "setting": runs[0]["setting"],
        "num_seeds": len(runs),
        "success_rate_mean": mean("success_rate"),
        "first_10_success_rate_mean": mean("first_10_success_rate"),
        "last_10_success_rate_mean": mean("last_10_success_rate"),
        "mean_reward": mean("mean_reward"),
        "mean_steps": mean("mean_steps"),
        "condensed_rule_count_mean": mean("condensed_rule_count"),
    }
