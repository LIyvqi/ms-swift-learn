#!/usr/bin/env python3
"""本地权重与 API 实验共用的 JitRL Agent 交互循环。"""

from __future__ import annotations

import argparse
import random

from jitrl_core import softmax, 修正动作_logits, 按概率采样, 经验记忆
from protocol_env import 隐式协议环境


def 运行单组(
    logits_cache: dict[tuple, list[float]],
    *,
    beta: float,
    seed: int,
    args: argparse.Namespace,
    initial_memory: 经验记忆 | None = None,
) -> tuple[dict, 经验记忆]:
    """固定基础策略和任务，只改变 JitRL 强度运行一组连续回合。"""

    environment = 隐式协议环境(args.task_seed, seed)
    policy_rng = random.Random(seed + 1_000_003)
    jitrl_rng = random.Random(seed + 2_000_003)
    memory = 经验记忆(initial_memory.entries if initial_memory else None)
    episodes = []

    for episode in range(args.episodes):
        state = environment.reset(episode)
        states: list[str] = []
        actions: list[str] = []
        rewards: list[float] = []
        corrections = []
        success = False
        while state is not None:
            key = (state.phase_index, state.batch_tag, state.candidates)
            base_logits = logits_cache[key]
            details = 修正动作_logits(
                state.retrieval_state,
                state.candidates,
                base_logits,
                memory,
                jitrl_rng,
                beta=beta,
                top_k=args.top_k,
                similarity_threshold=args.similarity_threshold,
                unseen_probability=args.unseen_probability,
                optimism_alpha=args.optimism_alpha,
            )
            corrected_in_order = [details.corrected_logits[action] for action in state.candidates]
            probabilities = softmax(corrected_in_order, args.temperature)
            action = 按概率采样(state.candidates, probabilities, policy_rng)
            states.append(state.retrieval_state)
            actions.append(action)
            next_state, reward, done, info = environment.step(action)
            rewards.append(reward)
            corrections.append({
                "phase": state.phase_name,
                "action": action,
                "neighbor_count": details.neighbor_count,
                "normalized_advantages": details.normalized_advantages,
                "probabilities": dict(zip(state.candidates, probabilities)),
            })
            success = bool(info["success"])
            state = next_state
            if done:
                break

        # beta=0 的静态策略不读取也不写入经验，保证它是真正的冻结基线。
        if beta > 0:
            memory.添加轨迹(states, actions, rewards, args.gamma, episode)
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
    first_success = next((index for index, value in enumerate(successes) if value), None)
    summary = {
        "beta": beta,
        "seed": seed,
        "success_rate": sum(successes) / len(successes),
        "first_10_success_rate": sum(successes[:window]) / window,
        "last_10_success_rate": sum(successes[-window:]) / window,
        "first_success_episode": first_success,
        "mean_steps": sum(item["steps"] for item in episodes) / len(episodes),
        "mean_reward": sum(sum(item["rewards"]) for item in episodes) / len(episodes),
        "memory_entries": len(memory),
        "episodes": episodes,
    }
    return summary, memory


def 汇总设置(runs: list[dict]) -> dict:
    """汇总同一个 beta 的多随机种子结果。"""

    def mean(key: str) -> float:
        return sum(run[key] for run in runs) / len(runs)

    return {
        "beta": runs[0]["beta"],
        "num_seeds": len(runs),
        "success_rate_mean": mean("success_rate"),
        "first_10_success_rate_mean": mean("first_10_success_rate"),
        "last_10_success_rate_mean": mean("last_10_success_rate"),
        "mean_reward": mean("mean_reward"),
        "mean_steps": mean("mean_steps"),
    }
