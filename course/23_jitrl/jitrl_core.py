#!/usr/bin/env python3
"""JitRL 的经验记忆、非参数价值估计与 logits 修正规则。"""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class 经验:
    """一条带折扣回报的状态—动作经验。"""

    state: str
    action: str
    return_value: float
    episode: int
    step: int


@dataclass(frozen=True)
class 修正详情:
    """一次 JitRL 修正的中间量，便于教学和排错。"""

    neighbor_count: int
    state_value: float
    action_values: dict[str, float]
    advantages: dict[str, float]
    normalized_advantages: dict[str, float]
    base_logits: dict[str, float]
    corrected_logits: dict[str, float]


def 状态词元(text: str) -> set[str]:
    """把中英文状态拆成集合；中文按单字切分以避免依赖分词器。"""

    return set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower()))


def 文本杰卡德(left: str, right: str) -> float:
    """计算两个状态词元集合的 Jaccard 相似度。"""

    left_tokens = 状态词元(left)
    right_tokens = 状态词元(right)
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def 折扣回报(rewards: Sequence[float], gamma: float) -> list[float]:
    """从后向前计算每一步的 Monte Carlo 折扣回报。"""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma 必须位于 [0, 1]")
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


class 经验记忆:
    """使用文本相似度检索的可持久化非参数经验记忆。"""

    def __init__(self, entries: Iterable[经验] | None = None):
        self.entries = list(entries or [])

    def __len__(self) -> int:
        return len(self.entries)

    def 添加轨迹(
        self,
        states: Sequence[str],
        actions: Sequence[str],
        rewards: Sequence[float],
        gamma: float,
        episode: int,
    ) -> None:
        """在一局结束后计算回报，并把该局每一步写入记忆。"""

        if not (len(states) == len(actions) == len(rewards)):
            raise ValueError("states、actions 与 rewards 长度必须一致")
        returns = 折扣回报(rewards, gamma)
        self.entries.extend(
            经验(state, action, return_value, episode, step)
            for step, (state, action, return_value) in enumerate(zip(states, actions, returns))
        )

    def 检索(self, state: str, top_k: int, similarity_threshold: float) -> list[经验]:
        """返回高于阈值的最相似经验；相似度并列时优先最近经验。"""

        ranked = []
        for index, entry in enumerate(self.entries):
            similarity = 文本杰卡德(state, entry.state)
            if similarity >= similarity_threshold:
                ranked.append((similarity, index, entry))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked[:top_k]]

    def 保存(self, path: Path) -> None:
        """按通用 JSONL 格式保存经验，便于跨进程持续学习。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    @classmethod
    def 加载(cls, path: Path) -> 经验记忆:
        """从 JSONL 恢复经验；文件不存在时返回空记忆。"""

        if not path.exists():
            return cls()
        entries = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    entries.append(经验(**row))
                except TypeError as error:
                    raise ValueError(f"经验文件第 {line_number} 行字段错误：{error}") from error
        return cls(entries)


def 修正动作_logits(
    state: str,
    actions: Sequence[str],
    base_logits: Sequence[float],
    memory: 经验记忆,
    rng: random.Random,
    *,
    beta: float,
    top_k: int,
    similarity_threshold: float,
    unseen_probability: float,
    optimism_alpha: float,
    epsilon: float = 1e-8,
) -> 修正详情:
    """按论文闭式解 ``z'=z+beta*A_norm`` 修正当前动作 logits。"""

    if len(actions) != len(base_logits):
        raise ValueError("动作数与基础 logits 数量必须一致")
    if len(set(actions)) != len(actions):
        raise ValueError("候选动作不能重复")
    if not 0.0 <= unseen_probability <= 1.0:
        raise ValueError("unseen_probability 必须位于 [0, 1]")

    neighbors = memory.检索(state, top_k, similarity_threshold)
    base = {action: float(logit) for action, logit in zip(actions, base_logits)}
    if not neighbors:
        zeros = {action: 0.0 for action in actions}
        return 修正详情(0, 0.0, zeros, zeros, zeros, base, dict(base))

    state_value = sum(entry.return_value for entry in neighbors) / len(neighbors)
    grouped_returns: dict[str, list[float]] = defaultdict(list)
    for entry in neighbors:
        grouped_returns[entry.action].append(entry.return_value)

    action_values: dict[str, float] = {}
    for action in actions:
        observed = grouped_returns.get(action)
        if observed:
            action_values[action] = sum(observed) / len(observed)
        elif rng.random() < unseen_probability:
            action_values[action] = state_value + optimism_alpha / len(neighbors)
        else:
            action_values[action] = 0.0

    advantages = {
        action: action_values[action] - state_value
        for action in actions
    }
    scale = max(max(abs(value) for value in advantages.values()), epsilon)
    normalized = {action: value / scale for action, value in advantages.items()}
    corrected = {
        action: base[action] + beta * normalized[action]
        for action in actions
    }
    return 修正详情(
        len(neighbors),
        state_value,
        action_values,
        advantages,
        normalized,
        base,
        corrected,
    )


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """稳定计算一维 softmax。"""

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    scaled = [float(value) / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def 按概率采样(actions: Sequence[str], probabilities: Sequence[float], rng: random.Random) -> str:
    """使用 Python 随机数生成器进行可复现采样。"""

    draw = rng.random()
    cumulative = 0.0
    for action, probability in zip(actions, probabilities):
        cumulative += probability
        if draw <= cumulative:
            return action
    return actions[-1]
