#!/usr/bin/env python3
"""用于 JitRL 教学复现的多阶段隐式协议环境。"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass

动作全集 = ("琥珀协议", "靛蓝协议", "白银协议")
阶段全集 = ("入口校验", "升降平台", "货物扫描", "出口放行")
批次全集 = ("A17", "B04", "C29", "D11")


@dataclass(frozen=True)
class 状态:
    """模型可见状态、检索状态和候选动作。"""

    phase_index: int
    phase_name: str
    batch_tag: str
    observation: str
    retrieval_state: str
    candidates: tuple[str, ...]


class 隐式协议环境:
    """正确协议不写在提示里，只能通过跨回合奖励经验学到。"""

    def __init__(self, task_seed: int = 2026, episode_seed: int = 0):
        mapping_rng = random.Random(task_seed)
        # 四个阶段允许复用协议，使任务具有状态依赖而不是固定动作排序。
        self.correct_actions = {
            phase: mapping_rng.choice(动作全集)
            for phase in 阶段全集
        }
        if len(set(self.correct_actions.values())) < 2:
            self.correct_actions[阶段全集[-1]] = 动作全集[
                (动作全集.index(self.correct_actions[阶段全集[-1]]) + 1) % len(动作全集)
            ]
        self.rng = random.Random(episode_seed)
        self.episode = -1
        self.phase_index = 0
        self.done = False

    def reset(self, episode: int) -> 状态:
        self.episode = episode
        self.phase_index = 0
        self.done = False
        return self._state()

    def _state(self) -> 状态:
        phase = 阶段全集[self.phase_index]
        batch_tag = self.rng.choice(批次全集)
        candidates = list(动作全集)
        self.rng.shuffle(candidates)
        observation = (
            f"物流恢复任务；当前阶段={phase}；已完成阶段数={self.phase_index}；"
            f"设备批次={batch_tag}。系统不会公开正确协议，需要根据历史反馈选择。"
        )
        # 检索键去掉每回合变化的设备批次，但保留真正决定动作的阶段信息。
        retrieval_state = f"任务 物流恢复 阶段 {phase} 进度 {self.phase_index}"
        return 状态(
            self.phase_index,
            phase,
            batch_tag,
            observation,
            retrieval_state,
            tuple(candidates),
        )

    def step(self, action: str) -> tuple[状态 | None, float, bool, dict]:
        if self.done:
            raise RuntimeError("当前回合已经结束，请先调用 reset")
        phase = 阶段全集[self.phase_index]
        correct = action == self.correct_actions[phase]
        if not correct:
            self.done = True
            return None, -1.0, True, {"correct": False, "success": False, "phase": phase}

        self.phase_index += 1
        if self.phase_index == len(阶段全集):
            self.done = True
            return None, 2.0, True, {"correct": True, "success": True, "phase": phase}
        return self._state(), 0.5, False, {"correct": True, "success": False, "phase": phase}


def 所有决策状态() -> list[状态]:
    """枚举有限环境的全部提示，用一轮批处理预计算冻结策略 logits。"""

    states = []
    for phase_index, phase in enumerate(阶段全集):
        for batch_tag in 批次全集:
            observation = (
                f"物流恢复任务；当前阶段={phase}；已完成阶段数={phase_index}；"
                f"设备批次={batch_tag}。系统不会公开正确协议，需要根据历史反馈选择。"
            )
            retrieval_state = f"任务 物流恢复 阶段 {phase} 进度 {phase_index}"
            for candidates in itertools.permutations(动作全集):
                states.append(状态(
                    phase_index,
                    phase,
                    batch_tag,
                    observation,
                    retrieval_state,
                    candidates,
                ))
    return states


def 构造提示(state: 状态, candidates: Sequence[str] | None = None) -> str:
    """构造只允许输出单个候选编号的 Base 模型提示。"""

    choices = tuple(candidates or state.candidates)
    lines = [
        "你是一个冻结参数的物流控制策略。根据状态从候选动作中选择一项。",
        f"当前状态：{state.observation}",
        "候选动作：",
    ]
    lines.extend(f"{index}. {action}" for index, action in enumerate(choices, 1))
    lines.append("只输出一个对应数字，不要解释。答案：")
    return "\n".join(lines)
