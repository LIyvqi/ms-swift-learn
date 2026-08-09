#!/usr/bin/env python3
"""不加载模型即可运行的 JitRL 数学与记忆单元测试。"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path

from jitrl_core import softmax, 修正动作_logits, 折扣回报, 文本杰卡德, 经验记忆


def 测试闭式解() -> None:
    logits = [0.2, -0.4, 1.1]
    advantages = [0.8, -0.3, 0.1]
    beta = 2.5
    corrected = softmax([z + beta * advantage for z, advantage in zip(logits, advantages)])
    base = softmax(logits)
    weighted = [probability * math.exp(beta * advantage) for probability, advantage in zip(base, advantages)]
    denominator = sum(weighted)
    policy_form = [value / denominator for value in weighted]
    assert max(abs(left - right) for left, right in zip(corrected, policy_form)) < 1e-12


def 测试记忆与持久化() -> None:
    memory = 经验记忆()
    memory.添加轨迹(
        ["任务 物流 阶段 入口", "任务 物流 阶段 扫描"],
        ["琥珀协议", "白银协议"],
        [0.5, 2.0],
        gamma=0.5,
        episode=0,
    )
    assert 折扣回报([0.5, 2.0], 0.5) == [1.5, 2.0]
    assert 文本杰卡德("阶段 入口", "阶段 入口") == 1.0
    details = 修正动作_logits(
        "任务 物流 阶段 入口",
        ["琥珀协议", "靛蓝协议", "白银协议"],
        [0.0, 0.0, 0.0],
        memory,
        random.Random(1),
        beta=3.0,
        top_k=10,
        similarity_threshold=0.95,
        unseen_probability=0.0,
        optimism_alpha=5.0,
    )
    assert details.neighbor_count == 1
    assert details.corrected_logits["琥珀协议"] > details.corrected_logits["靛蓝协议"]

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "memory.jsonl"
        memory.保存(path)
        restored = 经验记忆.加载(path)
        assert restored.entries == memory.entries


if __name__ == "__main__":
    测试闭式解()
    测试记忆与持久化()
    print("JITRL_CORE_TEST=PASS")
