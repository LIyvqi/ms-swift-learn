#!/usr/bin/env python3
"""置信度解析、Platt 校准、选择性预测与统一指标。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from rlcr_rewards import 是否完整格式, 解析答案, 解析置信度


def 限制概率(value: float, epsilon: float = 1e-6) -> float:
    """避免对数和 logit 出现无穷。"""

    return min(1 - epsilon, max(epsilon, float(value)))


@dataclass
class Platt校准器:
    """对原始概率的 logit 拟合一维逻辑校准。"""

    weight: float
    bias: float

    @classmethod
    def 拟合(
        cls, confidence: list[float], targets: list[int], steps: int = 4000,
        learning_rate: float = 0.03, l2: float = 0.01,
    ) -> "Platt校准器":
        """仅使用独立校准集拟合斜率和偏置。"""

        x = np.asarray([
            math.log(限制概率(value) / (1 - 限制概率(value)))
            for value in confidence
        ], dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if len(set(targets)) < 2:
            # 校准集若恰好全对或全错，斜率不可辨识；用拉普拉斯平滑常数概率稳健退化。
            rate = (float(y.sum()) + 1.0) / (len(y) + 2.0)
            return cls(weight=0.0, bias=math.log(rate / (1 - rate)))
        weight = 1.0
        rate = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        bias = math.log(rate / (1 - rate))
        for _ in range(steps):
            logits = np.clip(weight * x + bias, -30, 30)
            probabilities = 1 / (1 + np.exp(-logits))
            error = probabilities - y
            weight -= learning_rate * (float(np.mean(error * x)) + l2 * weight)
            bias -= learning_rate * float(np.mean(error))
        return cls(weight=weight, bias=bias)

    def 预测(self, confidence: float) -> float:
        """将一个原始置信度映射为校准概率。"""

        value = 限制概率(confidence)
        logit = math.log(value / (1 - value))
        calibrated = np.clip(self.weight * logit + self.bias, -30, 30)
        return float(1 / (1 + math.exp(-float(calibrated))))

    def 转字典(self) -> dict[str, float]:
        """导出可复现参数。"""

        return {
            "weight": self.weight,
            "bias": self.bias,
            "constant_fallback": self.weight == 0.0,
        }


def ECE(confidence: list[float], targets: list[int], bins: int = 10) -> float:
    """计算等宽期望校准误差。"""

    total = len(confidence)
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [
            i for i, score in enumerate(confidence)
            if low <= score < high or index == bins - 1 and score == 1
        ]
        if selected:
            mean_confidence = sum(confidence[i] for i in selected) / len(selected)
            accuracy = sum(targets[i] for i in selected) / len(selected)
            result += len(selected) / total * abs(mean_confidence - accuracy)
    return result


def AURC(confidence: list[float], targets: list[int]) -> float:
    """按置信度降序积分风险—覆盖率曲线。"""

    order = sorted(range(len(confidence)), key=lambda index: (-confidence[index], index))
    errors = 0
    risks = []
    for rank, index in enumerate(order, 1):
        errors += 1 - targets[index]
        risks.append(errors / rank)
    return sum(risks) / len(risks)


def AUROC(scores: list[float], targets: list[int]) -> float | None:
    """用带平票的秩和计算正确性 AUROC。"""

    positives = sum(targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    positive_rank_sum = sum(rank for rank, target in zip(ranks, targets) if target)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def 选择阈值(
    confidence: list[float], targets: list[int], maximum_risk: float = 0.05,
    minimum_coverage: float = 0.1,
) -> float:
    """在校准集上选择满足风险约束的最大覆盖阈值。"""

    candidates = sorted(set(confidence))
    best: tuple[float, float] | None = None
    for threshold in candidates:
        selected = [index for index, score in enumerate(confidence) if score >= threshold]
        coverage = len(selected) / len(confidence)
        if not selected or coverage < minimum_coverage:
            continue
        risk = sum(1 - targets[index] for index in selected) / len(selected)
        if risk <= maximum_risk and (best is None or coverage > best[0]):
            best = (coverage, threshold)
    return best[1] if best else 1.0


def 汇总置信指标(
    confidence: list[float], targets: list[int], threshold: float | None = None,
) -> dict[str, Any]:
    """汇总校准、排序和选择性指标。"""

    values = [限制概率(value) for value in confidence]
    result: dict[str, Any] = {
        "samples": len(values),
        "accuracy": sum(targets) / len(targets),
        "ece": ECE(values, targets),
        "brier": sum((value - target) ** 2 for value, target in zip(values, targets)) / len(targets),
        "nll": -sum(
            target * math.log(value) + (1 - target) * math.log(1 - value)
            for value, target in zip(values, targets)
        ) / len(targets),
        "aurc": AURC(values, targets),
        "correctness_auroc": AUROC(values, targets),
    }
    if threshold is not None:
        accepted = [index for index, value in enumerate(values) if value >= threshold]
        result.update({
            "threshold": threshold,
            "coverage": len(accepted) / len(values),
            "selective_risk": (
                sum(1 - targets[index] for index in accepted) / len(accepted) if accepted else None
            ),
        })
    return result


def 解析响应(text: str) -> dict[str, Any]:
    """返回评测所需的类别、置信度和格式标记。"""

    return {
        "predicted_label": 解析答案(text),
        "reported_confidence": 解析置信度(text),
        "format_valid": 是否完整格式(text),
    }
