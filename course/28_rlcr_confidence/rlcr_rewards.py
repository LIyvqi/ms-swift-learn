"""RLCR 新闻分类的正确性、Brier、对数与格式奖励。"""

from __future__ import annotations

import math
import re
from typing import List

from swift.rewards import ORM, orms


允许标签 = ("政治", "财经", "体育", "计算机")
标签模式 = "|".join(map(re.escape, 允许标签))
完整模式 = re.compile(
    rf"^\s*(?:<think>.*?</think>\s*)?"
    rf"<answer>\s*({标签模式})\s*</answer>\s*"
    rf"<confidence>\s*((?:0(?:\.\d+)?)|(?:1(?:\.0+)?))\s*</confidence>\s*$",
    re.DOTALL,
)
答案模式 = re.compile(rf"<answer>\s*({标签模式})\s*</answer>")
框选模式 = re.compile(rf"\\boxed\{{({标签模式})\}}")
置信模式 = re.compile(r"<confidence>\s*([^<]+)\s*</confidence>")


def 解析答案(text: str) -> str:
    """优先解析 RLCR 结构，同时兼容旧框选基线。"""

    match = 答案模式.search(str(text))
    if match:
        return match.group(1)
    boxed = 框选模式.findall(str(text))
    return boxed[-1] if boxed else ""


def 解析置信度(text: str) -> float | None:
    """只接受有限且位于 [0,1] 的数值。"""

    match = 置信模式.search(str(text))
    if not match:
        return None
    try:
        value = float(match.group(1).strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def 是否完整格式(text: str) -> bool:
    """验证类别和置信度都只出现一次。"""

    return bool(完整模式.fullmatch(str(text)))


class RLCRAccuracy(ORM):
    """奖励最终分类正确。"""

    def __call__(self, completions, label, **kwargs) -> List[float]:
        return [float(解析答案(text) == gold) for text, gold in zip(completions, label)]


class RLCRBrier(ORM):
    """用二元 Brier proper score 训练“本次预测正确”概率。"""

    def __call__(self, completions, label, **kwargs) -> List[float]:
        rewards = []
        for text, gold in zip(completions, label):
            confidence = 解析置信度(text)
            if confidence is None:
                rewards.append(-1.0)
                continue
            target = float(解析答案(text) == gold)
            rewards.append(-((confidence - target) ** 2))
        return rewards


class RLCRLogScore(ORM):
    """用对数 proper score 对照训练置信度。"""

    def __call__(self, completions, label, **kwargs) -> List[float]:
        rewards = []
        epsilon = 0.01
        for text, gold in zip(completions, label):
            confidence = 解析置信度(text)
            if confidence is None:
                rewards.append(math.log(epsilon))
                continue
            confidence = min(1 - epsilon, max(epsilon, confidence))
            correct = 解析答案(text) == gold
            rewards.append(math.log(confidence if correct else 1 - confidence))
        return rewards


class RLCRFormat(ORM):
    """奖励可机器解析的严格联合输出。"""

    def __call__(self, completions, **kwargs) -> List[float]:
        return [float(是否完整格式(text)) for text in completions]


orms["course_rlcr_accuracy"] = RLCRAccuracy
orms["course_rlcr_brier"] = RLCRBrier
orms["course_rlcr_log_score"] = RLCRLogScore
orms["course_rlcr_format"] = RLCRFormat
