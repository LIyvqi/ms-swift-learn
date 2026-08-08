"""供 ms-swift 在线强化学习使用的四分类正确性与格式奖励。"""

import re
from typing import List

from swift.rewards import ORM, orms


允许标签 = ("政治", "财经", "体育", "计算机")
标签模式 = "|".join(map(re.escape, 允许标签))


class ClassificationAccuracy(ORM):
    """检查回答表达的最后一个合法标签是否等于标准标签。"""

    @staticmethod
    def extract(text: str) -> str:
        boxed = re.findall(rf"\\boxed\{{({标签模式})\}}", text)
        if boxed:
            return boxed[-1]
        mentioned = re.findall(标签模式, text)
        return mentioned[-1] if mentioned else ""

    def __call__(self, completions, label, **kwargs) -> List[float]:
        return [
            float(self.extract(completion) == expected.strip())
            for completion, expected in zip(completions, label)
        ]


class ClassificationFormat(ORM):
    """奖励简洁的框选答案，并兼容 Qwen3.5 模板生成的空思考标签。"""

    def __call__(self, completions, **kwargs) -> List[float]:
        pattern = rf"^\s*(?:<think>\s*</think>\s*)?\\boxed\{{(?:{标签模式})\}}\s*$"
        return [float(bool(re.fullmatch(pattern, text, re.DOTALL))) for text in completions]


orms["course_classification_accuracy"] = ClassificationAccuracy
orms["course_classification_format"] = ClassificationFormat
