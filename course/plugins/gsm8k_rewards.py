"""供 ms-swift GRPO 使用的 GSM8K 答案正确性与输出格式奖励。"""

import re
from typing import List

from swift.rewards import ORM, orms


class GSM8KAccuracy(ORM):

    @staticmethod
    def extract(text: str) -> str:
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text[-1000:])
        if boxed:
            return boxed[-1].replace(",", "").replace(" ", "").strip()
        marked = re.findall(r"####\s*([\-\d,.\s]+)", text[-1000:])
        return marked[-1].replace(",", "").replace(" ", "").strip() if marked else ""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, target in zip(completions, solution):
            predicted, expected = self.extract(completion), self.extract(target)
            try:
                correct = bool(predicted and expected and abs(float(predicted) - float(expected)) < 1e-5)
            except (ValueError, OverflowError):
                correct = bool(predicted and expected and predicted == expected)
            rewards.append(float(correct))
        return rewards


class GSM8KFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        return [float(bool(re.search(r"\\boxed\{[^}]+\}", text))) for text in completions]


orms["course_gsm8k_accuracy"] = GSM8KAccuracy
orms["course_gsm8k_format"] = GSM8KFormat
