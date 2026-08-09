#!/usr/bin/env python3
"""用人工构造样例验证第 03 课奖励函数的关键边界。"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path

项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

奖励模块 = import_module("course.plugins.gsm8k_rewards")
GSM8KAccuracy = 奖励模块.GSM8KAccuracy
GSM8KCoTCalculation = 奖励模块.GSM8KCoTCalculation
GSM8KCoTConsistency = 奖励模块.GSM8KCoTConsistency
GSM8KCoTGrounding = 奖励模块.GSM8KCoTGrounding
GSM8KCoTLLMJudge = 奖励模块.GSM8KCoTLLMJudge
GSM8KCoTStructure = 奖励模块.GSM8KCoTStructure

正确回答 = (
    "<think>题目给出每箱 8 瓶和 5 箱，因此总数为 8*5=40 瓶。</think>\n\\boxed{40}"
)


class 奖励函数测试(unittest.TestCase):
    def test_答案正确性(self):
        reward = GSM8KAccuracy()([正确回答], ["计算后得到 #### 40"])
        self.assertEqual(reward, [1.0])

    def test_严格结构拒绝空思考(self):
        reward = GSM8KCoTStructure()([正确回答, "<think>\n</think>\n\\boxed{40}"])
        self.assertEqual(reward, [1.0, 0.0])

    def test_计算奖励执行算式(self):
        completions = [
            正确回答,
            "<think>使用题目条件计算 8*5=35。</think>\n\\boxed{35}",
            "<think>先写一个无关算式 1+1=2。</think>\n\\boxed{40}",
            "<think>每箱 $8，共 5 箱，因此 $8*5=$40。</think>\n\\boxed{40}",
        ]
        reward = GSM8KCoTCalculation()(
            completions,
            ["每箱 8 瓶，共 5 箱。"] * 4,
            ["40"] * 4,
        )
        self.assertEqual(reward, [1.0, 0.0, 0.0, 1.0])

    def test_题目数值覆盖(self):
        reward = GSM8KCoTGrounding()([正确回答], ["每箱 8 瓶，共 5 箱。"])
        self.assertEqual(reward, [1.0])

    def test_过程答案一致(self):
        reward = GSM8KCoTConsistency()(
            [正确回答, "<think>只算到 20。</think>\n\\boxed{40}"]
        )
        self.assertEqual(reward, [1.0, 0.0])

    def test_裁判分数解析(self):
        self.assertEqual(GSM8KCoTLLMJudge.提取裁判分数("评价完成。[[3]]"), 0.75)
        self.assertEqual(GSM8KCoTLLMJudge.提取裁判分数("没有合法分数"), 0.0)


if __name__ == "__main__":
    unittest.main()
