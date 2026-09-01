#!/usr/bin/env python3
"""验证两库隔离、动作协议、引用约束、专家闭环和 RiT 硬门控。"""

from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from importlib import import_module
from pathlib import Path


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))
记忆模块 = import_module("course.32_rit_rubric_rl.agent_memory")
环境模块 = import_module("course.32_rit_rubric_rl.agent_environment")
极简审核记忆 = 记忆模块.极简审核记忆
极简RiT审核环境 = 环境模块.极简RiT审核环境
动作文本 = 环境模块.动作文本
解析动作 = 环境模块.解析动作
集合_f1 = 环境模块.集合_f1
数据目录 = 项目根目录 / "datasets/rit_audit_agent"


def 读取_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class 极简RiTAgent测试(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.memory = 极简审核记忆(
            数据目录 / "rules.jsonl", 数据目录 / "cases.jsonl"
        )
        cls.smoke = 读取_jsonl(数据目录 / "rl_smoke.jsonl")

    def test_规则库和案例库独立且规模极简(self) -> None:
        self.assertEqual(len(self.memory.rules), 14)
        self.assertGreaterEqual(len(self.memory.cases), 100)
        self.assertLessEqual(len(self.memory.cases), 200)
        self.assertTrue(all(row["source_split"] == "train" for row in self.memory.cases))

    def test_验证测试样本不进入案例库(self) -> None:
        indexed = {row["record_id"] for row in self.memory.cases}
        for name in ("rl_validation.jsonl", "rl_test.jsonl"):
            for row in 读取_jsonl(数据目录 / name):
                self.assertNotIn(row["record_id"], indexed)

    def test_完整类别名能召回对应规则(self) -> None:
        for rule in self.memory.rules:
            results = self.memory.搜索规则(rule["category"], 1)
            self.assertEqual(results[0]["category"], rule["category"])

    def test_完全不相交的非空类别集合_f1为零(self) -> None:
        self.assertEqual(集合_f1(["animal_abuse"], ["child_abuse"]), 0.0)

    def test_案例检索排除当前训练样本(self) -> None:
        row = self.memory.cases[0]
        results = self.memory.搜索案例(
            f"{row['prompt']} {row['response']}", 5, "any", row["record_id"]
        )
        self.assertTrue(all(item["case_id"] != row["case_id"] for item in results))

    def test_只允许空思维前缀(self) -> None:
        action = 动作文本("search_rule", {"query": "violence", "top_k": 2})
        self.assertIsNotNone(解析动作("<think>\n\n</think>" + action)[0])
        payload, error = 解析动作("<think>先分析风险</think>" + action)
        self.assertIsNone(payload)
        self.assertIn("自由思维链", error)

    def test_虚构引用被拒绝(self) -> None:
        row = self.smoke[0]
        environment = 极简RiT审核环境(self.memory, row["env_config"])
        environment.reset()
        config = row["env_config"]
        source = config["response"] if len(config["response"].strip()) >= 3 else config["prompt"]
        categories = list(config["categories"])
        completion = 动作文本(
            "finish",
            {
                "is_safe": bool(config["is_safe"]),
                "categories": categories,
                "evidence": [source.strip()[:80]],
                "matched_rules": categories,
                "boundary": "已核对风险条件和安全边界，当前结论与证据一致。",
                "rule_ids": ["rule:invented@v9"],
                "case_ids": [],
            },
        )
        _, reward, done, info = environment.step(completion)
        self.assertFalse(done)
        self.assertLess(reward, 0)
        self.assertEqual(info["event"], "invalid_finish")

    def test_冒烟专家全部满分闭环(self) -> None:
        for row in self.smoke:
            environment = 极简RiT审核环境(self.memory, row["env_config"])
            environment.reset()
            info = {}
            while not environment.done:
                _, _, _, info = environment.step(environment.expert_action())
            self.assertEqual(info["event"], "finish", row["record_id"])
            self.assertEqual(info["metrics"]["response_reward"], 1.0)
            self.assertEqual(info["metrics"]["process_reward"], 1.0)
            self.assertEqual(info["metrics"]["gated_reward"], 1.0)

    def test_错误答案被最小值门控归零(self) -> None:
        row = next(item for item in self.smoke if item["env_config"]["is_safe"])
        environment = 极简RiT审核环境(self.memory, deepcopy(row["env_config"]))
        environment.reset()
        while environment.required_tools and not all(
            tool in environment.search_calls for tool in environment.required_tools
        ):
            environment.step(environment.expert_action())
        source = environment.response.strip() or environment.prompt.strip()
        wrong_category = self.memory.rules[0]["category"]
        completion = 动作文本(
            "finish",
            {
                "is_safe": False,
                "categories": [wrong_category],
                "evidence": [source[:80]],
                "matched_rules": [wrong_category],
                "boundary": "风险规则成立，未发现可以适用的安全例外。",
                "rule_ids": [],
                "case_ids": [],
            },
        )
        _, _, _, info = environment.step(completion)
        self.assertEqual(info["metrics"]["response_reward"], 0.0)
        self.assertEqual(info["metrics"]["gated_reward"], 0.0)

    def test_训练与环境数据格式完整(self) -> None:
        sft_rows = 读取_jsonl(数据目录 / "sft_smoke.jsonl")
        self.assertEqual(len(sft_rows), 32)
        self.assertTrue(all(row["messages"][-1]["role"] == "assistant" for row in sft_rows))
        routes = {tuple(row["expert_route"]) for row in 读取_jsonl(数据目录 / "sft_train.jsonl")}
        self.assertIn(("finish",), routes)
        self.assertIn(("search_rule", "finish"), routes)
        self.assertIn(("search_case", "finish"), routes)
        self.assertIn(("search_rule", "search_case", "finish"), routes)


if __name__ == "__main__":
    unittest.main()
