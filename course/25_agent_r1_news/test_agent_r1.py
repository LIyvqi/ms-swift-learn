"""验证 Agent-R1 新闻规则检索、组合、动作协议与环境奖励。"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path

项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

知识模块 = import_module("course.25_agent_r1_news.knowledge_pipeline")
环境模块 = import_module("course.25_agent_r1_news.agent_system")
RuleKnowledgeBase = 知识模块.RuleKnowledgeBase
NewsPolicyEnvironment = 环境模块.NewsPolicyEnvironment
导入动作 = 环境模块.导入动作


class AgentR1新闻测试(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = RuleKnowledgeBase.from_jsonl(
            项目根目录 / "datasets/agent_r1_news/knowledge_rules.jsonl"
        )

    def test_混合检索召回体育规则(self):
        results = self.knowledge.search("网球队参加比赛，最终比分三比二", top_k=5)
        self.assertTrue(any(row["category"] == "体育" for row in results[:2]))

    def test_组合器删除旧版重复(self):
        result = self.knowledge.compose(["FIN-ROOT-LEGACY", "FIN-ROOT", "FIN-MARKET"])
        self.assertEqual(result["canonical_rule_ids"].count("FIN-ROOT"), 1)
        self.assertEqual(result["output_count"], 2)

    def test_动作协议拒绝非_json(self):
        payload, score, error = 导入动作("我决定选择财经")
        self.assertIsNone(payload)
        self.assertEqual(score, 0.0)
        self.assertTrue(error)

    def test_专家完成决策轨迹(self):
        config = {
            "task": "decision",
            "article": "球队在网球比赛中以三比二获胜，运动员获得冠军。",
            "label": "体育",
            "gold_rule_ids": ["SPT-ROOT", "SPT-COMP"],
            "gold_evidence": ["比赛", "运动员", "冠军"],
            "record_id": "unit-sports",
            "max_steps": 4,
        }
        env = NewsPolicyEnvironment(self.knowledge, config)
        env.reset()
        total_reward = 0.0
        info = {}
        while not env.done:
            _, reward, _, info = env.step(env.expert_action())
            total_reward += reward
        self.assertEqual(
            [step["event"] for step in info["trace"]],
            ["search_rules", "reflect", "compose_rules", "finish"],
        )
        self.assertEqual(info["metrics"]["decision_accuracy"], 1.0)
        self.assertEqual(info["metrics"]["rule_compliance"], 1.0)
        self.assertGreater(total_reward, 1.5)

    def test_无效动作受到惩罚(self):
        config = {
            "task": "retrieve",
            "article": "银行调整利率。",
            "label": "财经",
            "gold_rule_ids": ["FIN-ROOT", "FIN-MARKET"],
            "gold_evidence": ["银行", "利率"],
            "record_id": "unit-finance",
            "max_steps": 2,
        }
        env = NewsPolicyEnvironment(self.knowledge, config)
        env.reset()
        _, reward, done, info = env.step("不是动作")
        self.assertLess(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["event"], "invalid")

    def test_反思必须先检索(self):
        config = {
            "task": "decision",
            "article": "银行宣布调整贷款利率。",
            "label": "财经",
            "gold_rule_ids": ["FIN-ROOT", "FIN-MARKET"],
            "gold_evidence": ["银行", "利率"],
            "record_id": "unit-reflect",
            "max_steps": 4,
        }
        env = NewsPolicyEnvironment(self.knowledge, config)
        env.reset()
        action = (
            '<think>先改写。</think><action>{"tool":"reflect","arguments":'
            '{"diagnosis":"候选不准","new_query":"银行 利率","top_k":8}}</action>'
        )
        _, reward, done, info = env.step(action)
        self.assertLess(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["event"], "invalid_reflect")

    def test_错误_top_k_不会让环境崩溃(self):
        config = {
            "task": "retrieve",
            "article": "银行调整利率。",
            "label": "财经",
            "gold_rule_ids": ["FIN-ROOT", "FIN-MARKET"],
            "gold_evidence": ["银行", "利率"],
            "record_id": "unit-top-k",
            "max_steps": 2,
        }
        env = NewsPolicyEnvironment(self.knowledge, config)
        env.reset()
        action = (
            '<think>先检索。</think><action>{"tool":"search_rules","arguments":'
            '{"query":"银行 利率","top_k":"不是数字"}}</action>'
        )
        _, reward, done, info = env.step(action)
        self.assertGreaterEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["event"], "search_rules")

    def test_决策不能跳过检索与组合(self):
        config = {
            "task": "decision",
            "article": "球队赢得比赛冠军。",
            "label": "体育",
            "gold_rule_ids": ["SPT-ROOT", "SPT-COMP"],
            "gold_evidence": ["球队", "比赛", "冠军"],
            "record_id": "unit-no-bypass",
            "max_steps": 4,
        }
        env = NewsPolicyEnvironment(self.knowledge, config)
        env.reset()
        action = (
            '<think>直接作答。</think><action>{"tool":"finish","arguments":'
            '{"decision":"体育","matched_rules":["SPT-ROOT","SPT-COMP"],'
            '"evidence":["球队","比赛"],"unmet_conditions":[],"reason":"命中"}}</action>'
        )
        _, reward, done, info = env.step(action)
        self.assertLess(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["event"], "invalid_finish")

    def test_系统只展示当前任务结束格式(self):
        config = {
            "task": "compose",
            "article": "银行调整利率。",
            "label": "财经",
            "gold_rule_ids": ["FIN-ROOT", "FIN-MARKET"],
            "gold_evidence": ["银行", "利率"],
            "record_id": "unit-system-schema",
            "max_steps": 4,
        }
        env = NewsPolicyEnvironment(self.knowledge, config)
        _, _, system_prompt = env.reset()
        self.assertIn("canonical_rules", system_prompt)
        self.assertNotIn("selected_rules", system_prompt)
        self.assertNotIn("matched_rules", system_prompt)

    def test_组合任务拒绝决策结束格式(self):
        config = {
            "task": "compose",
            "article": "银行调整利率。",
            "label": "财经",
            "gold_rule_ids": ["FIN-ROOT", "FIN-MARKET"],
            "gold_evidence": ["银行", "利率"],
            "record_id": "unit-compose-schema",
            "max_steps": 4,
        }
        env = NewsPolicyEnvironment(self.knowledge, config)
        env.reset()
        env.step(env.expert_action())
        env.step(env.expert_action())
        wrong_action = (
            '<think>误用决策格式。</think><action>{"tool":"finish","arguments":'
            '{"decision":"财经","matched_rules":["FIN-ROOT"],"evidence":["银行"],'
            '"unmet_conditions":[],"reason":"命中"}}</action>'
        )
        _, reward, done, info = env.step(wrong_action)
        self.assertLess(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["event"], "invalid_finish_schema")
        self.assertEqual(info["metrics"]["task_schema_score"], 0.0)

        _, reward, done, info = env.step(env.expert_action())
        self.assertGreater(reward, 0)
        self.assertTrue(done)
        self.assertEqual(info["metrics"]["task_schema_score"], 1.0)

    def test_决策可从一次无效动作中恢复(self):
        config = {
            "task": "decision",
            "article": "球队在网球比赛中获得冠军。",
            "label": "体育",
            "gold_rule_ids": ["SPT-ROOT", "SPT-COMP"],
            "gold_evidence": ["球队", "比赛", "冠军"],
            "record_id": "unit-recovery",
            "max_steps": 6,
        }
        env = NewsPolicyEnvironment(self.knowledge, config)
        env.reset()
        wrong_action = (
            '<think>过早反思。</think><action>{"tool":"reflect","arguments":'
            '{"diagnosis":"候选不准","new_query":"球队 比赛","top_k":8}}</action>'
        )
        _, reward, done, info = env.step(wrong_action)
        self.assertLess(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["event"], "invalid_reflect")

        while not env.done:
            _, _, _, info = env.step(env.expert_action())
        self.assertEqual(info["trace"][-1]["event"], "finish")
        self.assertEqual(info["metrics"]["decision_accuracy"], 1.0)
        self.assertEqual(len(info["trace"]), 5)


if __name__ == "__main__":
    unittest.main()
