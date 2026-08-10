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
奖励模块 = import_module("course.plugins.agent_r1_news")
选择模块 = import_module("course.25_agent_r1_news.select_best_evaluation")
评测模块 = import_module("course.25_agent_r1_news.evaluate_agent")
配对模块 = import_module("course.25_agent_r1_news.compare_paired_evaluations")
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

    def test_重排分数实际参与排序(self):
        results = self.knowledge.search("银行 利率", top_k=5, rerank=True)
        self.assertEqual(results[0]["rule_id"], "FIN-MARKET")
        self.assertIn("retrieval", results[0]["scores"])
        self.assertIn("rerank", results[0]["scores"])
        self.assertNotEqual(
            results[0]["scores"]["retrieval"], results[0]["scores"]["rerank"]
        )

    def test_组合器删除旧版重复(self):
        result = self.knowledge.compose(["FIN-ROOT-LEGACY", "FIN-ROOT", "FIN-MARKET"])
        self.assertEqual(result["canonical_rule_ids"].count("FIN-ROOT"), 1)
        self.assertEqual(result["output_count"], 2)

    def test_组合器报告跨类别冲突并保留例外(self):
        result = self.knowledge.compose(["FIN-ROOT", "SPT-ROOT", "FIN-ENTERPRISE"])
        self.assertEqual(result["conflicts"][0]["type"], "category_conflict")
        categories = result["conflicts"][0]["categories"]
        self.assertEqual(set(categories), {"财经", "体育"})
        enterprise = next(
            rule for rule in result["rules"] if rule["canonical_id"] == "FIN-ENTERPRISE"
        )
        self.assertTrue(enterprise["exceptions"])

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

    def test_错误首轮召回可由反思改写修正(self):
        config = {
            "task": "decision",
            "article": "球队在网球比赛中以三比二获胜，运动员获得冠军。",
            "label": "体育",
            "gold_rule_ids": ["SPT-ROOT", "SPT-COMP"],
            "gold_evidence": ["比赛", "运动员", "冠军"],
            "record_id": "unit-reflection-gain",
            "max_steps": 6,
        }
        env = NewsPolicyEnvironment(self.knowledge, config)
        env.reset()
        wrong_search = (
            "<think>先测试一个可能错误的查询。</think><action>"
            '{"tool":"search_rules","arguments":'
            '{"query":"银行 利率 证券","top_k":5}}</action>'
        )
        _, _, done, info = env.step(wrong_search)
        first_recall = info["metrics"]["retrieval_recall"]
        self.assertFalse(done)

        repaired_search = (
            "<think>首轮偏向财经，需要围绕赛事证据改写。</think><action>"
            '{"tool":"reflect","arguments":'
            '{"diagnosis":"首轮类别错误","new_query":"球队 比赛 冠军",'
            '"top_k":5}}</action>'
        )
        _, reward, done, info = env.step(repaired_search)
        self.assertFalse(done)
        self.assertGreaterEqual(reward, 0)
        self.assertEqual(
            [step["event"] for step in info["trace"]],
            ["search_rules", "reflect"],
        )
        self.assertGreater(info["metrics"]["retrieval_recall"], first_recall)
        self.assertGreater(info["metrics"]["reflection_gain"], 0)
        self.assertEqual(info["metrics"]["reflection_success"], 1.0)

    def test_单字证据不会让专家反复非法反思(self):
        for task in ("retrieve", "compose", "decision"):
            with self.subTest(task=task):
                config = {
                    "task": task,
                    "article": "两国执政党代表举行会谈并开展友好访问。",
                    "label": "政治",
                    "gold_rule_ids": ["POL-ROOT"],
                    "gold_evidence": ["党"],
                    "record_id": f"unit-single-character-{task}",
                    "max_steps": 6 if task == "decision" else 4,
                }
                env = NewsPolicyEnvironment(self.knowledge, config)
                env.reset()
                info = {}
                while not env.done:
                    _, _, _, info = env.step(env.expert_action())
                events = [step["event"] for step in info["trace"]]
                self.assertEqual(events[-1], "finish")
                self.assertFalse(
                    any(event.startswith("invalid") for event in events), events
                )

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

    def test_任务专属奖励正确使用掩码(self):
        reward = 奖励模块.AgentNewsRetrievalReward()
        values = reward(
            [],
            ["retrieve", "decision"],
            [
                {"task_metrics": {"retrieval_f1": 0.75}},
                {"task_metrics": {"retrieval_f1": 1.0}},
            ],
        )
        self.assertEqual(values, [0.75, None])

    def test_决策奖励同时检查分类规则和证据(self):
        reward = 奖励模块.AgentNewsDecisionReward()
        values = reward(
            [],
            ["decision"],
            [
                {
                    "task_metrics": {
                        "decision_accuracy": 1.0,
                        "rule_compliance": 0.5,
                        "evidence_coverage": 0.75,
                    }
                }
            ],
        )
        self.assertAlmostEqual(values[0], (1.0 + 0.3 * 0.5 + 0.2 * 0.75) / 1.5)

    def test_协议奖励惩罚无效动作和冗余轮次(self):
        reward = 奖励模块.AgentNewsProtocolReward()
        clean_trace = [
            {"event": "search_rules"},
            {"event": "reflect"},
            {"event": "finish"},
        ]
        noisy_trace = [
            {"event": "invalid_search"},
            {"event": "search_rules"},
            {"event": "reflect"},
            {"event": "search_rules"},
            {"event": "finish"},
        ]
        values = reward(
            [],
            ["retrieve", "retrieve"],
            [
                {
                    "task_metrics": {
                        "protocol_score": 1.0,
                        "task_schema_score": 1.0,
                    },
                    "agent_trace": clean_trace,
                },
                {
                    "task_metrics": {
                        "protocol_score": 1.0,
                        "task_schema_score": 1.0,
                    },
                    "agent_trace": noisy_trace,
                },
            ],
        )
        self.assertEqual(values[0], 1.0)
        self.assertLess(values[1], values[0])

    def test_反思奖励使用召回增益和成功标记(self):
        reward = 奖励模块.AgentNewsReflectionReward()
        values = reward(
            [],
            ["retrieve"],
            [
                {
                    "task_metrics": {
                        "reflection_gain": 0.4,
                        "reflection_success": 1.0,
                    }
                }
            ],
        )
        self.assertAlmostEqual(values[0], 0.7 * 0.4 + 0.3)

    def test_检查点选择同时覆盖三个任务(self):
        data = {
            "summary": {
                "retrieve": {"retrieval_f1": 0.6},
                "compose": {"composition_f1": 0.9},
                "decision": {
                    "decision_accuracy": 0.8,
                    "composition_f1": 0.7,
                    "evidence_coverage": 0.6,
                },
            }
        }
        score, metrics = 选择模块.计算选择分数(data)
        self.assertAlmostEqual(metrics["decision_subscore"], 0.7)
        self.assertAlmostEqual(score, (0.6 + 0.9 + 0.7) / 3)

    def test_检查点选择同分时偏好较早节点(self):
        common = {
            "summary": {
                "retrieve": {"retrieval_f1": 1.0},
                "compose": {"composition_f1": 1.0},
                "decision": {
                    "decision_accuracy": 1.0,
                    "composition_f1": 1.0,
                    "evidence_coverage": 1.0,
                },
            },
            "agent_summary": {
                "completion_rate": 1.0,
                "invalid_action_rate": 0.0,
            },
        }
        later = common | {"adapter": "run/checkpoint-1200"}
        earlier = common | {"adapter": "run/checkpoint-960"}
        selected = 选择模块.选择最佳结果([later, earlier])
        self.assertEqual(selected["best"]["step"], 960)

    def test_检查点选择集与最终留出集互不重叠(self):
        path = 项目根目录 / "datasets/agent_r1_news/rl_val.jsonl"
        selection = 评测模块.读取_jsonl(path, 120, 0)
        heldout = 评测模块.读取_jsonl(path, 840, 120)
        self.assertEqual(len(selection), 120)
        self.assertEqual(len(heldout), 840)
        self.assertEqual(
            {row["task"] for row in selection}, {"retrieve", "compose", "decision"}
        )
        self.assertEqual(
            {row["task"] for row in heldout}, {"retrieve", "compose", "decision"}
        )
        selection_ids = {row["record_id"] for row in selection}
        heldout_ids = {row["record_id"] for row in heldout}
        self.assertEqual(len(selection_ids), 40)
        self.assertEqual(len(heldout_ids), 280)
        self.assertFalse(selection_ids & heldout_ids)

    def test_留出评测按新闻做配对统计(self):
        def 构造结果(adapter, decision_a, retrieval):
            traces = []
            for record_id, decision in (("a", decision_a), ("b", 1.0)):
                traces.extend(
                    [
                        {
                            "record_id": record_id,
                            "task": "retrieve",
                            "metrics": {"retrieval_f1": retrieval},
                        },
                        {
                            "record_id": record_id,
                            "task": "compose",
                            "metrics": {"composition_f1": 0.8},
                        },
                        {
                            "record_id": record_id,
                            "task": "decision",
                            "metrics": {
                                "decision_accuracy": decision,
                                "composition_f1": 0.7,
                                "evidence_coverage": 0.6,
                            },
                        },
                    ]
                )
            return {"adapter": adapter, "traces": traces}

        baseline = 构造结果("sft", 0.0, 0.5)
        candidate = 构造结果("grpo", 1.0, 0.7)
        report = 配对模块.配对比较(baseline, candidate, bootstrap_samples=200, seed=7)
        self.assertEqual(report["paired_news"], 2)
        self.assertAlmostEqual(report["metrics"]["retrieval_f1"]["difference"], 0.2)
        self.assertAlmostEqual(
            report["metrics"]["decision_accuracy"]["difference"], 0.5
        )
        self.assertEqual(report["decision_mcnemar"]["improvements"], 1)
        self.assertEqual(report["decision_mcnemar"]["regressions"], 0)


if __name__ == "__main__":
    unittest.main()
