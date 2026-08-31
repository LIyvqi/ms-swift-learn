#!/usr/bin/env python3
"""覆盖独立后端、层级约束、防泄漏、动作协议和专家闭环。"""

from __future__ import annotations

import json
import sys
import unittest
from importlib import import_module
from pathlib import Path


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

网关模块 = import_module("course.31_hierarchical_memory_agent.memory_gateway")
环境模块 = import_module("course.31_hierarchical_memory_agent.agent_environment")
分层记忆网关 = 网关模块.分层记忆网关
分层记忆审核环境 = 环境模块.分层记忆审核环境
动作文本 = 环境模块.动作文本

数据目录 = 项目根目录 / "datasets/hierarchical_memory_audit"


def 读取_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class 分层记忆测试(unittest.TestCase):
    """所有测试都使用准备脚本生成的真实三库后端。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.gateway = 分层记忆网关(数据目录 / "source_registry.json")
        cls.rl_rows = 读取_jsonl(数据目录 / "rl_smoke.jsonl")

    def test_三个独立连接器和层级深度(self) -> None:
        summary = self.gateway.摘要()
        self.assertEqual({row["source_id"] for row in summary["sources"]}, {
            "rule_store", "case_store", "knowledge_store"
        })
        connectors = {row["connector"] for row in summary["sources"]}
        self.assertEqual(connectors, {"jsonl_rules", "sqlite_cases", "directory_knowledge"})
        self.assertGreaterEqual(max(row["max_depth"] for row in summary["sources"]), 6)

    def test_定位后才能搜索且路径必须原样返回(self) -> None:
        row = next(item for item in self.rl_rows if item["categories"])
        environment = 分层记忆审核环境(self.gateway, row["env_config"])
        environment.reset()
        invalid = 动作文本(
            "search",
            {
                "source_id": "rule_store",
                "path": "内容审核政策",
                "query": "violence",
                "top_k": 3,
            },
            "尝试跳过定位。",
        )
        _, reward, done, info = environment.step(invalid)
        self.assertFalse(done)
        self.assertLess(reward, 0)
        self.assertEqual(info["event"], "invalid_search_scope")

    def test_直接结束是合法策略(self) -> None:
        row = next(item for item in self.rl_rows if item["is_safe"])
        environment = 分层记忆审核环境(self.gateway, row["env_config"])
        environment.reset()
        evidence = row["env_config"]["response"][:80]
        action = 动作文本(
            "finish",
            {
                "is_safe": True,
                "categories": [],
                "evidence": [evidence],
                "memory_ids": [],
                "confidence": 0.9,
                "reason": "文本没有给出违规协助。",
            },
            "证据直接，当前不需要调用外部记忆。",
        )
        _, reward, done, info = environment.step(action)
        self.assertTrue(done)
        self.assertGreater(reward, 1.0)
        self.assertEqual(info["event"], "finish")
        self.assertEqual(info["metrics"]["direct_finish"], 1.0)

    def test_验证测试样本不在案例库(self) -> None:
        indexed = {
            str(record.metadata["record_id"])
            for record in self.gateway.connectors["case_store"].records
        }
        for split in ("rl_validation.jsonl", "rl_test.jsonl"):
            for row in 读取_jsonl(数据目录 / split):
                self.assertNotIn(row["record_id"], indexed)

    def test_多标签案例投影到各自正确路由(self) -> None:
        rules = 读取_jsonl(数据目录 / "rules.jsonl")
        route_by_category = {
            row["category"]: row["route"]
            for row in rules
            if row.get("status") == "active"
        }
        projections_by_record: dict[str, set[str]] = {}
        for record in self.gateway.connectors["case_store"].records:
            projection = str(record.metadata["projection_category"])
            if projection == "safe":
                continue
            self.assertEqual(record.path[-2], projection)
            self.assertEqual(record.path[-3], route_by_category[projection])
            record_id = str(record.metadata["record_id"])
            projections_by_record.setdefault(record_id, set()).add(projection)
        self.assertTrue(
            any(len(projections) > 1 for projections in projections_by_record.values()),
            "测试数据中应该至少有一个真实多标签 Case",
        )

    def test_候选缓冲区不属于任何可检索源(self) -> None:
        candidate_path = 数据目录 / "candidate_cases.jsonl"
        self.assertTrue(candidate_path.exists())
        self.assertNotIn(candidate_path.name, {
            connector.location.name for connector in self.gateway.connectors.values()
        })

    def test_冒烟专家逐条正常结束(self) -> None:
        for row in self.rl_rows:
            environment = 分层记忆审核环境(self.gateway, row["env_config"])
            environment.reset()
            info = {}
            while not environment.done:
                _, _, _, info = environment.step(environment.expert_action())
            self.assertEqual(info["event"], "finish", row["record_id"])
            self.assertEqual(info["metrics"]["safety_accuracy"], 1.0)
            self.assertEqual(info["metrics"]["category_f1"], 1.0)

    def test_状态转移数据对齐最后一轮动作(self) -> None:
        rows = 读取_jsonl(数据目录 / "sft_state_train.jsonl")
        full_rows = 读取_jsonl(数据目录 / "sft_train.jsonl")
        expected = sum(
            sum(message["role"] == "assistant" for message in row["messages"]) > 1
            for row in full_rows
        )
        self.assertEqual(len(rows), expected)
        self.assertEqual(
            {row["target_action"] for row in rows},
            {"locate", "search", "finish"},
        )
        for row in rows:
            self.assertEqual(row["messages"][-1]["role"], "assistant")
            self.assertIn(
                f'"tool":"{row["target_action"]}"',
                row["messages"][-1]["content"],
            )


if __name__ == "__main__":
    unittest.main()
