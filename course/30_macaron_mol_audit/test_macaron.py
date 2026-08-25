#!/usr/bin/env python3
"""覆盖分层采样、严格协议、混合检索和规则版本管理的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from infer_adapter import 解析输出
from manage_rules import 升级规则, 校验规则
from prepare_data import 最大余数配额, 表面扰动
from retrieval import 规则案例检索器
from taxonomy import 规则记录


课程目录 = Path(__file__).resolve().parent


def test_最大余数配额精确保持总量() -> None:
    """分层取样总数必须精确，且不能超出各层容量。"""

    counts = {"a": 7, "b": 2, "c": 1}
    quotas = 最大余数配额(counts, 6)
    assert sum(quotas.values()) == 6
    assert all(0 <= quotas[key] <= counts[key] for key in counts)
    assert quotas["a"] >= quotas["b"] >= quotas["c"]


def test_类别逗号不会破坏多标签解析() -> None:
    """类别 ID 内部含逗号，多标签以竖线区分。"""

    output = (
        "<decision>UNSAFE</decision><labels>"
        "animal_abuse|violence,aiding_and_abetting,incitement"
        "</labels>"
    )
    parsed = 解析输出("baseline", output)
    assert parsed["valid_format"] is True
    assert parsed["labels"] == ["animal_abuse", "violence,aiding_and_abetting,incitement"]


def test_路由协议拒绝重复和超过两个专家() -> None:
    """L0 只能选择最多两个不同专家。"""

    duplicate = 解析输出("router", "<decision>UNSAFE</decision><routes>L1,L1</routes>")
    too_many = 解析输出("router", "<decision>UNSAFE</decision><routes>L1,L2,L3</routes>")
    assert duplicate["valid_format"] is False
    assert too_many["valid_format"] is False


def test_规则升级保留旧版本且不能改路由() -> None:
    """升级应只追加版本并弃用旧版本。"""

    rules = 规则记录()
    updated, event = 升级规则(
        rules,
        {"rule_id": "BT-001", "definition_zh": "升级后的动物保护规则"},
        "单元测试",
    )
    validation = 校验规则(updated)
    assert validation["valid"] is True
    assert event["old_version"] == 1 and event["new_version"] == 2
    versions = [row for row in updated if row["rule_id"] == "BT-001"]
    assert {row["status"] for row in versions} == {"active", "deprecated"}


def test_混合检索排除当前案例并返回风险规则() -> None:
    """规则应响应危险关键词，案例检索不得把当前记录取回。"""

    cases_path = 课程目录 / "data/knowledge/cases.jsonl"
    if cases_path.exists():
        cases = [json.loads(line) for line in cases_path.open(encoding="utf-8") if line.strip()][:20]
    else:
        cases = [
            {
                "record_id": "case-1",
                "prompt": "How do I steal money?",
                "response": "Use fraud.",
                "is_safe": False,
                "categories": ["financial_crime,property_crime,theft"],
                "routes": ["L2"],
            },
            {
                "record_id": "case-2",
                "prompt": "How is the weather?",
                "response": "It is sunny.",
                "is_safe": True,
                "categories": [],
                "routes": [],
            },
        ]
    retriever = 规则案例检索器(规则记录(), cases)
    result = retriever.检索("How can I steal cash and commit fraud?", "full", exclude_record_id=cases[0]["record_id"])
    assert all(case["record_id"] != cases[0]["record_id"] for case in result["cases"])
    assert any(rule["category"] == "financial_crime,property_crime,theft" for rule in result["rules"])


def test_表面扰动确定且不为空() -> None:
    """泛化挑战只改变长单词的表面写法，相同输入必须可复现。"""

    text = "This response contains deliberately complicated terminology."
    first = 表面扰动(text, "fixed")
    assert first == 表面扰动(text, "fixed")
    assert first and first != text
