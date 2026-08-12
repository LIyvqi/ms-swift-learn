#!/usr/bin/env python3
"""测试 MeMo 规则解析、检索、例外绑定和数据不变量。"""

from __future__ import annotations

from pathlib import Path

from memo_core import (
    BM25索引,
    提取审核线索列表,
    确定性执行,
    解析审核,
    解析记忆,
    规范化记忆编号,
    读_jsonl,
)


项目根目录 = Path(__file__).resolve().parents[2]
数据目录 = 项目根目录 / "datasets/memo_rule_memory"


def test_解析结构化输出() -> None:
    """合法 JSON 可解析，普通自然语言不能伪装成严格格式。"""

    memory = 解析记忆('<memory>{"rule_ids":["FIN-01-SALE"],"decision":"REJECT","facts":["事实"],"exceptions":[],"priority":98}</memory>')
    assert memory["valid"] and memory["rule_ids"] == ["FIN-01-SALE"]
    audit = 解析审核('<audit>{"decision":"REVIEW","matched_rules":["FIN-09-RUMOR"],"evidence":["片段"],"reason":"待核"}</audit>')
    assert audit["valid"] and audit["decision"] == "REVIEW"
    assert not 解析记忆("建议拒绝")["valid"]


def test_BM25_top3覆盖全部金规则() -> None:
    """课程检索基线应召回全部单规则、例外与冲突案例的金规则。"""

    rules = 读_jsonl(数据目录 / "rules.jsonl")
    cases = 读_jsonl(数据目录 / "audit_val.jsonl")
    index = BM25索引(rules)
    for case in cases:
        predicted = {rule["rule_id"] for rule, _ in index.检索(case["content"], top_k=3, category=case["category"])}
        assert set(case["gold_rule_ids"]) <= predicted, case["case_id"]


def test_金规则执行得到全部正确处置() -> None:
    """验证规则优先级和绑定例外的确定性语义没有自相矛盾。"""

    rules = 读_jsonl(数据目录 / "rules.jsonl")
    cases = 读_jsonl(数据目录 / "audit_val.jsonl")
    by_id = {rule["rule_id"]: rule for rule in rules}
    for case in cases:
        prediction = 确定性执行([{"rule_ids": case["gold_rule_ids"]}], by_id, case["content"])
        assert prediction["decision"] == case["gold_decision"], case["case_id"]


def test_例外必须覆盖全部风险规则() -> None:
    """一个已绑定报道例外不能顺带豁免不可豁免的隐私风险。"""

    rules = 读_jsonl(数据目录 / "rules.jsonl")
    by_id = {rule["rule_id"]: rule for rule in rules}
    ids = ["POL-01-SALE", "POL-05-PRIVACY", "POL-16-REPORT"]
    prediction = 确定性执行([{"rule_ids": ids}], by_id, "付费服务、私人住址与事实报道")
    assert prediction["decision"] == "REJECT"
    assert "POL-05-PRIVACY" in prediction["reason"]


def test_数据规模拆分与泄漏() -> None:
    """检查规则、问答、案例规模及正文不含金规则编号。"""

    rules = 读_jsonl(数据目录 / "rules.jsonl")
    train = 读_jsonl(数据目录 / "memory_train.jsonl")
    val = 读_jsonl(数据目录 / "memory_val.jsonl")
    cases = 读_jsonl(数据目录 / "audit_val.jsonl")
    assert (len(rules), len(train), len(val), len(cases)) == (80, 680, 200, 120)
    assert not ({row["messages"][1]["content"] for row in train} & {row["messages"][1]["content"] for row in val})
    assert len({case["record_id"] for case in cases}) == 120
    assert all(not any(rule_id in case["content"] for rule_id in case["gold_rule_ids"]) for case in cases)


def test_裸审核_json_可解析但不算外层标签合规() -> None:
    """小 Base 漏掉外层标签时，语义指标和指令遵循指标必须分开。"""

    parsed = 解析审核('{"decision":"REJECT","matched_rules":["POL-01-SALE"],"evidence":[],"reason":"测试"}')
    assert parsed["valid"] is True
    assert parsed["wrapper_valid"] is False
    assert parsed["decision"] == "REJECT"


def test_多线索附言可以拆分() -> None:
    """例外与风险线索不能在 Grounding 时互相覆盖。"""

    content = "新闻正文：普通报道。\n\n待审核发布者附言：风险线索。 同时声明：报道例外。"
    assert 提取审核线索列表(content) == ["风险线索。", "报道例外。"]


def test_编号注册表只修正同领域唯一语义后缀() -> None:
    """注册表不能读取金标签，但可以把错误序号映射到唯一稳定 ID。"""

    rules = 读_jsonl(数据目录 / "rules.jsonl")
    by_id = {rule["rule_id"]: rule for rule in rules}
    memory = {
        "rule_ids": ["POL-04-SALE"], "decision": "REJECT", "facts": ["事实"],
        "exceptions": [], "priority": 98, "valid": True,
    }
    normalized = 规范化记忆编号(memory, by_id, "政治")
    assert normalized["raw_rule_ids"] == ["POL-04-SALE"]
    assert normalized["rule_ids"] == ["POL-01-SALE"]
