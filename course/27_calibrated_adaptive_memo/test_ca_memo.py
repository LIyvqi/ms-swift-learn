#!/usr/bin/env python3
"""CA-MeMo 数据、校准、共形集合、验证器与黑盒特征测试。"""

from __future__ import annotations

import json
from pathlib import Path

from ca_memo_core import (
    BM25索引,
    共形处置集合,
    拟合共形阈值,
    拟合权威检索阈值,
    权威检索候选,
    独立权威验证,
    解析大模型裁判,
    验证大模型裁判,
    逻辑校准器,
    硬验证,
    提取可靠性特征,
)
from inference_backend import 生成记录


项目根目录 = Path(__file__).resolve().parents[2]


def 读_jsonl(path: Path) -> list[dict]:
    """读取测试数据。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 模拟记录(logprob: float | None = None) -> 生成记录:
    """构造不依赖模型加载的生成统计。"""

    return 生成记录(
        text="", mean_logprob=logprob, prompt_tokens=10, completion_tokens=5,
        elapsed_seconds=0.1, estimated_cost=0.0, has_logprobs=logprob is not None,
    )


def 模拟记忆(rule_ids: list[str], decision: str = "REJECT") -> dict:
    """构造解析后的 Memory。"""

    return {
        "rule_ids": rule_ids,
        "raw_rule_ids": rule_ids,
        "decision": decision,
        "facts": ["规则事实"] if rule_ids else [],
        "exceptions": [],
        "priority": 90 if rule_ids else 0,
        "valid": bool(rule_ids),
    }


def test_严格数据拆分且六场景平衡() -> None:
    """校准和测试不得共享新闻或审核片段。"""

    root = 项目根目录 / "datasets/calibrated_adaptive_memo"
    calibration = 读_jsonl(root / "calibration.jsonl")
    test = 读_jsonl(root / "test.jsonl")
    assert len(calibration) == len(test) == 72
    assert not ({row["record_id"] for row in calibration} & {row["record_id"] for row in test})
    assert not ({row["audit_span"] for row in calibration} & {row["audit_span"] for row in test})
    expected = {
        "standard", "adjacent_boundary", "ood_no_rule", "multi_rule_conflict",
        "bound_exception", "adversarial_rewrite",
    }
    assert {row["scenario_type"] for row in calibration} == expected
    assert {row["scenario_type"] for row in test} == expected
    assert all(not any(rule_id in row["content"] for rule_id in row["gold_rule_ids"]) for row in calibration + test)


def test_无logprobs黑盒仍能形成完整特征() -> None:
    """API 不返回概率时使用缺省值和一致性特征。"""

    rules = 读_jsonl(项目根目录 / "datasets/memo_rule_memory/rules.jsonl")
    by_id = {rule["rule_id"]: rule for rule in rules}
    memories = [模拟记忆(["FIN-01-SALE"]), 模拟记忆(["FIN-01-SALE"])]
    features = 提取可靠性特征([模拟记录(), 模拟记录()], memories, by_id)
    assert features["has_logprobs"] == 0.0
    assert features["mean_logprob"] == -5.0
    assert features["query_id_agreement"] == 1.0
    assert features["semantic_entropy"] == 0.0


def test_逻辑校准器概率有限且能区分样本() -> None:
    """逻辑校准器在小型可分数据上应给正例更高概率。"""

    base = {
        "mean_logprob": -1.0, "has_logprobs": 1.0, "query_id_agreement": 0.0,
        "decision_agreement": 0.0, "stage_consistency": 0.0, "valid_id_rate": 0.0,
        "fact_completeness": 0.25, "exception_conflict": 1.0, "semantic_entropy": 1.0,
        "no_match_rate": 1.0, "mean_rule_count": 0.0,
    }
    positive = {**base, "mean_logprob": -0.01, "query_id_agreement": 1.0,
                "decision_agreement": 1.0, "stage_consistency": 1.0,
                "valid_id_rate": 1.0, "fact_completeness": 1.0,
                "exception_conflict": 0.0, "semantic_entropy": 0.0,
                "no_match_rate": 0.0, "mean_rule_count": 1.0}
    calibrator = 逻辑校准器.拟合([base, positive, base, positive], [0, 1, 0, 1], steps=1200)
    bad = calibrator.预测(base)
    good = calibrator.预测(positive)
    assert 0.0 < bad < good < 1.0


def test_共形集合包含校准金标签() -> None:
    """有限样本修正阈值应覆盖给定校准标签。"""

    probabilities = [
        {"PASS": 0.8, "REVIEW": 0.1, "REJECT": 0.1},
        {"PASS": 0.1, "REVIEW": 0.8, "REJECT": 0.1},
        {"PASS": 0.1, "REVIEW": 0.1, "REJECT": 0.8},
    ]
    gold = ["PASS", "REVIEW", "REJECT"]
    threshold = 拟合共形阈值(probabilities, gold, alpha=0.1)
    assert all(label in 共形处置集合(row, threshold) for row, label in zip(probabilities, gold))


def test_硬验证器拒绝未知编号() -> None:
    """不存在的生成式规则 ID 不能通过硬验证。"""

    rules = 读_jsonl(项目根目录 / "datasets/memo_rule_memory/rules.jsonl")
    by_id = {rule["rule_id"]: rule for rule in rules}
    result = 硬验证(模拟记忆(["FIN-99-NOT_EXISTS"]), by_id, "财经")
    assert not result["passed"]
    assert not result["checks"]["all_ids_known"]


def test_权威检索慢路径不读取金标签() -> None:
    """通过移除 gold 字段验证慢路径只使用内容、类别和规则库。"""

    rules = 读_jsonl(项目根目录 / "datasets/memo_rule_memory/rules.jsonl")
    calibration = 读_jsonl(项目根目录 / "datasets/calibrated_adaptive_memo/calibration.jsonl")
    index = BM25索引(rules)
    threshold = 拟合权威检索阈值(calibration, index)
    case = {
        "category": "财经",
        "audit_span": "内容换了一种说法：无牌照高收益理财产品。请核对条件和例外。",
    }
    rule_ids, _ = 权威检索候选(case, index, threshold)
    assert "FIN-01-SALE" in rule_ids


def test_独立验证器能够纠正错误Memory候选() -> None:
    """权威规则源与同一 Memory 参数独立，可以暴露错误候选。"""

    rules = 读_jsonl(项目根目录 / "datasets/memo_rule_memory/rules.jsonl")
    by_id = {rule["rule_id"]: rule for rule in rules}
    calibration = 读_jsonl(项目根目录 / "datasets/calibrated_adaptive_memo/calibration.jsonl")
    index = BM25索引(rules)
    threshold = 拟合权威检索阈值(calibration, index)
    case = {
        "category": "财经",
        "audit_span": "待核查内容称，无牌照高收益理财产品。请按现行规则处理。",
    }
    wrong = 模拟记忆(["FIN-04-FORGE"])
    result = 独立权威验证(case, wrong, by_id, index, threshold)
    assert result["corrected"]
    assert "FIN-01-SALE" in result["authority_rule_ids"]


def test_API裁判只接受合法结构() -> None:
    """可选裁判必须返回结构化字段，普通自然语言不能被误判为通过。"""

    good = 解析大模型裁判(
        '```json\n{"accepted":true,"rule_ids":["FIN-01-SALE"],"decision":"REJECT","reason":"条件成立"}\n```'
    )
    bad = 解析大模型裁判("我认为大概可以通过，但没有按协议返回。")
    assert good["valid"] and good["accepted"]
    assert good["rule_ids"] == ["FIN-01-SALE"]
    assert not bad["valid"] and not bad["accepted"]


def test_API裁判不能绕过候选ID和处置硬检查() -> None:
    """即使裁判 JSON 合法，陌生 ID 或错误处置也不能自动接受。"""

    rules = 读_jsonl(项目根目录 / "datasets/memo_rule_memory/rules.jsonl")
    by_id = {rule["rule_id"]: rule for rule in rules}
    valid = {
        "valid": True, "accepted": True, "rule_ids": ["FIN-01-SALE"],
        "decision": "REJECT", "reason": "条件成立",
    }
    unknown = {**valid, "rule_ids": ["FIN-99-NOT_EXISTS"]}
    inconsistent = {**valid, "decision": "PASS"}
    assert 验证大模型裁判(
        valid, ["FIN-01-SALE"], by_id, "无牌照高收益理财产品"
    )["passed"]
    assert not 验证大模型裁判(
        unknown, ["FIN-01-SALE"], by_id, "无牌照高收益理财产品"
    )["passed"]
    assert not 验证大模型裁判(
        inconsistent, ["FIN-01-SALE"], by_id, "无牌照高收益理财产品"
    )["passed"]
