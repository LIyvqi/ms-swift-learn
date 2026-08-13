#!/usr/bin/env python3
"""CA-MeMo 的问题模板、可靠性特征、校准、验证与指标。"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


项目根目录 = Path(__file__).resolve().parents[2]
第26课目录 = 项目根目录 / "course/26_memo_rule_memory"
if str(第26课目录) not in sys.path:
    sys.path.insert(0, str(第26课目录))

from memo_core import (
    BM25索引,
    决策等级,
    决策集合,
    宏平均F1,
    确定性执行,
    解析记忆,
    规范化记忆编号,
    集合指标,
)

from inference_backend import 生成记录


特征名称 = (
    "mean_logprob",
    "has_logprobs",
    "query_id_agreement",
    "decision_agreement",
    "stage_consistency",
    "valid_id_rate",
    "fact_completeness",
    "exception_conflict",
    "semantic_entropy",
    "no_match_rate",
    "mean_rule_count",
)


def 记忆消息(question: str) -> list[dict[str, str]]:
    """构造与第26课训练模板一致、但允许显式 NO_MATCH 的消息。"""

    return [
        {
            "role": "system",
            "content": (
                "你是内容审核规则记忆模型。只根据已经记住的目标规则库回答，不审核整篇新闻。"
                "若没有任何已记住规则适用，返回空 rule_ids、空 facts 和 decision=REVIEW，禁止硬猜。"
                "严格输出 <memory>JSON</memory>，JSON 字段为 rule_ids、decision、facts、exceptions、priority。"
            ),
        },
        {"role": "user", "content": question},
    ]


def 主召回问题(case: dict[str, Any]) -> str:
    """快速主召回问题。"""

    return (
        f"领域：{case['category']}。从规则记忆中找出与下列内容最相关的规则，最多返回三条；"
        f"必须返回条件、处置和例外。若不属于记忆范围就明确 NO_MATCH。\n\n{case['audit_span']}"
    )


def 独立检查问题(case: dict[str, Any]) -> str:
    """改变提问视角，检查召回稳定性。"""

    return (
        f"不要沿用其他回答。请独立判断领域“{case['category']}”的规则库是否覆盖该片段。"
        f"若覆盖，返回规则 ID、完整事实和绑定例外；若不覆盖，返回空规则。片段：{case['audit_span']}"
    )


def 第三投票问题(case: dict[str, Any]) -> str:
    """供简单多次投票使用的第三种表述。"""

    return (
        f"反向查找规则：下面的内容条件最像哪一至三条已学习规范？不要根据领域常识创造规则。"
        f"领域：{case['category']}；内容：{case['audit_span']}"
    )


def 固定确认问题(case: dict[str, Any], first: dict[str, Any]) -> str:
    """固定三阶段协议的条件确认问题。"""

    ids = "、".join(first.get("rule_ids", [])) or "尚未识别"
    return (
        f"领域：{case['category']}。候选规则是 {ids}。确认这些规则的必要条件、处置、优先级和例外；"
        f"候选错误时请纠正。内容：{case['audit_span']}"
    )


def 固定冲突问题(case: dict[str, Any], memories: list[dict[str, Any]]) -> str:
    """固定三阶段协议的例外与冲突检查。"""

    ids = list(dict.fromkeys(rule_id for memory in memories for rule_id in memory.get("rule_ids", [])))
    return (
        f"请检查规则 {('、'.join(ids) if ids else '未知')} 的例外绑定、相互冲突和最终优先级。"
        f"只返回记忆中的事实。领域：{case['category']}；内容：{case['audit_span']}"
    )


def 主动搜索问题(case: dict[str, Any], first: dict[str, Any]) -> list[str]:
    """生成反事实、相邻边界和例外冲突三条有目的的搜索分支。"""

    ids = "、".join(first.get("rule_ids", [])) or "尚未识别"
    span = case["audit_span"]
    category = case["category"]
    return [
        (
            f"反事实检查：领域 {category} 中，如果候选 {ids} 不适用，最可能的相邻规则是什么？"
            f"逐项比较必要条件；没有规则就返回空。内容：{span}"
        ),
        (
            f"边界辨析：候选 {ids} 与容易混淆的规则之间，关键区分条件是什么？"
            f"只返回该内容真正满足的规则事实。领域：{category}；内容：{span}"
        ),
        (
            f"例外与冲突检查：内容是否同时触发允许例外、第二条风险规则或更高优先级规则？"
            f"返回全部真正命中的规则；没有就不要补造。领域：{category}；内容：{span}"
        ),
    ]


def 解析并规范化(
    record: 生成记录,
    rules_by_id: dict[str, dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    """解析 Memory 文本并保留白盒生成统计。"""

    memory = 规范化记忆编号(解析记忆(record.text), rules_by_id, category)
    memory["mean_logprob"] = record.mean_logprob
    memory["has_logprobs"] = record.has_logprobs
    memory["prompt_tokens"] = record.prompt_tokens
    memory["completion_tokens"] = record.completion_tokens
    memory["elapsed_seconds"] = record.elapsed_seconds
    memory["estimated_cost"] = record.estimated_cost
    return memory


def 集合相似度(first: list[str], second: list[str]) -> float:
    """计算 Jaccard，相同空集合记为 1。"""

    a, b = set(first), set(second)
    return len(a & b) / len(a | b) if a | b else 1.0


def 语义熵(memories: list[dict[str, Any]]) -> float:
    """按“规则集合+处置”语义簇计算归一化熵。"""

    if len(memories) <= 1:
        return 0.0
    clusters = Counter(
        (tuple(sorted(memory.get("rule_ids", []))), memory.get("decision", ""))
        for memory in memories
    )
    probabilities = [count / len(memories) for count in clusters.values()]
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    return max(0.0, entropy / math.log(len(memories)))


def 提取可靠性特征(
    records: list[生成记录],
    memories: list[dict[str, Any]],
    rules_by_id: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """提取不依赖金标签、也不相信自报 confidence 的可靠性特征。"""

    if not memories:
        raise ValueError("至少需要一个 Memory 响应")
    log_values = [record.mean_logprob for record in records if record.mean_logprob is not None]
    pair_scores = []
    decision_scores = []
    for index in range(len(memories)):
        for other in range(index + 1, len(memories)):
            pair_scores.append(集合相似度(memories[index].get("rule_ids", []), memories[other].get("rule_ids", [])))
            decision_scores.append(float(memories[index].get("decision") == memories[other].get("decision")))
    id_agreement = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
    decision_agreement = sum(decision_scores) / len(decision_scores) if decision_scores else 1.0

    raw_ids = [rule_id for memory in memories for rule_id in memory.get("raw_rule_ids", memory.get("rule_ids", []))]
    valid_ids = [rule_id for rule_id in raw_ids if rule_id in rules_by_id]
    valid_rate = len(valid_ids) / len(raw_ids) if raw_ids else 0.0

    completeness = []
    for memory in memories:
        fields = (
            bool(memory.get("rule_ids")),
            memory.get("decision") in 决策集合,
            bool(memory.get("facts")),
            isinstance(memory.get("priority"), int) and memory.get("priority", 0) > 0,
        )
        completeness.append(sum(fields) / len(fields))

    exception_presence = [bool(memory.get("exceptions")) for memory in memories]
    priority_values = [int(memory.get("priority", 0)) for memory in memories]
    exception_conflict = float(
        len(set(exception_presence)) > 1
        or len({memory.get("decision", "") for memory in memories}) > 1
        or (max(priority_values, default=0) - min(priority_values, default=0) > 15)
    )
    return {
        "mean_logprob": sum(log_values) / len(log_values) if log_values else -5.0,
        "has_logprobs": len(log_values) / len(records),
        "query_id_agreement": id_agreement,
        "decision_agreement": decision_agreement,
        "stage_consistency": 0.5 * id_agreement + 0.5 * decision_agreement,
        "valid_id_rate": valid_rate,
        "fact_completeness": sum(completeness) / len(completeness),
        "exception_conflict": exception_conflict,
        "semantic_entropy": 语义熵(memories),
        "no_match_rate": sum(not memory.get("rule_ids") for memory in memories) / len(memories),
        "mean_rule_count": sum(len(memory.get("rule_ids", [])) for memory in memories) / len(memories),
    }


def 特征向量(features: dict[str, float]) -> np.ndarray:
    """按固定字段顺序转换特征。"""

    return np.asarray([float(features[name]) for name in 特征名称], dtype=np.float64)


@dataclass
class 逻辑校准器:
    """不依赖 sklearn 的带 L2 二分类逻辑校准器。"""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    @classmethod
    def 拟合(
        cls,
        rows: list[dict[str, float]],
        labels: list[int],
        learning_rate: float = 0.04,
        steps: int = 4000,
        l2: float = 0.03,
    ) -> "逻辑校准器":
        """用类别均衡权重拟合校准概率。"""

        x = np.stack([特征向量(row) for row in rows])
        y = np.asarray(labels, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        z = (x - mean) / scale
        weights = np.zeros(z.shape[1], dtype=np.float64)
        positive = max(1.0, y.sum())
        negative = max(1.0, len(y) - y.sum())
        sample_weight = np.where(y > 0.5, len(y) / (2 * positive), len(y) / (2 * negative))
        base_rate = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        bias = math.log(base_rate / (1 - base_rate))
        for _ in range(steps):
            logits = np.clip(z @ weights + bias, -30, 30)
            probabilities = 1 / (1 + np.exp(-logits))
            error = (probabilities - y) * sample_weight
            gradient = z.T @ error / len(y) + l2 * weights
            bias_gradient = float(error.mean())
            weights -= learning_rate * gradient
            bias -= learning_rate * bias_gradient
        return cls(mean=mean, scale=scale, weights=weights, bias=bias)

    def 预测(self, features: dict[str, float]) -> float:
        """输出经校准的“首轮规则与处置均正确”概率。"""

        z = (特征向量(features) - self.mean) / self.scale
        logit = float(np.clip(z @ self.weights + self.bias, -30, 30))
        return 1 / (1 + math.exp(-logit))

    def 转字典(self) -> dict[str, Any]:
        """导出模型，便于复现实验。"""

        return {
            "feature_names": list(特征名称),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }


def 聚合记忆(memories: list[dict[str, Any]], minimum_votes: int | None = None) -> dict[str, Any]:
    """按规则 ID 投票聚合；平票时保留最高票规则。"""

    counts = Counter(rule_id for memory in memories for rule_id in set(memory.get("rule_ids", [])))
    if minimum_votes is None:
        minimum_votes = max(1, math.ceil(len(memories) / 2))
    selected = [rule_id for rule_id, count in counts.items() if count >= minimum_votes]
    if not selected and counts:
        best = max(counts.values())
        selected = [rule_id for rule_id, count in counts.items() if count == best]
    decisions = [memory.get("decision", "") for memory in memories if memory.get("decision") in 决策集合]
    facts = [fact for memory in memories for fact in memory.get("facts", [])]
    exceptions = [item for memory in memories for item in memory.get("exceptions", [])]
    return {
        "rule_ids": sorted(selected),
        "decision": Counter(decisions).most_common(1)[0][0] if decisions else "REVIEW",
        "facts": list(dict.fromkeys(facts)),
        "exceptions": list(dict.fromkeys(exceptions)),
        "priority": max((int(memory.get("priority", 0)) for memory in memories), default=0),
        "valid": bool(selected and facts),
        "vote_counts": dict(counts),
    }


def 首轮是否完全正确(case: dict[str, Any], memory: dict[str, Any], rules_by_id: dict[str, dict[str, Any]]) -> bool:
    """生成校准标签：规则集合与确定性处置必须同时正确。"""

    prediction = 确定性执行([memory], rules_by_id, case["audit_span"])
    return (
        set(memory.get("rule_ids", [])) == set(case["gold_rule_ids"])
        and prediction["decision"] == case["gold_decision"]
    )


def 选择路由阈值(probabilities: list[float], labels: list[int]) -> dict[str, float]:
    """从独立校准集选择高/中/低路由阈值。

    高阈值优先满足自动接受精度 90%；若数据太难，则退化到 75% 分位点。
    低阈值取 30% 分位点，从而保留可观察的中置信搜索区间。
    """

    values = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    candidates = sorted(set(values.tolist()))
    high = None
    for threshold in candidates:
        mask = values >= threshold
        if mask.sum() >= max(5, math.ceil(len(values) * 0.1)) and y[mask].mean() >= 0.9:
            high = threshold
            break
    if high is None:
        high = float(np.quantile(values, 0.75))
    low = float(np.quantile(values, 0.30))
    if low >= high:
        low = float(np.quantile(values, 0.20))
        high = float(np.quantile(values, 0.80))
    return {"low": low, "high": high}


def 处置概率(memories: list[dict[str, Any]], smoothing: float = 0.15) -> dict[str, float]:
    """由多询问频率构造三类处置非一致性分数。"""

    counts = Counter(memory.get("decision") for memory in memories if memory.get("decision") in 决策集合)
    denominator = sum(counts.values()) + smoothing * len(决策集合)
    return {label: (counts[label] + smoothing) / denominator for label in 决策集合}


def 拟合共形阈值(
    decision_probabilities: list[dict[str, float]],
    gold: list[str],
    alpha: float = 0.1,
) -> float:
    """按有限样本修正分位数拟合 split-conformal 阈值。"""

    scores = sorted(1 - probabilities[label] for probabilities, label in zip(decision_probabilities, gold))
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1 - alpha)))
    return float(scores[max(0, rank - 1)])


def 共形处置集合(probabilities: dict[str, float], threshold: float) -> list[str]:
    """返回满足非一致性阈值的候选处置集合。"""

    selected = [label for label in 决策集合 if 1 - probabilities[label] <= threshold]
    return selected or [max(probabilities, key=probabilities.get)]


def 拆分审核片段(span: str) -> list[str]:
    """拆分多规则和绑定例外片段，供独立权威检索逐项验证。"""

    return [
        part.strip()
        for part in re.split(r"\s*(?:同时还有另一条线索|同时声明例外背景)[：:]\s*", span)
        if part.strip()
    ] or [span]


def 权威检索候选(case: dict[str, Any], index: BM25索引, threshold: float) -> tuple[list[str], list[dict[str, Any]]]:
    """从原始规则源逐片检索，不读取案例 gold 字段。"""

    selected: list[str] = []
    diagnostics = []
    for span in 拆分审核片段(case["audit_span"]):
        ranked = index.检索(span, top_k=3, category=case["category"])
        best_score = float(ranked[0][1]) if ranked else 0.0
        chosen = ranked[0][0]["rule_id"] if ranked and best_score >= threshold else None
        if chosen:
            selected.append(chosen)
        diagnostics.append({
            "span": span,
            "best_score": best_score,
            "chosen_rule_id": chosen,
            "top3": [{"rule_id": rule["rule_id"], "score": float(score)} for rule, score in ranked],
        })
    return list(dict.fromkeys(selected)), diagnostics


def 拟合权威检索阈值(cases: list[dict[str, Any]], index: BM25索引) -> float:
    """只用校准集标签选择 OOD/有规则的 BM25 分界。"""

    scores = []
    labels = []
    for case in cases:
        fragment_scores = []
        for span in 拆分审核片段(case["audit_span"]):
            ranked = index.检索(span, top_k=1, category=case["category"])
            fragment_scores.append(float(ranked[0][1]) if ranked else 0.0)
        scores.append(max(fragment_scores, default=0.0))
        labels.append(not case.get("is_ood", False))
    candidates = sorted(set([0.0, *scores]))
    best = (float("-inf"), 0.0)
    for threshold in candidates:
        predictions = [score >= threshold and threshold > 0 for score in scores]
        tp = sum(pred and label for pred, label in zip(predictions, labels))
        tn = sum(not pred and not label for pred, label in zip(predictions, labels))
        positives = max(1, sum(labels))
        negatives = max(1, len(labels) - sum(labels))
        balanced_accuracy = 0.5 * (tp / positives + tn / negatives)
        if balanced_accuracy > best[0]:
            best = (balanced_accuracy, threshold)
    return float(best[1])


def 硬验证(memory: dict[str, Any], rules_by_id: dict[str, dict[str, Any]], category: str) -> dict[str, Any]:
    """验证结构、ID、领域、事实和例外自洽性。"""

    ids = memory.get("rule_ids", [])
    checks = {
        "format_valid": bool(memory.get("valid")),
        "all_ids_known": bool(ids) and all(rule_id in rules_by_id for rule_id in ids),
        "category_match": bool(ids) and all(rules_by_id.get(rule_id, {}).get("category") == category for rule_id in ids),
        "facts_present": bool(memory.get("facts")),
        "decision_valid": memory.get("decision") in 决策集合,
    }
    return {"passed": all(checks.values()), "checks": checks}


def 独立权威验证(
    case: dict[str, Any],
    memory: dict[str, Any],
    rules_by_id: dict[str, dict[str, Any]],
    index: BM25索引,
    threshold: float,
) -> dict[str, Any]:
    """组合硬验证与权威规则源检索，并允许慢路径纠正候选。"""

    hard = 硬验证(memory, rules_by_id, case["category"])
    authority_ids, diagnostics = 权威检索候选(case, index, threshold)
    memory_ids = set(memory.get("rule_ids", []))
    authority_set = set(authority_ids)
    agreement = len(memory_ids & authority_set) / len(memory_ids | authority_set) if memory_ids | authority_set else 1.0
    return {
        "passed": hard["passed"] and agreement >= 0.5,
        "hard": hard,
        "authority_rule_ids": authority_ids,
        "memory_authority_agreement": agreement,
        "retrieval": diagnostics,
        "corrected": authority_set != memory_ids,
    }


def 大模型裁判消息(
    case: dict[str, Any],
    candidate_rule_ids: list[str],
    rules_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """给可选 API 裁判提供候选与少量权威规则，不暴露完整规则库。"""

    policies = [rules_by_id[rule_id] for rule_id in candidate_rule_ids if rule_id in rules_by_id]
    compact = [
        {
            "rule_id": rule["rule_id"],
            "decision": rule["decision"],
            "priority": rule["priority"],
            "conditions": rule["conditions"],
            "exceptions": rule["exceptions"],
            "exception_rule_ids": rule.get("exception_rule_ids", []),
        }
        for rule in policies
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是独立内容政策裁判。只依据给定权威规则检查候选，不使用外部常识补造规则。"
                "严格输出 JSON：accepted、rule_ids、decision、reason。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"领域：{case['category']}\n内容：{case['audit_span']}\n"
                f"候选规则：{json.dumps(candidate_rule_ids, ensure_ascii=False)}\n"
                f"权威规则：{json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]


def 解析大模型裁判(text: str) -> dict[str, Any]:
    """容忍代码围栏，但只提取结构化裁判字段。"""

    decoder = json.JSONDecoder()
    payload: dict[str, Any] = {}
    for match in re.finditer(r"\{", str(text)):
        try:
            candidate, _ = decoder.raw_decode(str(text)[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "accepted" in candidate:
            payload = candidate
            break
    rule_ids = [value for value in payload.get("rule_ids", []) if isinstance(value, str)]
    decision = str(payload.get("decision", "")).upper()
    return {
        "accepted": bool(payload.get("accepted")) and decision in 决策集合,
        "rule_ids": rule_ids,
        "decision": decision if decision in 决策集合 else "REVIEW",
        "reason": str(payload.get("reason", "")),
        "valid": bool(payload),
    }


def 验证大模型裁判(
    judge: dict[str, Any],
    allowed_rule_ids: list[str],
    rules_by_id: dict[str, dict[str, Any]],
    audit_span: str,
) -> dict[str, Any]:
    """硬检查 API 裁判的 ID 边界与处置一致性。"""

    judge_ids = list(dict.fromkeys(judge.get("rule_ids", [])))
    known = bool(judge_ids) and all(rule_id in rules_by_id for rule_id in judge_ids)
    allowed = known and set(judge_ids) <= set(allowed_rule_ids)
    prediction = 确定性执行([{"rule_ids": judge_ids}], rules_by_id, audit_span)
    decision_consistent = prediction["decision"] == judge.get("decision")
    checks = {
        "structure_valid": bool(judge.get("valid")),
        "judge_accepted": bool(judge.get("accepted")),
        "all_ids_known": known,
        "ids_within_candidates": allowed,
        "decision_consistent": decision_consistent,
    }
    return {"passed": all(checks.values()), "checks": checks, "prediction": prediction}


def 调用统计(memories: list[dict[str, Any]]) -> dict[str, float]:
    """汇总某条轨迹真实使用的生成资源。"""

    return {
        "memory_calls": float(len(memories)),
        "judge_calls": 0.0,
        "prompt_tokens": float(sum(memory.get("prompt_tokens", 0) for memory in memories)),
        "completion_tokens": float(sum(memory.get("completion_tokens", 0) for memory in memories)),
        "elapsed_seconds": float(sum(memory.get("elapsed_seconds", 0.0) for memory in memories)),
        "authority_seconds": 0.0,
        "estimated_cost": float(sum(memory.get("estimated_cost", 0.0) for memory in memories)),
    }


def ECE(confidence: list[float], correct: list[int], bins: int = 10) -> float:
    """计算等宽 Expected Calibration Error。"""

    total = len(confidence)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [i for i, score in enumerate(confidence) if low <= score < high or index == bins - 1 and score == 1]
        if not selected:
            continue
        mean_conf = sum(confidence[i] for i in selected) / len(selected)
        accuracy = sum(correct[i] for i in selected) / len(selected)
        value += len(selected) / total * abs(mean_conf - accuracy)
    return value


def AURC(confidence: list[float], correct: list[int]) -> float:
    """按置信度排序，计算离散风险—覆盖率曲线面积。"""

    order = sorted(range(len(confidence)), key=lambda index: (-confidence[index], index))
    errors = 0
    risks = []
    for rank, index in enumerate(order, 1):
        errors += 1 - correct[index]
        risks.append(errors / rank)
    return sum(risks) / len(risks)


def 汇总方法(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总决策、校准、拒答、搜索、成本和场景指标。"""

    gold = [trace["gold_decision"] for trace in traces]
    predicted = [trace["prediction"]["decision"] for trace in traces]
    correct = [int(g == p) for g, p in zip(gold, predicted)]
    complete_correct = [
        int(
            trace["prediction"]["decision"] == trace["gold_decision"]
            and set(trace["prediction"]["matched_rules"]) == set(trace["gold_rule_ids"])
        )
        for trace in traces
    ]
    calibration_target = [
        int(trace.get("confidence_target", complete))
        for trace, complete in zip(traces, complete_correct)
    ]
    confidence = [float(trace["confidence"]) for trace in traces]
    accepted = [bool(trace["accepted"]) for trace in traces]
    rule_scores = [集合指标(trace["prediction"]["matched_rules"], trace["gold_rule_ids"]) for trace in traces]
    accepted_indices = [index for index, value in enumerate(accepted) if value]
    ood = [trace for trace in traces if trace.get("is_ood")]

    base_wrong = [trace for trace in traces if not trace.get("base_correct", False)]
    rescued = [
        trace for trace in base_wrong
        if (
            trace.get("searched")
            and trace["accepted"]
            and trace["prediction"]["decision"] == trace["gold_decision"]
            and set(trace["prediction"]["matched_rules"]) == set(trace["gold_rule_ids"])
        )
    ]
    candidate_has_gold = [trace for trace in traces if trace.get("candidate_contains_gold")]
    picked = [
        trace for trace in candidate_has_gold
        if set(trace["prediction"]["matched_rules"]) == set(trace["gold_rule_ids"])
    ]

    scenario_metrics = {}
    for scenario in sorted({trace["scenario_type"] for trace in traces}):
        subset = [trace for trace in traces if trace["scenario_type"] == scenario]
        scenario_metrics[scenario] = {
            "samples": len(subset),
            "accuracy": sum(trace["prediction"]["decision"] == trace["gold_decision"] for trace in subset) / len(subset),
            "coverage": sum(bool(trace["accepted"]) for trace in subset) / len(subset),
        }

    conformal = [trace for trace in traces if trace.get("conformal_set") is not None]
    return {
        "method": traces[0]["method"],
        "samples": len(traces),
        "decision_accuracy": sum(correct) / len(correct),
        "complete_accuracy": sum(complete_correct) / len(complete_correct),
        "decision_macro_f1": 宏平均F1(gold, predicted),
        "rule_precision": sum(value[0] for value in rule_scores) / len(rule_scores),
        "rule_recall": sum(value[1] for value in rule_scores) / len(rule_scores),
        "rule_f1": sum(value[2] for value in rule_scores) / len(rule_scores),
        "ece": ECE(confidence, calibration_target),
        "brier": sum((score - label) ** 2 for score, label in zip(confidence, calibration_target)) / len(calibration_target),
        "coverage": sum(accepted) / len(accepted),
        "selective_risk": (
            sum(1 - correct[index] for index in accepted_indices) / len(accepted_indices)
            if accepted_indices else None
        ),
        "aurc": AURC(confidence, calibration_target),
        "ood_false_accept_rate": (
            sum(bool(trace["accepted"]) for trace in ood) / len(ood) if ood else None
        ),
        "search_rescue_rate": len(rescued) / len(base_wrong) if base_wrong else None,
        "pick_at_n": len(picked) / len(candidate_has_gold) if candidate_has_gold else None,
        "mean_memory_calls": sum(trace["resource"]["memory_calls"] for trace in traces) / len(traces),
        "mean_judge_calls": sum(trace["resource"].get("judge_calls", 0.0) for trace in traces) / len(traces),
        "mean_prompt_tokens": sum(trace["resource"]["prompt_tokens"] for trace in traces) / len(traces),
        "mean_completion_tokens": sum(trace["resource"]["completion_tokens"] for trace in traces) / len(traces),
        "mean_latency_seconds": sum(trace["resource"]["elapsed_seconds"] for trace in traces) / len(traces),
        "mean_authority_seconds": sum(trace["resource"].get("authority_seconds", 0.0) for trace in traces) / len(traces),
        "total_estimated_api_cost": sum(trace["resource"]["estimated_cost"] for trace in traces),
        "conformal_empirical_coverage": (
            sum(trace["gold_decision"] in trace["conformal_set"] for trace in conformal) / len(conformal)
            if conformal else None
        ),
        "conformal_mean_set_size": (
            sum(len(trace["conformal_set"]) for trace in conformal) / len(conformal) if conformal else None
        ),
        "scenario_metrics": scenario_metrics,
    }
