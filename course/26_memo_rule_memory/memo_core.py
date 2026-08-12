#!/usr/bin/env python3
"""MeMo 规则记忆课程共享的解析、检索、提示和评测工具。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


记忆模式 = re.compile(r"<memory>\s*(\{.*?\})\s*</memory>", re.DOTALL | re.IGNORECASE)
审核模式 = re.compile(r"<audit>\s*(\{.*?\})\s*</audit>", re.DOTALL | re.IGNORECASE)
规则编号模式 = re.compile(r"\b(?:POL|FIN|SPT|TEC)-\d{2}-[A-Z_]+\b")
决策集合 = ("PASS", "REVIEW", "REJECT")
决策等级 = {"PASS": 0, "REVIEW": 1, "REJECT": 2}


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 记录。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 解析记忆(text: str) -> dict[str, Any]:
    """容忍空 thinking 前缀，但只接受合法的 memory JSON。"""

    match = 记忆模式.search(str(text))
    if match is None:
        return {"rule_ids": [], "decision": "", "facts": [], "exceptions": [], "priority": 0, "valid": False}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"rule_ids": [], "decision": "", "facts": [], "exceptions": [], "priority": 0, "valid": False}
    rule_ids = [value for value in payload.get("rule_ids", []) if isinstance(value, str)]
    decision = str(payload.get("decision", "")).upper()
    facts = [str(value) for value in payload.get("facts", [])]
    exceptions = [str(value) for value in payload.get("exceptions", [])]
    try:
        priority = int(payload.get("priority", 0))
    except (TypeError, ValueError):
        priority = 0
    valid = bool(rule_ids and decision in 决策集合 and facts)
    return {
        "rule_ids": rule_ids,
        "decision": decision,
        "facts": facts,
        "exceptions": exceptions,
        "priority": priority,
        "valid": valid,
    }


def 规范化记忆编号(memory: dict[str, Any], rules_by_id: dict[str, dict[str, Any]], category: str) -> dict[str, Any]:
    """用稳定 ID 注册表修正“语义后缀正确、数字序号错误”的生成式编号。

    该步骤只查看规则 ID，不读取规则正文、条件或金标准标签。线上可把它实现为
    独立的规则 ID/别名注册服务。原始编号保存在 `raw_rule_ids`，便于如实评测。
    """

    result = dict(memory)
    raw_ids = list(memory.get("rule_ids", []))
    result["raw_rule_ids"] = raw_ids
    category_ids = [rule_id for rule_id, rule in rules_by_id.items() if rule["category"] == category]
    normalized = []
    for raw_id in raw_ids:
        if raw_id in rules_by_id and rules_by_id[raw_id]["category"] == category:
            normalized.append(raw_id)
            continue
        parts = raw_id.upper().split("-", 2)
        if len(parts) != 3:
            continue
        slug = parts[2].strip("_-")
        candidates = []
        for candidate in category_ids:
            candidate_slug = candidate.split("-", 2)[2]
            if candidate_slug == slug or candidate_slug.startswith(slug + "_") or slug.startswith(candidate_slug + "_"):
                candidates.append(candidate)
        if len(candidates) == 1:
            normalized.append(candidates[0])
    result["rule_ids"] = list(dict.fromkeys(normalized))
    result["id_resolution_changed"] = result["rule_ids"] != raw_ids
    return result


def 解析审核(text: str) -> dict[str, Any]:
    """解析 Executive 的审核 JSON，并区分 JSON 合规与外层标签合规。"""

    source = str(text)
    match = 审核模式.search(source)
    wrapper_valid = match is not None
    if match is not None:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
        # 0.8B Base 经常输出合法裸 JSON，但漏掉 <audit> 外层标签。这里接受 JSON，
        # 同时用 wrapper_valid 单独记录指令遵循，避免把语义解析失败混为一谈。
        decoder = json.JSONDecoder()
        for found in re.finditer(r"\{", source):
            try:
                candidate, _ = decoder.raw_decode(source[found.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "decision" in candidate:
                payload = candidate
                break
    decision = str(payload.get("decision", "")).upper()
    if decision not in 决策集合:
        found = re.search(r"\b(PASS|REVIEW|REJECT)\b", source.upper())
        decision = found.group(1) if found else ""
    rule_ids = [value for value in payload.get("matched_rules", []) if isinstance(value, str)]
    if not rule_ids:
        rule_ids = list(dict.fromkeys(规则编号模式.findall(source)))
    evidence = [str(value) for value in payload.get("evidence", [])]
    reason = str(payload.get("reason", ""))
    return {
        "decision": decision,
        "matched_rules": rule_ids,
        "evidence": evidence,
        "reason": reason,
        "valid": bool(payload and decision in 决策集合),
        "wrapper_valid": wrapper_valid,
    }


def 中文词元(text: str) -> list[str]:
    """组合英文词、连续中文词和中文二元组，避免依赖额外分词包。"""

    lowered = str(text).lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    for block in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.append(block)
        tokens.extend(block[index:index + 2] for index in range(max(0, len(block) - 1)))
    return tokens


@dataclass
class BM25索引:
    """适用于小型规则库的纯 Python BM25。"""

    rules: list[dict[str, Any]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.documents = [中文词元(self._规则文本(rule)) for rule in self.rules]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        frequencies = Counter(token for document in self.documents for token in set(document))
        total = len(self.documents)
        self.idf = {
            token: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for token, count in frequencies.items()
        }
        self.term_frequencies = [Counter(document) for document in self.documents]

    @staticmethod
    def _规则文本(rule: dict[str, Any]) -> str:
        return " ".join(
            [
                rule["rule_id"], rule["title"], rule["category"], rule["text"],
                *rule.get("conditions", []), *rule.get("exceptions", []),
                *rule.get("keywords", []), *rule.get("train_cues", []),
            ]
        )

    def 检索(self, query: str, top_k: int = 5, category: str | None = None) -> list[tuple[dict[str, Any], float]]:
        """返回按得分降序排列的规则。"""

        query_tokens = 中文词元(query)
        scores = []
        for rule, frequencies, length in zip(self.rules, self.term_frequencies, self.lengths):
            if category and rule["category"] != category:
                continue
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            scores.append((rule, score))
        scores.sort(key=lambda pair: (-pair[1], -pair[0]["priority"], pair[0]["rule_id"]))
        return scores[:top_k]


def 规则上下文(rules: list[dict[str, Any]], compact: bool = False) -> str:
    """把规则转换为 Executive 可读上下文。"""

    lines = []
    for rule in sorted(rules, key=lambda item: (item["category"], item["rule_id"])):
        if compact:
            lines.append(f"{rule['rule_id']}|{rule['decision']}|P{rule['priority']}|{rule['text']}")
        else:
            lines.append(json.dumps({
                "rule_id": rule["rule_id"], "title": rule["title"],
                "decision": rule["decision"], "priority": rule["priority"],
                "conditions": rule["conditions"], "exceptions": rule["exceptions"],
                "text": rule["text"],
            }, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def 记忆消息(question: str) -> list[dict[str, str]]:
    """构造与 SFT 一致的 Memory 单轮消息。"""

    return [
        {
            "role": "system",
            "content": "你是内容审核规则记忆模型。只根据已经记住的目标规则库回答，不审核整篇新闻。严格输出 <memory>JSON</memory>，JSON 字段为 rule_ids、decision、facts、exceptions、priority。",
        },
        {"role": "user", "content": question},
    ]


def 提取审核线索(content: str, mode: str = "span") -> str:
    """从长新闻中取得待审核发布片段，不读取金标准规则或标签。

    真实系统通常由上游内容解析器提供标题、评论、附言等字段。本课程数据把
    新闻正文与发布者附言分成两个段落，因此默认取最后一个非空段；没有结构
    标记时退化为末尾 500 字。`full` 仅用于展示长正文干扰的消融实验。
    """

    text = str(content).strip()
    if mode == "full":
        return text
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    grounded = paragraphs[-1] if len(paragraphs) > 1 else text[-500:]
    return re.sub(r"^待审核发布者附言[：:]\s*", "", grounded).strip()


def 提取审核线索列表(content: str, mode: str = "span") -> list[str]:
    """把同一附言内的多条独立线索拆开，避免小 Memory 只保留其中一条。"""

    grounded = 提取审核线索(content, mode)
    if mode == "full":
        return [grounded]
    parts = [
        part.strip()
        for part in re.split(r"\s*(?:同时声明|同时还写道)[：:]\s*", grounded)
        if part.strip()
    ]
    return parts or [grounded]


def 单轮记忆问题(case: dict[str, Any], grounding_mode: str = "span") -> str:
    """让 Memory 根据 Grounding 后的内容片段反查最相关规则。"""

    return (
        f"领域：{case['category']}。从规则记忆中找出与下列待审核内容最相关的规则，最多返回三条。"
        f"必须给出规则事实和例外，不要自行创造规则。\n\n{提取审核线索(case['content'], grounding_mode)}"
    )


def 确认记忆问题(case: dict[str, Any], first: dict[str, Any], grounding_mode: str = "span") -> str:
    """第二阶段确认候选规则的完整条件、优先级和例外。"""

    candidates = first.get("rule_ids") or []
    candidate_text = "、".join(candidates) if candidates else "尚未识别"
    return (
        f"领域：{case['category']}。候选规则为 {candidate_text}。请从记忆中确认这些规则的条件、处置、"
        f"优先级和绑定例外；若候选不准确，请返回更相关的规则。"
        f"内容线索：{提取审核线索(case['content'], grounding_mode)}"
    )


def 冲突记忆问题(case: dict[str, Any], memories: list[dict[str, Any]], grounding_mode: str = "span") -> str:
    """第三阶段针对例外与多规则冲突做定向回忆。"""

    ids = list(dict.fromkeys(rule_id for memory in memories for rule_id in memory.get("rule_ids", [])))
    return (
        f"请检查规则 {('、'.join(ids) if ids else '未知')} 是否存在例外绑定或优先级冲突。"
        f"仅返回记忆中的相关规则事实。领域：{case['category']}；"
        f"内容线索：{提取审核线索(case['content'], grounding_mode)}"
    )


def Executive消息(case: dict[str, Any], knowledge: str, method: str) -> list[dict[str, str]]:
    """构造冻结 Executive 的最终决策提示。"""

    return [
        {
            "role": "system",
            "content": (
                "你是新闻内容审核执行模型。只能依据给定政策上下文判断。处置只有 PASS、REVIEW、REJECT。"
                "例外必须与基础规则绑定；冲突时先判断例外，再按风险与优先级选择处置。严格输出 "
                "<audit>{\"decision\":\"...\",\"matched_rules\":[\"...\"],"
                "\"evidence\":[\"原文片段\"],\"reason\":\"...\"}</audit>，不要输出其他文字。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"方法：{method}\n领域：{case['category']}\n待审核内容：\n{case['content']}\n\n"
                f"政策上下文：\n{knowledge if knowledge else '没有可用政策记忆，请仅依据常识判断。'}"
            ),
        },
    ]


def 确定性执行(memory_payloads: list[dict[str, Any]], rules_by_id: dict[str, dict[str, Any]], content: str) -> dict[str, Any]:
    """把 Memory 返回的规则事实转换为可审计的线上兜底决策。"""

    rule_ids = list(dict.fromkeys(
        rule_id for payload in memory_payloads for rule_id in payload.get("rule_ids", [])
        if rule_id in rules_by_id
    ))
    rules = [rules_by_id[rule_id] for rule_id in rule_ids]
    if not rules:
        return {
            "decision": "REVIEW", "matched_rules": [], "evidence": [],
            "reason": "规则记忆未返回有效条目", "valid": True, "wrapper_valid": True,
        }

    pass_rules = [rule for rule in rules if rule["decision"] == "PASS"]
    risky_rules = [rule for rule in rules if rule["decision"] != "PASS"]
    exempted = {
        risky["rule_id"]
        for risky in risky_rules
        if any(passed["rule_id"] in risky.get("exception_rule_ids", []) for passed in pass_rules)
    }
    if risky_rules and len(exempted) == len(risky_rules):
        decision = "PASS"
        applied = [
            passed["rule_id"]
            for passed in pass_rules
            if any(passed["rule_id"] in risky.get("exception_rule_ids", []) for risky in risky_rules)
        ]
        reason = f"{('、'.join(applied))} 覆盖了全部已命中风险规则的绑定例外"
    else:
        unexempted = [rule for rule in risky_rules if rule["rule_id"] not in exempted]
        strongest = max(unexempted or rules, key=lambda item: (决策等级[item["decision"]], item["priority"]))
        decision = strongest["decision"]
        reason = f"按风险等级和优先级采用 {strongest['rule_id']}"

    evidence = []
    for rule in rules:
        for cue in rule.get("eval_cues", []) + rule.get("train_cues", []):
            if cue in content:
                evidence.append(cue)
    return {
        "decision": decision,
        "matched_rules": rule_ids,
        "evidence": list(dict.fromkeys(evidence)),
        "reason": reason,
        "valid": True,
        "wrapper_valid": True,
    }


def 集合指标(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    """计算集合精确率、召回率和 F1。"""

    p, g = set(predicted), set(gold)
    precision = len(p & g) / len(p) if p else float(not g)
    recall = len(p & g) / len(g) if g else float(not p)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def 宏平均F1(gold: list[str], predicted: list[str]) -> float:
    """不依赖 sklearn 计算三分类宏平均 F1。"""

    values = []
    for label in 决策集合:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values)


def 汇总审核(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总决策、规则、证据、格式和场景指标。"""

    gold = [trace["gold_decision"] for trace in traces]
    predicted = [trace["prediction"]["decision"] for trace in traces]
    rule_scores = [
        集合指标(trace["prediction"]["matched_rules"], trace["gold_rule_ids"])
        for trace in traces
    ]
    evidence_scores = []
    for trace in traces:
        prediction_text = " ".join(trace["prediction"].get("evidence", []))
        evidence_scores.append(sum(piece in prediction_text for piece in trace["gold_evidence"]) / len(trace["gold_evidence"]))

    def 场景准确率(kind: str) -> float | None:
        subset = [trace for trace in traces if trace["scenario_type"] == kind]
        if not subset:
            return None
        return sum(trace["prediction"]["decision"] == trace["gold_decision"] for trace in subset) / len(subset)

    return {
        "samples": len(traces),
        "decision_accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(traces),
        "decision_macro_f1": 宏平均F1(gold, predicted),
        "rule_precision": sum(score[0] for score in rule_scores) / len(rule_scores),
        "rule_recall": sum(score[1] for score in rule_scores) / len(rule_scores),
        "rule_f1": sum(score[2] for score in rule_scores) / len(rule_scores),
        "evidence_coverage": sum(evidence_scores) / len(evidence_scores),
        "format_rate": sum(bool(trace["prediction"].get("valid")) for trace in traces) / len(traces),
        "wrapper_rate": sum(bool(trace["prediction"].get("wrapper_valid")) for trace in traces) / len(traces),
        "single_rule_accuracy": 场景准确率("single_rule"),
        "bound_exception_accuracy": 场景准确率("bound_exception"),
        "cross_rule_accuracy": 场景准确率("cross_rule_priority"),
        "mean_prompt_chars": sum(trace.get("prompt_chars", 0) for trace in traces) / len(traces),
        "elapsed_seconds": sum(trace.get("elapsed_seconds", 0.0) for trace in traces),
    }
