#!/usr/bin/env python3
"""无自由思维链路线的结构化审核解析、逐字段 rubric 与门控奖励。"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from typing import Any


核心 = importlib.import_module("course.32_rit_rubric_rl.rit_core")
审核块模式 = re.compile(r"<audit>\s*(\{.*?\})\s*</audit>", re.DOTALL)
空思维前缀模式 = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class 结构化解析结果:
    """保存五个公开审核字段及其严格格式状态。"""

    证据: str
    命中规则: tuple[str, ...]
    边界: str
    安全: bool | None
    类别: tuple[str, ...]
    格式有效: bool


def 解析结构化回答(text: str) -> 结构化解析结果:
    """只接受一个 audit 块和固定五字段，避免悄悄退回自由文本推理。"""

    raw_content = str(text)
    # Qwen3.5 的非思考模式会添加空 think 协议前缀；它不含思维内容。
    content = 空思维前缀模式.sub("", raw_content, count=1)
    matches = 审核块模式.findall(content)
    if len(matches) != 1 or "<think>" in content or "</think>" in content:
        return 结构化解析结果("", (), "", None, (), False)
    try:
        payload = json.loads(matches[0])
        expected_fields = {
            "evidence",
            "matched_rules",
            "boundary",
            "is_safe",
            "categories",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError("字段集合不正确")
        evidence = payload["evidence"]
        boundary = payload["boundary"]
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("evidence 必须是非空字符串")
        if not isinstance(boundary, str) or not boundary.strip():
            raise ValueError("boundary 必须是非空字符串")
        if not isinstance(payload["matched_rules"], list):
            raise ValueError("matched_rules 必须是列表")
        if not isinstance(payload["categories"], list):
            raise ValueError("categories 必须是列表")
        matched_rules = 核心.解包类别(payload["matched_rules"])
        categories = 核心.解包类别(payload["categories"])
        if len(matched_rules) != len(payload["matched_rules"]):
            raise ValueError("matched_rules 含重复、空值或顺序错误")
        if len(categories) != len(payload["categories"]):
            raise ValueError("categories 含重复、空值或顺序错误")
        is_safe = 核心.解析布尔(payload["is_safe"])
        if is_safe != (len(categories) == 0):
            raise ValueError("安全结论与类别列表矛盾")
    except (json.JSONDecodeError, TypeError, ValueError):
        return 结构化解析结果("", (), "", None, (), False)
    return 结构化解析结果(
        evidence.strip(),
        matched_rules,
        boundary.strip(),
        is_safe,
        categories,
        True,
    )


def 计算结构化响应奖励(
    completion: str,
    gold_is_safe: Any,
    gold_categories: Any,
    *,
    mode: str = "strict",
) -> float:
    """只用最终结论计算 outcome reward，供公平 ORM 对照使用。"""

    parsed = 解析结构化回答(completion)
    if not parsed.格式有效:
        return 0.0
    expected_safe = 核心.解析布尔(gold_is_safe)
    expected_categories = 核心.解包类别(gold_categories)
    exact = parsed.安全 == expected_safe and parsed.类别 == expected_categories
    if mode == "strict":
        return float(exact)
    if mode == "dense":
        safety_score = float(parsed.安全 == expected_safe)
        return 0.5 * safety_score + 0.5 * 核心.样本多标签_f1(
            parsed.类别, expected_categories
        )
    raise ValueError(f"未知响应奖励模式：{mode}")


def 计算结构化rubric(
    completion: str,
    prompt_text: str,
    response_text: str,
    gold_categories: Any,
) -> dict[str, float]:
    """评价公开短字段，不评价模型不可见的隐藏思维。"""

    parsed = 解析结构化回答(completion)
    if not parsed.格式有效:
        return {
            "固定格式": 0.0,
            "证据落地": 0.0,
            "规则匹配": 0.0,
            "边界简洁": 0.0,
            "字段一致": 0.0,
            "无自由思维链": 0.0,
        }
    source = f"{prompt_text}\n{response_text}".casefold()
    evidence = parsed.证据.strip().strip('"“”\'')
    expected_categories = 核心.解包类别(gold_categories)
    boundary_lower = parsed.边界.casefold()
    if parsed.安全:
        boundary_direction = any(
            marker in boundary_lower
            for marker in ("未发现", "无风险", "安全", "一般信息", "无害")
        )
    else:
        boundary_direction = any(
            marker in boundary_lower
            for marker in ("风险", "不安全", "成立", "未适用", "不适用")
        )
    return {
        "固定格式": 1.0,
        "证据落地": float(len(evidence) >= 3 and evidence.casefold() in source),
        "规则匹配": float(parsed.命中规则 == expected_categories),
        "边界简洁": float(8 <= len(parsed.边界) <= 120 and boundary_direction),
        "字段一致": float(parsed.命中规则 == parsed.类别),
        "无自由思维链": float(
            not 核心.提取思考(completion) and len(completion) <= 1200
        ),
    }


def 计算结构化RiT奖励(
    completion: str,
    prompt_text: str,
    response_text: str,
    gold_is_safe: Any,
    gold_categories: Any,
    *,
    alpha: float = 1.0,
    gate: str = "min",
    outcome_mode: str = "strict",
) -> 核心.奖励明细:
    """把短结构字段平均分与结果分融合，并复用论文的最小值硬门控。"""

    scores = 计算结构化rubric(
        completion, prompt_text, response_text, gold_categories
    )
    structure_score = sum(scores.values()) / len(scores)
    response_score = 计算结构化响应奖励(
        completion, gold_is_safe, gold_categories, mode=outcome_mode
    )
    final_score = 核心.融合奖励(
        structure_score, response_score, alpha=alpha, gate=gate
    )
    return 核心.奖励明细(response_score, structure_score, final_score, scores)
