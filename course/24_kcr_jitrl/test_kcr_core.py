#!/usr/bin/env python3
"""KCR-JitRL 三库、门控、硬规则和规则浓缩单元测试。"""

from __future__ import annotations

import random

from kcr_core import (
    修正动作_logits,
    折扣回报,
    支持库,
    支持条目,
    案例,
    案例库,
    规则,
    规则库,
)

动作 = ("甲", "乙", "丙")
状态 = "任务 测试 阶段 校验 进度 0"


assert 折扣回报([0.5, 2.0], 0.5) == [1.5, 2.0]

support = 支持库([
    支持条目("doc1", 状态, "甲", 1.0, 0.8, "测试文档"),
])
support_scores, support_confidence, support_ids = support.评分(状态, 动作, 0.95)
assert support_scores["甲"] == 1.0
assert support_confidence == 0.8
assert support_ids == ["doc1"]

rules = 规则库([
    规则("soft", 状态, "乙", "recommend", 1.0, 0.9, True, False, "人工规则"),
    规则("hard", 状态, "丙", "forbid", 1.0, 1.0, True, True, "人工规则"),
])
rule_scores, _, rule_ids, blocked = rules.评分(状态, 动作, 0.95)
assert rule_scores["乙"] == 1.0
assert set(rule_ids) == {"soft", "hard"}
assert blocked == {"丙"}
rules.启用("soft", False)
assert rules.评分(状态, 动作, 0.95)[0] == {"甲": 0.0, "乙": 0.0, "丙": 0.0}
rules.启用("soft", True)

cases = 案例库([
    案例(状态, "甲", 2.0, 0, 0),
    案例(状态, "甲", 2.0, 1, 0),
    案例(状态, "乙", -1.0, 2, 0),
    案例(状态, "乙", -1.0, 3, 0),
    案例(状态, "丙", -1.0, 4, 0),
    案例(状态, "丙", -1.0, 5, 0),
])
changed = rules.从案例浓缩(cases, min_evidence=6, margin=0.5)
assert changed == 1
condensed = [entry for entry in rules.entries if entry.source == "案例浓缩"]
assert len(condensed) == 1 and condensed[0].action == "甲"

details = 修正动作_logits(
    状态,
    动作,
    [0.0, 0.0, 0.0],
    cases,
    support,
    rules,
    random.Random(7),
    enable_cases=True,
    enable_support=True,
    enable_rules=True,
    use_confidence_gate=True,
    beta_case=2.0,
    beta_support=1.0,
    beta_rule=1.0,
    top_k=10,
    case_threshold=0.95,
    support_threshold=0.95,
    rule_threshold=0.95,
    unseen_probability=0.0,
    optimism_alpha=5.0,
    min_confidence_samples=4,
)
assert details.neighbor_count == 6
assert details.corrected_logits["甲"] > details.corrected_logits["乙"]
assert details.corrected_logits["丙"] == -1e9
assert details.case_signal.confidence > 0
assert details.support_signal.matched_ids == ["doc1"]

rules.删除("hard")
assert all(entry.rule_id != "hard" for entry in rules.entries)
print("KCR_JITRL_CORE_TEST=PASS")
