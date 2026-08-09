#!/usr/bin/env python3
"""KCR-JitRL 的支持库、案例库、规则库与组合 logits 修正。"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path


def 状态词元(text: str) -> set[str]:
    """把中英文状态拆成集合，保持实现不依赖额外分词模型。"""

    return set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower()))


def 文本杰卡德(left: str, right: str) -> float:
    """计算两段结构化状态文本的 Jaccard 相似度。"""

    left_tokens = 状态词元(left)
    right_tokens = 状态词元(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def 折扣回报(rewards: Sequence[float], gamma: float) -> list[float]:
    """从轨迹末端向前计算每一步的 Monte Carlo 折扣回报。"""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma 必须位于 [0, 1]")
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def _归一化分数(scores: dict[str, float], actions: Sequence[str], epsilon: float = 1e-8) -> dict[str, float]:
    """先中心化再缩放到 [-1, 1]，避免不同来源的量纲互相污染。"""

    mean = sum(scores[action] for action in actions) / len(actions)
    centered = {action: scores[action] - mean for action in actions}
    scale = max(max(abs(value) for value in centered.values()), epsilon)
    return {action: value / scale for action, value in centered.items()}


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """稳定计算一维 softmax。"""

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    scaled = [float(value) / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def 按概率采样(actions: Sequence[str], probabilities: Sequence[float], rng: random.Random) -> str:
    """按给定概率进行可复现采样。"""

    draw = rng.random()
    cumulative = 0.0
    for action, probability in zip(actions, probabilities):
        cumulative += probability
        if draw <= cumulative:
            return action
    return actions[-1]


@dataclass(frozen=True)
class 案例:
    """一次真实交互产生的状态—动作—折扣回报。"""

    state: str
    action: str
    return_value: float
    episode: int
    step: int


class 案例库:
    """保存成功和失败案例，并提供带相似度的局部检索。"""

    def __init__(self, entries: Sequence[案例] | None = None):
        self.entries = list(entries or [])

    def __len__(self) -> int:
        return len(self.entries)

    def 添加轨迹(
        self,
        states: Sequence[str],
        actions: Sequence[str],
        rewards: Sequence[float],
        gamma: float,
        episode: int,
    ) -> None:
        """把完整轨迹转换成逐步案例后写入案例库。"""

        if not (len(states) == len(actions) == len(rewards)):
            raise ValueError("states、actions 与 rewards 长度必须一致")
        returns = 折扣回报(rewards, gamma)
        self.entries.extend(
            案例(state, action, return_value, episode, step)
            for step, (state, action, return_value) in enumerate(zip(states, actions, returns))
        )

    def 检索(self, state: str, top_k: int, threshold: float) -> list[tuple[float, 案例]]:
        """返回超过阈值的最相似案例，相似度并列时优先较新案例。"""

        ranked = []
        for index, entry in enumerate(self.entries):
            similarity = 文本杰卡德(state, entry.state)
            if similarity >= threshold:
                ranked.append((similarity, index, entry))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [(similarity, entry) for similarity, _, entry in ranked[:top_k]]

    def 保存(self, path: Path) -> None:
        """按 UTF-8 JSONL 保存案例。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class 支持条目:
    """一条带来源置信度的外部知识或操作依据。"""

    entry_id: str
    state: str
    action: str
    score: float
    confidence: float
    text: str
    enabled: bool = True


class 支持库:
    """只提供动作证据，不把文档内容伪装成环境奖励。"""

    def __init__(self, entries: Sequence[支持条目] | None = None):
        self.entries = list(entries or [])

    @classmethod
    def 加载(cls, path: Path) -> 支持库:
        """从 JSONL 加载支持条目。"""

        entries = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entry = 支持条目(**json.loads(line))
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError(f"支持库第 {line_number} 行格式错误：{error}") from error
                if not 0.0 <= entry.confidence <= 1.0:
                    raise ValueError(f"支持条目 {entry.entry_id} 的 confidence 必须位于 [0, 1]")
                entries.append(entry)
        return cls(entries)

    def 评分(
        self,
        state: str,
        actions: Sequence[str],
        threshold: float,
    ) -> tuple[dict[str, float], float, list[str]]:
        """把相关文档对候选动作的支持强度转换为归一化分数。"""

        raw = {action: 0.0 for action in actions}
        matched: list[tuple[float, 支持条目]] = []
        for entry in self.entries:
            if not entry.enabled or entry.action not in raw:
                continue
            similarity = 文本杰卡德(state, entry.state)
            if similarity >= threshold:
                raw[entry.action] += similarity * entry.score
                matched.append((similarity, entry))
        if not matched:
            return raw, 0.0, []
        confidence = sum(similarity * entry.confidence for similarity, entry in matched) / sum(
            similarity for similarity, _ in matched
        )
        return _归一化分数(raw, actions), confidence, [entry.entry_id for _, entry in matched]


@dataclass(frozen=True)
class 规则:
    """一条人工补充或从案例中浓缩的可审计规则。"""

    rule_id: str
    state: str
    action: str
    effect: str
    strength: float
    confidence: float
    enabled: bool
    hard: bool
    source: str
    evidence_count: int = 0


class 规则库:
    """规则只支持手动启停和删除，不包含自动过期机制。"""

    def __init__(self, entries: Sequence[规则] | None = None):
        self.entries = list(entries or [])

    @classmethod
    def 加载(cls, path: Path) -> 规则库:
        """从 JSONL 加载规则。"""

        entries = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entry = 规则(**json.loads(line))
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError(f"规则库第 {line_number} 行格式错误：{error}") from error
                if entry.effect not in {"recommend", "forbid"}:
                    raise ValueError(f"规则 {entry.rule_id} 的 effect 只能是 recommend 或 forbid")
                if not 0.0 <= entry.confidence <= 1.0:
                    raise ValueError(f"规则 {entry.rule_id} 的 confidence 必须位于 [0, 1]")
                entries.append(entry)
        return cls(entries)

    def 克隆(self) -> 规则库:
        """为不同随机种子建立互不污染的规则库。"""

        return 规则库(list(self.entries))

    def 启用(self, rule_id: str, enabled: bool) -> None:
        """手动启用或停用一条规则。"""

        for index, entry in enumerate(self.entries):
            if entry.rule_id == rule_id:
                self.entries[index] = replace(entry, enabled=enabled)
                return
        raise KeyError(f"找不到规则：{rule_id}")

    def 删除(self, rule_id: str) -> None:
        """手动删除一条规则；不存在时明确报错。"""

        old_length = len(self.entries)
        self.entries = [entry for entry in self.entries if entry.rule_id != rule_id]
        if len(self.entries) == old_length:
            raise KeyError(f"找不到规则：{rule_id}")

    def 评分(
        self,
        state: str,
        actions: Sequence[str],
        threshold: float,
    ) -> tuple[dict[str, float], float, list[str], set[str]]:
        """计算软规则分数，并单独返回硬禁止动作集合。"""

        raw = {action: 0.0 for action in actions}
        matched: list[tuple[float, 规则]] = []
        hard_blocked: set[str] = set()
        for entry in self.entries:
            if not entry.enabled or entry.action not in raw:
                continue
            similarity = 文本杰卡德(state, entry.state)
            if similarity < threshold:
                continue
            matched.append((similarity, entry))
            if entry.hard and entry.effect == "forbid":
                hard_blocked.add(entry.action)
                continue
            direction = 1.0 if entry.effect == "recommend" else -1.0
            raw[entry.action] += similarity * direction * entry.strength
        if not matched:
            return raw, 0.0, [], hard_blocked
        confidence = sum(similarity * entry.confidence for similarity, entry in matched) / sum(
            similarity for similarity, _ in matched
        )
        normalized = _归一化分数(raw, actions) if any(raw.values()) else raw
        return normalized, confidence, [entry.rule_id for _, entry in matched], hard_blocked

    def 从案例浓缩(self, cases: 案例库, min_evidence: int, margin: float) -> int:
        """把重复且有明显回报优势的案例压缩成软规则。"""

        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for entry in cases.entries:
            grouped[entry.state][entry.action].append(entry.return_value)

        changed = 0
        existing = {entry.rule_id: index for index, entry in enumerate(self.entries)}
        for state, action_returns in grouped.items():
            total = sum(len(values) for values in action_returns.values())
            if total < min_evidence or len(action_returns) < 2:
                continue
            means = {
                action: sum(values) / len(values)
                for action, values in action_returns.items()
            }
            ranked = sorted(means.items(), key=lambda item: item[1], reverse=True)
            best_action, best_value = ranked[0]
            second_value = ranked[1][1]
            best_count = len(action_returns[best_action])
            if best_count < 2 or best_value - second_value < margin:
                continue
            digest = hashlib.sha1(state.encode()).hexdigest()[:10]
            rule_id = f"condensed_{digest}"
            confidence = min(0.95, 0.50 + 0.05 * best_count)
            rule = 规则(
                rule_id=rule_id,
                state=state,
                action=best_action,
                effect="recommend",
                strength=1.0,
                confidence=confidence,
                enabled=True,
                hard=False,
                source="案例浓缩",
                evidence_count=total,
            )
            if rule_id in existing:
                index = existing[rule_id]
                if self.entries[index] != rule:
                    self.entries[index] = rule
                    changed += 1
            else:
                existing[rule_id] = len(self.entries)
                self.entries.append(rule)
                changed += 1
        return changed

    def 保存(self, path: Path) -> None:
        """保存完整规则库，方便人工查看、停用或删除。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class 来源信号:
    """一个知识来源对当前动作集合给出的分数和可信度。"""

    scores: dict[str, float]
    confidence: float
    matched_ids: list[str]


@dataclass(frozen=True)
class KCR修正详情:
    """记录三种来源及其最终 logits 贡献，便于审计和消融。"""

    neighbor_count: int
    case_signal: 来源信号
    support_signal: 来源信号
    rule_signal: 来源信号
    hard_blocked_actions: list[str]
    base_logits: dict[str, float]
    contributions: dict[str, dict[str, float]]
    corrected_logits: dict[str, float]


def _加权均值(pairs: Sequence[tuple[float, float]]) -> float:
    denominator = sum(weight for weight, _ in pairs)
    return sum(weight * value for weight, value in pairs) / denominator


def _案例信号(
    state: str,
    actions: Sequence[str],
    cases: 案例库,
    rng: random.Random,
    *,
    top_k: int,
    threshold: float,
    unseen_probability: float,
    optimism_alpha: float,
    min_confidence_samples: int,
) -> tuple[来源信号, int]:
    """用相似度加权回报估计案例优势和数据置信度。"""

    neighbors = cases.检索(state, top_k, threshold)
    zeros = {action: 0.0 for action in actions}
    if not neighbors:
        return 来源信号(zeros, 0.0, []), 0

    state_pairs = [(similarity, entry.return_value) for similarity, entry in neighbors]
    state_value = _加权均值(state_pairs)
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for similarity, entry in neighbors:
        grouped[entry.action].append((similarity, entry.return_value))

    action_values = {}
    for action in actions:
        observed = grouped.get(action)
        if observed:
            action_values[action] = _加权均值(observed)
        elif rng.random() < unseen_probability:
            action_values[action] = state_value + optimism_alpha / len(neighbors)
        else:
            action_values[action] = 0.0
    advantages = {action: action_values[action] - state_value for action in actions}
    normalized = _归一化分数(advantages, actions)

    mean_similarity = sum(similarity for similarity, _ in neighbors) / len(neighbors)
    variance = sum(
        weight * (value - state_value) ** 2
        for weight, value in state_pairs
    ) / sum(weight for weight, _ in state_pairs)
    agreement = 1.0 / (1.0 + math.sqrt(variance))
    coverage = min(1.0, len(neighbors) / max(min_confidence_samples, 1))
    confidence = mean_similarity * agreement * coverage
    matched_ids = [f"episode={entry.episode},step={entry.step}" for _, entry in neighbors]
    return 来源信号(normalized, confidence, matched_ids), len(neighbors)


def 修正动作_logits(
    state: str,
    actions: Sequence[str],
    base_logits: Sequence[float],
    cases: 案例库,
    support: 支持库,
    rules: 规则库,
    rng: random.Random,
    *,
    enable_cases: bool,
    enable_support: bool,
    enable_rules: bool,
    use_confidence_gate: bool,
    beta_case: float,
    beta_support: float,
    beta_rule: float,
    top_k: int,
    case_threshold: float,
    support_threshold: float,
    rule_threshold: float,
    unseen_probability: float,
    optimism_alpha: float,
    min_confidence_samples: int,
) -> KCR修正详情:
    """把案例优势、文档证据和规则先验相加到冻结策略 logits。"""

    if len(actions) != len(base_logits):
        raise ValueError("动作数与基础 logits 数量必须一致")
    if len(set(actions)) != len(actions):
        raise ValueError("候选动作不能重复")
    if not 0.0 <= unseen_probability <= 1.0:
        raise ValueError("unseen_probability 必须位于 [0, 1]")

    zeros = {action: 0.0 for action in actions}
    case_signal, neighbor_count = (
        _案例信号(
            state,
            actions,
            cases,
            rng,
            top_k=top_k,
            threshold=case_threshold,
            unseen_probability=unseen_probability,
            optimism_alpha=optimism_alpha,
            min_confidence_samples=min_confidence_samples,
        )
        if enable_cases
        else (来源信号(dict(zeros), 0.0, []), 0)
    )
    if enable_support:
        scores, confidence, matched = support.评分(state, actions, support_threshold)
        support_signal = 来源信号(scores, confidence, matched)
    else:
        support_signal = 来源信号(dict(zeros), 0.0, [])
    if enable_rules:
        scores, confidence, matched, hard_blocked = rules.评分(state, actions, rule_threshold)
        rule_signal = 来源信号(scores, confidence, matched)
    else:
        rule_signal = 来源信号(dict(zeros), 0.0, [])
        hard_blocked = set()

    def source_contribution(signal: 来源信号, beta: float) -> dict[str, float]:
        gate = signal.confidence if use_confidence_gate else float(bool(signal.matched_ids))
        return {action: beta * gate * signal.scores[action] for action in actions}

    contributions = {
        "case": source_contribution(case_signal, beta_case),
        "support": source_contribution(support_signal, beta_support),
        "rule": source_contribution(rule_signal, beta_rule),
    }
    base = {action: float(logit) for action, logit in zip(actions, base_logits)}
    corrected = {
        action: base[action] + sum(source[action] for source in contributions.values())
        for action in actions
    }
    # 使用有限大负数而不是负无穷，保证结果可以安全写入标准 JSON。
    for action in hard_blocked:
        corrected[action] = -1e9

    return KCR修正详情(
        neighbor_count=neighbor_count,
        case_signal=case_signal,
        support_signal=support_signal,
        rule_signal=rule_signal,
        hard_blocked_actions=sorted(hard_blocked),
        base_logits=base,
        contributions=contributions,
        corrected_logits=corrected,
    )
