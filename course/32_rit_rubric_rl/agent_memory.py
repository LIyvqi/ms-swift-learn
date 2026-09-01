#!/usr/bin/env python3
"""为安全审核 Agent 提供两个独立、可人工维护的极简检索库。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 中的全部非空记录。"""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 检索特征(text: str) -> list[str]:
    """抽取英文词、中文单字和中文双字，避免依赖额外向量模型。"""

    normalized = re.sub(r"\s+", " ", str(text).casefold()).strip()
    features = re.findall(r"[a-z0-9][a-z0-9_,.+-]*", normalized)
    for segment in re.findall(r"[\u3400-\u9fff]+", normalized):
        features.extend(segment)
        features.extend(
            segment[index : index + 2] for index in range(len(segment) - 1)
        )
    return features


class 极简BM25索引:
    """实现可解释的小规模 BM25，不把它包装成语义向量检索。"""

    def __init__(self, texts: Iterable[str]):
        self.texts = list(texts)
        self.tokens = [检索特征(text) for text in self.texts]
        self.counts = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(self.tokens)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def _score(self, query_tokens: list[str], index: int) -> float:
        k1, b = 1.5, 0.75
        score = 0.0
        for term in set(query_tokens):
            frequency = self.counts[index].get(term, 0)
            if not frequency:
                continue
            normalization = k1 * (
                1 - b + b * self.lengths[index] / max(self.average_length, 1.0)
            )
            score += self.idf.get(term, 0.0) * (
                frequency * (k1 + 1) / (frequency + normalization)
            )
        return score

    def 搜索(
        self,
        query: str,
        top_k: int,
        candidates: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """在给定候选中返回稳定排序的索引和 BM25 分数。"""

        indices = list(candidates) if candidates is not None else list(range(len(self.texts)))
        tokens = 检索特征(query)
        scored = [(index, self._score(tokens, index)) for index in indices]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: max(1, min(int(top_k), len(scored)))]


class 极简审核记忆:
    """分别加载版本化规则 JSONL 和人工确认 Case JSONL。"""

    def __init__(self, rules_path: Path, cases_path: Path):
        self.rules_path = rules_path
        self.cases_path = cases_path
        self.rules = [
            row for row in 读取_jsonl(rules_path) if row.get("status") == "active"
        ]
        self.cases = [
            row
            for row in 读取_jsonl(cases_path)
            if row.get("review_status") == "approved"
        ]
        if not self.rules or not self.cases:
            raise ValueError("规则库和案例库都必须至少包含一条有效记录")
        allowed_case_sources = {"train", "human"}
        if any(row.get("source_split") not in allowed_case_sources for row in self.cases):
            raise ValueError("案例库只能包含 train 或独立人工确认来源，禁止验证和测试样本")
        self.rule_index = 极简BM25索引(self._规则检索文本(row) for row in self.rules)
        self.case_index = 极简BM25索引(self._案例检索文本(row) for row in self.cases)
        self.rule_by_id = {self._规则_id(row): row for row in self.rules}
        self.case_by_id = {str(row["case_id"]): row for row in self.cases}

    @staticmethod
    def _规则_id(row: dict[str, Any]) -> str:
        return f"rule:{row['rule_id']}@v{row['version']}"

    @staticmethod
    def _规则检索文本(row: dict[str, Any]) -> str:
        return " ".join(
            str(value)
            for value in (
                row["category"],
                row["name_zh"],
                row["definition_zh"],
                row.get("definition_en", ""),
                *row.get("inclusions", []),
                *row.get("exceptions", []),
            )
        )

    @staticmethod
    def _案例检索文本(row: dict[str, Any]) -> str:
        verdict = "SAFE 安全" if row["is_safe"] else "UNSAFE 风险"
        return " ".join(
            [
                verdict,
                *row.get("categories", []),
                str(row["prompt"]),
                str(row["response"]),
                str(row.get("review_note", "")),
            ]
        )

    def 搜索规则(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """返回当前 active 规则，保留版本、条件和例外。"""

        normalized_query = str(query).casefold()
        ranked = self.rule_index.搜索(query, len(self.rules))
        ranked.sort(
            key=lambda item: (
                -(
                    item[1]
                    + (
                        100.0
                        if str(self.rules[item[0]]["category"]).casefold()
                        in normalized_query
                        else 0.0
                    )
                ),
                item[0],
            )
        )
        rows = []
        for index, score in ranked[: max(1, min(int(top_k), len(ranked)))]:
            rule = self.rules[index]
            rows.append(
                {
                    "rule_id": self._规则_id(rule),
                    "category": rule["category"],
                    "name_zh": rule["name_zh"],
                    "definition": rule["definition_zh"],
                    "inclusions": rule.get("inclusions", [])[:4],
                    "exceptions": rule.get("exceptions", [])[:3],
                    "priority": rule.get("priority", 0),
                    "score": round(score, 6),
                }
            )
        return rows

    def 搜索案例(
        self,
        query: str,
        top_k: int = 3,
        verdict: str = "any",
        exclude_record_id: str = "",
    ) -> list[dict[str, Any]]:
        """按结论过滤相似案例，并排除当前训练样本本身。"""

        if verdict not in {"safe", "unsafe", "any"}:
            raise ValueError("verdict 只能是 safe、unsafe 或 any")
        candidates = []
        for index, row in enumerate(self.cases):
            if exclude_record_id and row.get("record_id") == exclude_record_id:
                continue
            if verdict == "safe" and not row["is_safe"]:
                continue
            if verdict == "unsafe" and row["is_safe"]:
                continue
            candidates.append(index)
        rows = []
        for index, score in self.case_index.搜索(query, top_k, candidates):
            case = self.cases[index]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "prompt": str(case["prompt"])[:180],
                    "response": str(case["response"])[:260],
                    "is_safe": bool(case["is_safe"]),
                    "categories": list(case["categories"]),
                    "review_note": case.get("review_note", ""),
                    "score": round(score, 6),
                }
            )
        return rows

    def 类别名称(self) -> dict[str, str]:
        """提供紧凑的类别名称，不把完整规则提前塞入系统提示。"""

        return {str(row["category"]): str(row["name_zh"]) for row in self.rules}
