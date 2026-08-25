#!/usr/bin/env python3
"""提供无需额外向量模型的规则库与案例库混合检索。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 记录。"""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 检索词(text: str) -> list[str]:
    """同时提取英文词、数字、中文单字和中文双字词。"""

    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    return [*words, *chinese, *("".join(chinese[index:index + 2]) for index in range(len(chinese) - 1))]


def 字符片段(text: str, size: int = 3) -> Counter[str]:
    """用字符片段容忍拼写变化、变体词和未知词。"""

    compact = re.sub(r"\s+", " ", text.lower()).strip()
    if len(compact) < size:
        return Counter((compact,)) if compact else Counter()
    return Counter(compact[index:index + size] for index in range(len(compact) - size + 1))


def 余弦相似度(left: Counter[str], right: Counter[str]) -> float:
    """计算稀疏计数向量的余弦相似度。"""

    if not left or not right:
        return 0.0
    overlap = left.keys() & right.keys()
    numerator = sum(left[key] * right[key] for key in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class BM25索引:
    """小规模课程数据使用的可审计 BM25 实现。"""

    def __init__(self, documents: Iterable[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokens = [检索词(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.postings: dict[str, list[tuple[int, int]]] = {}
        document_frequency: Counter[str] = Counter()
        for index, counts in enumerate(self.term_counts):
            document_frequency.update(counts.keys())
            for term, frequency in counts.items():
                self.postings.setdefault(term, []).append((index, frequency))
        total = len(self.tokens)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def 分数(self, query: str) -> list[float]:
        """返回查询与全部文档的原始 BM25 分数。"""

        query_terms = set(检索词(query))
        scores = [0.0] * len(self.tokens)
        for term in query_terms:
            for index, frequency in self.postings.get(term, ()):
                normalization = self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / max(self.average_length, 1e-8)
                )
                scores[index] += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / (frequency + normalization)
                )
        return scores


def 归一化(values: list[float]) -> list[float]:
    """把一组检索分数缩放到零至一。"""

    if not values:
        return []
    minimum, maximum = min(values), max(values)
    if math.isclose(minimum, maximum):
        return [1.0 if maximum > 0 else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


class 文档检索器:
    """组合 BM25 与字符片段余弦，保留每部分分数供审计。"""

    def __init__(self, rows: list[dict[str, Any]], text_key: str):
        self.rows = rows
        self.text_key = text_key
        self.texts = [str(row[text_key]) for row in rows]
        self.bm25 = BM25索引(self.texts)
        self.fragments = [字符片段(text) for text in self.texts]

    def 搜索(
        self,
        query: str,
        top_k: int,
        exclude_record_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """返回融合分数最高的文档，可排除当前案例防止自检索。"""

        bm25_raw = self.bm25.分数(query)
        query_fragments = 字符片段(query)
        # 字符片段只为 BM25 候选做重排；避免对 1600 个长案例逐一做大集合相交。
        candidate_count = min(len(self.rows), max(64, top_k * 20))
        candidate_indices = sorted(range(len(self.rows)), key=lambda index: -bm25_raw[index])[:candidate_count]
        character_raw = [0.0] * len(self.rows)
        for index in candidate_indices:
            character_raw[index] = 余弦相似度(query_fragments, self.fragments[index])
        bm25_scores = 归一化(bm25_raw)
        character_scores = 归一化(character_raw)
        ranked = []
        for index, row in enumerate(self.rows):
            if exclude_record_id and row.get("record_id") == exclude_record_id:
                continue
            score = 0.65 * bm25_scores[index] + 0.35 * character_scores[index]
            ranked.append(
                {
                    **row,
                    "retrieval_score": score,
                    "bm25_score": bm25_scores[index],
                    "character_score": character_scores[index],
                }
            )
        ranked.sort(key=lambda row: (-row["retrieval_score"], str(row.get("record_id", row.get("rule_id", "")))))
        return ranked[:top_k]


def 规则检索文本(rule: dict[str, Any]) -> str:
    """把结构化规则转换成只供检索的文本。"""

    fields = [
        rule["category"],
        rule["name_zh"],
        rule["definition_zh"],
        rule["definition_en"],
        *rule["inclusions"],
        *rule["exceptions"],
    ]
    return " ".join(str(field) for field in fields)


def 案例检索文本(case: dict[str, Any]) -> str:
    """案例相似性只使用可观测文本，不把标签写入检索向量。"""

    return f"{case['prompt']}\n{case['response']}"


class 规则案例检索器:
    """管理两种独立索引并生成模型可读的证据上下文。"""

    def __init__(self, rules: list[dict[str, Any]], cases: list[dict[str, Any]]):
        active_rules = [rule for rule in rules if rule.get("status") == "active"]
        rule_documents = [{**rule, "retrieval_text": 规则检索文本(rule)} for rule in active_rules]
        case_documents = [{**case, "retrieval_text": 案例检索文本(case)} for case in cases]
        self.rule_retriever = 文档检索器(rule_documents, "retrieval_text")
        self.case_retriever = 文档检索器(case_documents, "retrieval_text")
        self.cache: dict[tuple[str, str, str | None, int, int], dict[str, list[dict[str, Any]]]] = {}

    @classmethod
    def 从目录(cls, knowledge_dir: Path) -> "规则案例检索器":
        """从版本化 JSONL 文件恢复检索器。"""

        return cls(
            读取_jsonl(knowledge_dir / "rules.jsonl"),
            读取_jsonl(knowledge_dir / "cases.jsonl"),
        )

    def 检索(
        self,
        query: str,
        mode: str,
        exclude_record_id: str | None = None,
        rule_top_k: int = 3,
        case_top_k: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """按消融模式检索规则和案例，并让案例包含安全/违规两种结论。"""

        if mode not in {"none", "rules", "cases", "full"}:
            raise ValueError(f"未知检索模式：{mode}")
        cache_key = (query, mode, exclude_record_id, rule_top_k, case_top_k)
        if cache_key in self.cache:
            return self.cache[cache_key]
        rules = (
            self.rule_retriever.搜索(query, rule_top_k)
            if mode in {"rules", "full"}
            else []
        )
        candidates = (
            self.case_retriever.搜索(query, max(case_top_k * 6, case_top_k), exclude_record_id)
            if mode in {"cases", "full"}
            else []
        )
        selected: list[dict[str, Any]] = []
        # 先放入最相似的安全和违规案例，减少单一结论邻居造成的确认偏差。
        for wanted_safe in (True, False):
            match = next((case for case in candidates if bool(case["is_safe"]) == wanted_safe), None)
            if match is not None and match not in selected and len(selected) < case_top_k:
                selected.append(match)
        for case in candidates:
            if case not in selected and len(selected) < case_top_k:
                selected.append(case)
        selected.sort(key=lambda row: -row["retrieval_score"])
        result = {"rules": rules, "cases": selected}
        self.cache[cache_key] = result
        return result


def 渲染上下文(retrieved: dict[str, list[dict[str, Any]]]) -> str:
    """把结构化命中结果压缩成稳定、可审计的模型上下文。"""

    rule_lines = []
    for rule in retrieved["rules"]:
        rule_lines.append(
            f"{rule['rule_id']}@v{rule['version']} | {rule['route']} | {rule['category']} | "
            f"定义：{rule['definition_zh']} | 例外：{'；'.join(rule['exceptions'])}"
        )
    case_lines = []
    for case in retrieved["cases"]:
        labels = ",".join(case["categories"]) if case["categories"] else "NONE"
        case_lines.append(
            f"{case['record_id']} | 结论：{'SAFE' if case['is_safe'] else 'UNSAFE'} | "
            f"类别：{labels} | 请求：{case['prompt'][:180]} | 回复：{case['response'][:260]}"
        )
    return (
        "[检索到的规则]\n"
        + ("\n".join(rule_lines) if rule_lines else "NONE")
        + "\n[相似历史案例]\n"
        + ("\n".join(case_lines) if case_lines else "NONE")
    )


def 审核输入(prompt: str, response: str, context: str) -> str:
    """组合审核对象和可选外部证据。"""

    return (
        f"{context}\n\n[用户请求]\n{prompt}\n\n[模型回复]\n{response}\n\n"
        "请按系统提示给出结构化审核结果。"
    )
