"""实现新闻规则库的混合检索、重排、规则组合与分层指标。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

允许类别 = ("政治", "财经", "体育", "计算机")

# 小型课程不额外下载向量模型，先用可解释同义概念增强字符特征。
# 后续可保持搜索接口不变，把该层替换成真实 embedding 服务。
同义概念组 = {
    "政治制度": ("政治制度", "政府职能", "人大", "政协", "行政", "治理"),
    "政党": ("共产党", "党组织", "党员", "党建", "党代会"),
    "国际关系": ("外交", "国际关系", "北约", "联合国", "峰会", "条约"),
    "宏观经济": ("宏观经济", "经济增长", "通货膨胀", "收入分配", "经济体制"),
    "金融市场": ("金融", "银行", "保险", "证券", "股票", "汇率", "利率"),
    "产业贸易": ("企业", "产业", "贸易", "市场竞争", "投资", "商业"),
    "体育竞赛": ("比赛", "联赛", "锦标赛", "奥运会", "比分", "冠军"),
    "运动训练": ("运动员", "教练", "训练", "体能", "冬泳", "田径"),
    "计算机软件": ("软件", "程序", "操作系统", "算法", "数据库", "编程"),
    "计算机网络": ("互联网", "网络", "协议", "服务器", "通信", "信息安全"),
    "人工智能": ("人工智能", "机器学习", "神经网络", "智能系统", "知识工程"),
    "计算机硬件": ("芯片", "处理器", "计算机", "硬件", "存储器", "微机"),
}


def 中文特征(text: str) -> list[str]:
    """生成稳定的中文单字、双字、英文词和同义概念特征。"""

    normalized = text.lower()
    features: list[str] = []
    features.extend(re.findall(r"[a-z0-9][a-z0-9_.+-]*", normalized))
    for segment in re.findall(r"[\u3400-\u9fff]+", normalized):
        features.extend(segment)
        features.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    for concept, aliases in 同义概念组.items():
        if any(alias.lower() in normalized for alias in aliases):
            features.append(f"概念:{concept}")
    return features


def 稳定稠密向量(text: str, dimensions: int = 256) -> list[float]:
    """用特征哈希构造稠密向量，提供无需额外模型的可复现向量基线。"""

    vector = [0.0] * dimensions
    for feature in 中文特征(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dimensions
        sign = 1.0 if value & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def 余弦相似度(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def 集合指标(predicted: Iterable[str], expected: Iterable[str]) -> dict[str, float]:
    """计算集合精确率、召回率和 F1。"""

    predicted_set = set(predicted)
    expected_set = set(expected)
    overlap = len(predicted_set & expected_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    canonical_id: str
    title: str
    text: str
    category: str
    conditions: tuple[str, ...]
    exceptions: tuple[str, ...]
    keywords: tuple[str, ...]
    priority: int
    source: str
    status: str = "active"

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Rule:
        return cls(
            rule_id=row["rule_id"],
            canonical_id=row.get("canonical_id", row["rule_id"]),
            title=row["title"],
            text=row["text"],
            category=row["category"],
            conditions=tuple(row.get("conditions", [])),
            exceptions=tuple(row.get("exceptions", [])),
            keywords=tuple(row.get("keywords", [])),
            priority=int(row.get("priority", 0)),
            source=row.get("source", "course"),
            status=row.get("status", "active"),
        )

    def searchable_text(self) -> str:
        parts = [self.title, self.text, self.category]
        parts.extend(self.conditions)
        parts.extend(self.exceptions)
        parts.extend(self.keywords)
        return " ".join(parts)

    def public_dict(self, scores: dict[str, float] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rule_id": self.rule_id,
            "canonical_id": self.canonical_id,
            "title": self.title,
            "text": self.text,
            "category": self.category,
            "conditions": list(self.conditions),
            "exceptions": list(self.exceptions),
            "priority": self.priority,
            "source": self.source,
        }
        if scores:
            result["scores"] = {key: round(value, 6) for key, value in scores.items()}
        return result


class RuleKnowledgeBase:
    """提供 BM25、哈希稠密向量、融合重排和确定性规则组合。"""

    def __init__(self, rules: list[Rule]):
        active_rules = [rule for rule in rules if rule.status == "active"]
        if not active_rules:
            raise ValueError("规则库没有可用规则")
        self.rules = active_rules
        self.by_id = {rule.rule_id: rule for rule in active_rules}
        if len(self.by_id) != len(active_rules):
            raise ValueError("规则库存在重复 rule_id")

        self._documents = [中文特征(rule.searchable_text()) for rule in self.rules]
        self._document_counts = [Counter(tokens) for tokens in self._documents]
        self._document_lengths = [len(tokens) for tokens in self._documents]
        self._average_length = sum(self._document_lengths) / len(self._document_lengths)
        document_frequency: Counter[str] = Counter()
        for tokens in self._documents:
            document_frequency.update(set(tokens))
        self._document_frequency = document_frequency
        self._vectors = [稳定稠密向量(rule.searchable_text()) for rule in self.rules]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> RuleKnowledgeBase:
        with Path(path).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return cls([Rule.from_dict(row) for row in rows])

    def _bm25(self, query_tokens: list[str], index: int) -> float:
        count = self._document_counts[index]
        document_length = self._document_lengths[index]
        score = 0.0
        k1, b = 1.5, 0.75
        for token in set(query_tokens):
            frequency = count.get(token, 0)
            if not frequency:
                continue
            documents_with_token = self._document_frequency[token]
            inverse_frequency = math.log(
                1
                + (len(self.rules) - documents_with_token + 0.5)
                / (documents_with_token + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * document_length / max(self._average_length, 1.0)
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        return score

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        maximum = max(values, default=0.0)
        minimum = min(values, default=0.0)
        if maximum <= minimum:
            return [float(value > 0) for value in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: str | None = None,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """检索规则并返回各阶段分数，支持独立消融。"""

        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("mode 只能是 bm25、dense 或 hybrid")
        top_k = max(1, min(int(top_k), len(self.rules)))
        query_tokens = 中文特征(query)
        query_vector = 稳定稠密向量(query)
        bm25_raw = [self._bm25(query_tokens, index) for index in range(len(self.rules))]
        dense_raw = [余弦相似度(query_vector, vector) for vector in self._vectors]
        bm25 = self._normalize(bm25_raw)
        dense = self._normalize(dense_raw)

        query_lower = query.lower()
        candidates: list[tuple[float, Rule, dict[str, float]]] = []
        for index, rule in enumerate(self.rules):
            if category_filter and rule.category != category_filter:
                continue
            keyword_hits = sum(
                keyword.lower() in query_lower for keyword in rule.keywords
            )
            keyword_score = keyword_hits / max(len(rule.keywords), 1)
            if mode == "bm25":
                retrieval_score = bm25[index]
            elif mode == "dense":
                retrieval_score = dense[index]
            else:
                retrieval_score = (
                    0.52 * bm25[index] + 0.38 * dense[index] + 0.10 * keyword_score
                )
            priority_score = max(0.0, min(rule.priority / 100.0, 1.0))
            final_score = retrieval_score
            if rerank:
                final_score = (
                    0.72 * retrieval_score
                    + 0.23 * keyword_score
                    + 0.05 * priority_score
                )
            scores = {
                "bm25": bm25[index],
                "dense": dense[index],
                "keyword": keyword_score,
                "retrieval": retrieval_score,
                "rerank": final_score,
            }
            candidates.append((final_score, rule, scores))

        candidates.sort(
            key=lambda item: (item[0], item[1].priority, item[1].rule_id), reverse=True
        )
        return [rule.public_dict(scores) for _, rule, scores in candidates[:top_k]]

    def compose(self, rule_ids: Iterable[str]) -> dict[str, Any]:
        """按 canonical_id 去重，选择高优先级版本并报告类别冲突。"""

        requested = list(dict.fromkeys(str(rule_id) for rule_id in rule_ids))
        unknown = [rule_id for rule_id in requested if rule_id not in self.by_id]
        groups: dict[str, list[Rule]] = defaultdict(list)
        for rule_id in requested:
            rule = self.by_id.get(rule_id)
            if rule:
                groups[rule.canonical_id].append(rule)

        selected = [
            max(group, key=lambda rule: (rule.priority, rule.rule_id))
            for group in groups.values()
        ]
        selected.sort(key=lambda rule: (rule.priority, rule.canonical_id), reverse=True)
        category_groups: dict[str, list[str]] = defaultdict(list)
        for rule in selected:
            if rule.category in 允许类别:
                category_groups[rule.category].append(rule.canonical_id)
        conflicts = []
        if len(category_groups) > 1:
            conflicts.append(
                {
                    "type": "category_conflict",
                    "categories": dict(category_groups),
                    "resolution": "优先保留高优先级规则，并由最终决策结合新闻主焦点处理",
                }
            )

        return {
            "canonical_rule_ids": [rule.canonical_id for rule in selected],
            "rules": [rule.public_dict() for rule in selected],
            "unknown_rule_ids": unknown,
            "conflicts": conflicts,
            "input_count": len(requested),
            "output_count": len(selected),
            "compression_ratio": 1 - len(selected) / len(requested)
            if requested
            else 0.0,
        }

    def gold_rules(
        self, article: str, label: str, maximum_subrules: int = 2
    ) -> list[str]:
        """用人工规则定义为已有标签构造可核验的 canonical rule 标注。"""

        if label not in 允许类别:
            raise ValueError(f"不支持的新闻类别：{label}")
        roots = [
            rule
            for rule in self.rules
            if rule.category == label and rule.canonical_id.endswith("-ROOT")
        ]
        if not roots:
            raise ValueError(f"类别 {label} 缺少 ROOT 规则")
        root = max(roots, key=lambda rule: rule.priority)
        article_lower = article.lower()
        scored: list[tuple[int, int, str]] = []
        for rule in self.rules:
            if rule.category != label or rule.canonical_id == root.canonical_id:
                continue
            hits = sum(keyword.lower() in article_lower for keyword in rule.keywords)
            if hits:
                scored.append((hits, rule.priority, rule.canonical_id))
        scored.sort(reverse=True)
        selected = [root.canonical_id]
        selected.extend(item[2] for item in scored[:maximum_subrules])
        return list(dict.fromkeys(selected))

    def evidence_terms(
        self, article: str, canonical_ids: Iterable[str], maximum: int = 3
    ) -> list[str]:
        """从命中的 canonical 规则中抽取确实出现在新闻里的证据词。"""

        ids = set(canonical_ids)
        terms: list[str] = []
        for rule in sorted(self.rules, key=lambda item: item.priority, reverse=True):
            if rule.canonical_id not in ids:
                continue
            for keyword in rule.keywords:
                if keyword in article and keyword not in terms:
                    terms.append(keyword)
                    if len(terms) >= maximum:
                        return terms
        return terms


def render_search_results(results: list[dict[str, Any]]) -> str:
    """把结构化检索结果压缩成适合环境观察的文本。"""

    compact = [
        {
            "rule_id": row["rule_id"],
            "canonical_id": row["canonical_id"],
            "category": row["category"],
            "title": row["title"],
            "text": row["text"],
            "score": row.get("scores", {}).get("rerank", 0.0),
        }
        for row in results
    ]
    return json.dumps(
        {"retrieved_rules": compact}, ensure_ascii=False, separators=(",", ":")
    )


def render_composition(result: dict[str, Any]) -> str:
    """把组合结果压缩成环境反馈。"""

    compact = {
        "canonical_rule_ids": result["canonical_rule_ids"],
        "rules": [
            {
                "canonical_id": row["canonical_id"],
                "category": row["category"],
                "text": row["text"],
                "conditions": row["conditions"],
                "exceptions": row["exceptions"],
            }
            for row in result["rules"]
        ],
        "conflicts": result["conflicts"],
        "compression_ratio": result["compression_ratio"],
    }
    return json.dumps(
        {"composed_policy": compact}, ensure_ascii=False, separators=(",", ":")
    )
