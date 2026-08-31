#!/usr/bin/env python3
"""实现面向独立规则库、Case 库和知识库的分层联邦检索。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 中的全部非空记录。"""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 检索特征(text: str) -> list[str]:
    """同时保留英文词、数字、中文单字、中文双字和短字符片段。"""

    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    features = re.findall(r"[a-z0-9][a-z0-9_,.+-]*", normalized)
    for segment in re.findall(r"[\u3400-\u9fff]+", normalized):
        features.extend(segment)
        features.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    compact = re.sub(r"\s+", "", normalized)
    features.extend(
        f"片段:{compact[index:index + 3]}"
        for index in range(max(0, len(compact) - 2))
    )
    return features


def 稳定哈希向量(text: str, dimensions: int = 256) -> list[float]:
    """构造无需下载额外模型的确定性向量基线，不冒充语义嵌入。"""

    vector = [0.0] * dimensions
    for feature in 检索特征(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        vector[value % dimensions] += 1.0 if value & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def 余弦相似度(left: list[float], right: list[float]) -> float:
    """计算两个相同维度向量的余弦相似度。"""

    return sum(a * b for a, b in zip(left, right))


class 混合索引:
    """使用 BM25 与哈希向量分别排序，再用 RRF 融合名次。"""

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
        self.vectors = [稳定哈希向量(text) for text in self.texts]

    def _bm25(self, query_tokens: list[str], index: int) -> float:
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
    ) -> list[tuple[int, float, dict[str, float]]]:
        """返回 RRF 排名及可审计的 BM25、向量和名次分数。"""

        indices = list(candidates) if candidates is not None else list(range(len(self.texts)))
        if not indices:
            return []
        query_tokens = 检索特征(query)
        query_vector = 稳定哈希向量(query)
        bm25 = {index: self._bm25(query_tokens, index) for index in indices}
        dense = {index: 余弦相似度(query_vector, self.vectors[index]) for index in indices}
        lexical_rank = sorted(indices, key=lambda index: (-bm25[index], index))
        dense_rank = sorted(indices, key=lambda index: (-dense[index], index))
        lexical_position = {index: rank for rank, index in enumerate(lexical_rank, 1)}
        dense_position = {index: rank for rank, index in enumerate(dense_rank, 1)}
        ranked = []
        for index in indices:
            rrf = 1 / (60 + lexical_position[index]) + 1 / (60 + dense_position[index])
            ranked.append(
                (
                    index,
                    rrf,
                    {
                        "bm25": bm25[index],
                        "hash_dense": dense[index],
                        "rrf": rrf,
                    },
                )
            )
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[: max(1, min(int(top_k), len(ranked)))]


@dataclass(frozen=True)
class 记忆记录:
    """不同物理后端向 Agent 暴露的统一只读记录。"""

    memory_id: str
    source_id: str
    memory_type: str
    path: tuple[str, ...]
    title: str
    search_text: str
    content: dict[str, Any]
    categories: tuple[str, ...]
    metadata: dict[str, Any]

    def 公开(self, scores: dict[str, float] | None = None) -> dict[str, Any]:
        """返回包含来源和层级路径的可审计结果。"""

        result = {
            "memory_id": self.memory_id,
            "source_id": self.source_id,
            "memory_type": self.memory_type,
            "path": "/".join(self.path),
            "title": self.title,
            "content": self.content,
            "categories": list(self.categories),
            "metadata": self.metadata,
        }
        if scores:
            result["scores"] = {key: round(value, 6) for key, value in scores.items()}
        return result


@dataclass(frozen=True)
class 层级节点:
    """用于先定位库和目录、再检索具体记录的目录摘要。"""

    source_id: str
    path: tuple[str, ...]
    summary: str
    record_indices: tuple[int, ...]
    categories: tuple[str, ...]


class 独立库连接器(ABC):
    """屏蔽 JSONL、SQLite 和目录文档之间的物理差异。"""

    def __init__(self, source: dict[str, Any], location: Path):
        self.source = source
        self.source_id = str(source["source_id"])
        self.location = location
        self.records = self._load_records()
        if not self.records:
            raise ValueError(f"{self.source_id} 没有可检索记录：{location}")
        self.by_id = {record.memory_id: record for record in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError(f"{self.source_id} 存在重复 memory_id")
        self.record_index = 混合索引(record.search_text for record in self.records)
        self.nodes = self._build_nodes()
        self.node_index = 混合索引(node.summary for node in self.nodes)

    @abstractmethod
    def _load_records(self) -> list[记忆记录]:
        """从当前后端读取并标准化记录。"""

    def _build_nodes(self) -> list[层级节点]:
        grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            for depth in range(1, len(record.path) + 1):
                grouped[record.path[:depth]].append(index)
        nodes = []
        for path, indices in sorted(grouped.items()):
            titles = [self.records[index].title for index in indices[:12]]
            categories = sorted(
                {category for index in indices for category in self.records[index].categories}
            )
            summary = " ".join(
                [
                    str(self.source.get("name", self.source_id)),
                    str(self.source.get("description", "")),
                    *path,
                    *categories,
                    *titles,
                ]
            )
            nodes.append(
                层级节点(
                    source_id=self.source_id,
                    path=path,
                    summary=summary,
                    record_indices=tuple(indices),
                    categories=tuple(categories),
                )
            )
        return nodes

    def 定位(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """融合目录摘要和高相关子记录信号，再定位可搜索的深层节点。"""

        minimum_depth = int(self.source.get("minimum_search_depth", 1))
        searchable_indices = [
            index for index, node in enumerate(self.nodes) if len(node.path) >= minimum_depth
        ]
        node_ranked = self.node_index.搜索(
            query,
            max(top_k * 3, top_k),
            candidates=searchable_indices,
        )
        node_scores = {
            index: (score, detail) for index, score, detail in node_ranked
        }
        node_by_path = {node.path: index for index, node in enumerate(self.nodes)}
        child_support: dict[int, float] = defaultdict(float)
        child_hits = self.record_index.搜索(query, max(32, top_k * 8))
        for rank, (record_index, _, _) in enumerate(child_hits, 1):
            record = self.records[record_index]
            for depth in range(minimum_depth, len(record.path) + 1):
                node_index = node_by_path.get(record.path[:depth])
                if node_index is not None:
                    child_support[node_index] += 1 / (60 + rank)
        candidates = []
        for index in searchable_indices:
            node = self.nodes[index]
            node_score, detail = node_scores.get(
                index,
                (0.0, {"bm25": 0.0, "hash_dense": 0.0, "rrf": 0.0}),
            )
            support = child_support.get(index, 0.0)
            depth_bonus = min(len(node.path), 6) * 0.00015
            score = node_score + 1.5 * support + depth_bonus
            candidates.append((score, node, {**detail, "child_support": support}))
        candidates.sort(key=lambda item: (-item[0], "/".join(item[1].path)))
        return [
            {
                "source_id": self.source_id,
                "path": "/".join(node.path),
                "depth": len(node.path),
                "record_count": len(node.record_indices),
                "categories": list(node.categories),
                "summary": node.summary[:420],
                "local_score": score,
                "scores": detail,
            }
            for score, node, detail in candidates[:top_k]
        ]

    def 搜索(self, path: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """只在给定目录及其子树内搜索具体记录。"""

        prefix = tuple(part for part in path.split("/") if part)
        candidates = [
            index
            for index, record in enumerate(self.records)
            if record.path[: len(prefix)] == prefix
        ]
        if not candidates:
            raise ValueError(f"{self.source_id} 中不存在目录：{path}")
        ranked = self.record_index.搜索(query, max(top_k * 4, top_k), candidates)
        selected = []
        seen_entities = set()
        for index, _, detail in ranked:
            record = self.records[index]
            entity_id = str(record.metadata.get("record_id", record.memory_id))
            if entity_id in seen_entities:
                continue
            selected.append(record.公开(detail))
            seen_entities.add(entity_id)
            if len(selected) >= top_k:
                break
        return selected


class JSONL规则库连接器(独立库连接器):
    """连接独立的版本化 JSONL 规则库。"""

    def _load_records(self) -> list[记忆记录]:
        records = []
        for row in 读取_jsonl(self.location):
            if row.get("status") != "active":
                continue
            path = (
                "内容审核政策",
                str(row.get("scope", "通用平台")),
                str(row["route"]),
                str(row["category"]),
                f"v{row['version']}",
            )
            search_text = " ".join(
                str(value)
                for value in (
                    row["category"],
                    row["name_zh"],
                    row["definition_zh"],
                    row["definition_en"],
                    *row["inclusions"],
                    *row["exceptions"],
                )
            )
            records.append(
                记忆记录(
                    memory_id=f"rule:{row['rule_id']}@v{row['version']}",
                    source_id=self.source_id,
                    memory_type="rule",
                    path=path,
                    title=str(row["name_zh"]),
                    search_text=search_text,
                    content={
                        "definition": row["definition_zh"],
                        "inclusions": row["inclusions"],
                        "exceptions": row["exceptions"],
                        "priority": row["priority"],
                    },
                    categories=(str(row["category"]),),
                    metadata={
                        "rule_id": row["rule_id"],
                        "version": row["version"],
                        "status": row["status"],
                        "source": row["source"],
                    },
                )
            )
        return records


class SQLite案例库连接器(独立库连接器):
    """连接独立 SQLite Case 库，模拟可替换的业务数据库。"""

    def _load_records(self) -> list[记忆记录]:
        connection = sqlite3.connect(f"file:{self.location}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT memory_id, path_json, title, search_text, content_json, "
                "categories_json, metadata_json FROM memory_items ORDER BY memory_id"
            ).fetchall()
        finally:
            connection.close()
        return [
            记忆记录(
                memory_id=str(row["memory_id"]),
                source_id=self.source_id,
                memory_type="case",
                path=tuple(json.loads(row["path_json"])),
                title=str(row["title"]),
                search_text=str(row["search_text"]),
                content=json.loads(row["content_json"]),
                categories=tuple(json.loads(row["categories_json"])),
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]


class 目录知识库连接器(独立库连接器):
    """连接具有真实目录层级的独立知识文档库。"""

    def _load_records(self) -> list[记忆记录]:
        records = []
        for path in sorted(self.location.rglob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            relative_parts = path.relative_to(self.location).with_suffix("").parts
            hierarchy = tuple(str(item) for item in row.get("path", relative_parts))
            records.append(
                记忆记录(
                    memory_id=str(row["memory_id"]),
                    source_id=self.source_id,
                    memory_type="knowledge",
                    path=hierarchy,
                    title=str(row["title"]),
                    search_text=" ".join(
                        [
                            str(row["title"]),
                            str(row["body"]),
                            *[str(item) for item in row.get("aliases", [])],
                            *hierarchy,
                        ]
                    ),
                    content={
                        "body": row["body"],
                        "aliases": row.get("aliases", []),
                    },
                    categories=tuple(str(item) for item in row.get("categories", [])),
                    metadata=dict(row.get("metadata", {})),
                )
            )
        return records


连接器类型 = {
    "jsonl_rules": JSONL规则库连接器,
    "sqlite_cases": SQLite案例库连接器,
    "directory_knowledge": 目录知识库连接器,
}


class 分层记忆网关:
    """只统一访问协议，保留每个库自己的后端、目录和记录身份。"""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path.resolve()
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        base_dir = self.registry_path.parent
        self.sources = registry["sources"]
        self.connectors: dict[str, 独立库连接器] = {}
        for source in self.sources:
            connector_name = str(source["connector"])
            connector_class = 连接器类型.get(connector_name)
            if connector_class is None:
                raise ValueError(f"未知连接器类型：{connector_name}")
            location = (base_dir / str(source["location"])).resolve()
            connector = connector_class(source, location)
            self.connectors[connector.source_id] = connector
        source_texts = [
            " ".join(
                [
                    str(source["source_id"]),
                    str(source.get("name", "")),
                    str(source.get("description", "")),
                    *[str(item) for item in source.get("search_terms", [])],
                ]
            )
            for source in self.sources
        ]
        self.source_index = 混合索引(source_texts)
        self.by_id = {
            memory_id: record
            for connector in self.connectors.values()
            for memory_id, record in connector.by_id.items()
        }

    def 定位(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        """先选知识源，再在候选源的目录摘要中定位路径。"""

        source_ranked = self.source_index.搜索(query, len(self.sources))
        results = []
        first_by_source = []
        for source_rank, (source_index, _, source_detail) in enumerate(source_ranked, 1):
            source = self.sources[source_index]
            source_id = str(source["source_id"])
            local = self.connectors[source_id].定位(query, max(3, top_k))
            current_source = []
            for local_rank, row in enumerate(local, 1):
                federated_score = 1 / (60 + source_rank) + 1 / (60 + local_rank)
                item = {
                    **row,
                    "source_name": source.get("name", source_id),
                    "federated_score": federated_score,
                    "source_scores": source_detail,
                }
                results.append(item)
                current_source.append(item)
            if current_source:
                first_by_source.append(current_source[0])
        results.sort(
            key=lambda row: (
                -row["federated_score"],
                -row["depth"],
                row["source_id"],
                row["path"],
            )
        )
        # 联邦候选先保证每个独立库都有一个入口，剩余名额再按全局 RRF 排序。
        selected = list(first_by_source)
        selected.extend(row for row in results if row not in selected)
        return selected[: max(1, min(int(top_k), len(selected)))]

    def 搜索(
        self, source_id: str, path: str, query: str, top_k: int = 6
    ) -> list[dict[str, Any]]:
        """把检索委派给指定独立库，不跨库偷取结果。"""

        connector = self.connectors.get(source_id)
        if connector is None:
            raise ValueError(f"未知知识源：{source_id}")
        return connector.搜索(path, query, top_k)

    def 记录(self, memory_id: str) -> 记忆记录 | None:
        """按统一身份读取已加载记录，仅供环境核验引用。"""

        return self.by_id.get(memory_id)

    def 摘要(self) -> dict[str, Any]:
        """汇总后端、层级和记录数量，便于数据审计。"""

        return {
            "sources": [
                {
                    "source_id": source_id,
                    "connector": connector.source["connector"],
                    "records": len(connector.records),
                    "unique_entities": len(
                        {
                            str(record.metadata.get("record_id", record.memory_id))
                            for record in connector.records
                        }
                    ),
                    "hierarchy_nodes": len(connector.nodes),
                    "max_depth": max(len(record.path) for record in connector.records),
                }
                for source_id, connector in sorted(self.connectors.items())
            ],
            "total_records": len(self.by_id),
        }
