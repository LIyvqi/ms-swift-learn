"""离线评测检索、组合和确定性决策的分层指标与消融。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from statistics import mean
from typing import Any

项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

知识模块 = import_module("course.25_agent_r1_news.knowledge_pipeline")
RuleKnowledgeBase = 知识模块.RuleKnowledgeBase
集合指标 = 知识模块.集合指标


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def canonical_rank(results: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(row["canonical_id"] for row in results))


def reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    for rank, rule_id in enumerate(ranked, 1):
        if rule_id in expected_set:
            return 1 / rank
    return 0.0


def category_from_rules(rules: list[dict[str, Any]]) -> str:
    scores: dict[str, float] = defaultdict(float)
    for index, rule in enumerate(rules, 1):
        category = rule["category"]
        if category not in {"政治", "财经", "体育", "计算机"}:
            continue
        scores[category] += rule["priority"] / index
    return max(scores, key=scores.get) if scores else ""


def no_knowledge_prediction(article: str) -> str:
    cues = {
        "政治": ("政府", "政治", "党", "外交", "国家"),
        "财经": ("经济", "金融", "市场", "企业", "贸易"),
        "体育": ("比赛", "运动员", "体育", "球队", "冠军"),
        "计算机": ("计算机", "软件", "网络", "数据", "程序"),
    }
    scores = {
        label: sum(article.count(cue) for cue in terms) for label, terms in cues.items()
    }
    return max(scores, key=scores.get)


def full_knowledge_prediction(knowledge: Any, article: str) -> str:
    scores: dict[str, float] = defaultdict(float)
    for rule in knowledge.rules:
        if rule.category not in {"政治", "财经", "体育", "计算机"}:
            continue
        hits = sum(keyword in article for keyword in rule.keywords)
        scores[rule.category] += hits * (1 + rule.priority / 100)
    return max(scores, key=scores.get) if scores else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 Knowledge-to-Policy 新闻流水线")
    parser.add_argument(
        "--rules",
        type=Path,
        default=项目根目录 / "datasets/agent_r1_news/knowledge_rules.jsonl",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=项目根目录 / "datasets/agent_r1_news/rl_val.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=项目根目录 / "outputs/25_agent_r1_news/offline_metrics.json",
    )
    args = parser.parse_args()

    knowledge = RuleKnowledgeBase.from_jsonl(args.rules)
    rows_by_record = {}
    for row in 读取_jsonl(args.dataset):
        rows_by_record.setdefault(row["record_id"], row)
    rows = list(rows_by_record.values())

    retrieval_metrics: dict[str, dict[str, list[float]]] = {}
    decision_correct = defaultdict(list)
    composition_scores = []
    for mode, rerank in (
        ("bm25", False),
        ("dense", False),
        ("hybrid", False),
        ("hybrid", True),
    ):
        name = f"{mode}{'_rerank' if rerank else ''}"
        retrieval_metrics[name] = defaultdict(list)
        for row in rows:
            config = row["env_config"]
            article = config["article"]
            expected = config["gold_rule_ids"]
            results = knowledge.search(article, top_k=10, mode=mode, rerank=rerank)
            ranked = canonical_rank(results)
            for k in (5, 10):
                result = 集合指标(ranked[:k], expected)
                retrieval_metrics[name][f"recall@{k}"].append(result["recall"])
                retrieval_metrics[name][f"precision@{k}"].append(result["precision"])
            retrieval_metrics[name]["mrr"].append(reciprocal_rank(ranked, expected))

            if name == "hybrid_rerank":
                composed = knowledge.compose([item["rule_id"] for item in results[:8]])
                composition_scores.append(
                    集合指标(composed["canonical_rule_ids"], expected)["f1"]
                )
                decision_correct["rag_top1"].append(
                    float(results[0]["category"] == config["label"])
                )
                decision_correct["rag_composer"].append(
                    float(category_from_rules(composed["rules"]) == config["label"])
                )
                decision_correct["no_knowledge_keyword"].append(
                    float(no_knowledge_prediction(article) == config["label"])
                )
                decision_correct["full_knowledge_no_retrieval"].append(
                    float(
                        full_knowledge_prediction(knowledge, article) == config["label"]
                    )
                )

    summary = {
        "samples": len(rows),
        "retrieval": {
            name: {metric: mean(values) for metric, values in metrics.items()}
            for name, metrics in retrieval_metrics.items()
        },
        "composition": {"canonical_f1@8": mean(composition_scores)},
        "decision_ablation": {
            name: mean(values) for name, values in decision_correct.items()
        },
        "说明": "该离线决策消融是确定性流水线，不是训练后 LLM 准确率。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
