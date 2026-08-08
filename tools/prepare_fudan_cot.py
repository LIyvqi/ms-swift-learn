#!/usr/bin/env python3
"""根据人工证据标注构造 CoT-SFT、CoT-RLOO 和两种验证视图。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


标签 = ("政治", "财经", "体育", "计算机")
证据分隔符 = "|||"
系统提示 = (
    "你是中文新闻分类器。可选类别只有：政治、财经、体育、计算机。"
    "先在<think>和</think>之间用一至三句话引用新闻中的关键信息并判断主题，"
    "再单独输出一个形如\\boxed{类别}的最终答案。推理必须简洁，不能照抄整篇新闻。"
)


def 读取_jsonl(path: Path) -> list[dict]:
    """读取非空 JSONL 记录。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def 文件摘要(path: Path) -> str:
    """计算文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 提取新闻(user_content: str) -> str:
    """从 Direct 分类提示中提取新闻正文。"""
    prefix = "请判断下面新闻的主要类别。\n\n新闻："
    suffix = "\n\n只输出一个类别答案。"
    if not user_content.startswith(prefix) or not user_content.endswith(suffix):
        raise ValueError("原始 user 提示不符合预期模板")
    return user_content[len(prefix) : -len(suffix)]


def 构造用户消息(news: str) -> str:
    """构造要求显式证据推理的用户提示。"""
    return (
        "请判断下面新闻的主要类别。\n\n"
        f"新闻：{news}\n\n"
        "请先引用正文中的关键词进行简短分析，再给出最终类别。"
    )


def 构造参考理由(label: str, terms: list[str]) -> str:
    """把人工证据词转换成简洁、可审计的参考理由。"""
    quoted = "、".join(f"“{term}”" for term in terms)
    return f"文中出现{quoted}等关键信息，核心内容指向{label}主题，因此应归入{label}类。"


def 构造标注记录(source: dict, annotation: dict, include_answer: bool) -> dict:
    """构造一条带证据字段的 CoT 记录。"""
    label = source["label"]
    terms = annotation["evidence_terms"]
    news = 提取新闻(source["messages"][1]["content"])
    for term in terms:
        if term not in news:
            raise ValueError(f"{source['record_id']} 的证据词不在正文中：{term}")
    reason = 构造参考理由(label, terms)
    messages = [
        {"role": "system", "content": 系统提示},
        {"role": "user", "content": 构造用户消息(news)},
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>\n{reason}\n</think>\n\\boxed{{{label}}}",
            }
        )
    return {
        "messages": messages,
        "label": label,
        "evidence_terms": 证据分隔符.join(terms),
        "reference_reason": reason,
        "source_record_id": source["record_id"],
    }


def 构造通用验证记录(source: dict) -> dict:
    """把原有独立验证集转换为 CoT 提示，不伪造人工证据标注。"""
    label = source["label"]
    news = 提取新闻(source["messages"][1]["content"])
    reason = f"这篇新闻的核心内容属于{label}主题，因此应归入{label}类。"
    return {
        "messages": [
            {"role": "system", "content": 系统提示},
            {"role": "user", "content": 构造用户消息(news)},
            {
                "role": "assistant",
                "content": f"<think>\n{reason}\n</think>\n\\boxed{{{label}}}",
            },
        ],
        "label": label,
        "source_record_id": source["record_id"],
    }


def 写入_jsonl(path: Path, rows: list[dict]) -> None:
    """以稳定字段顺序写入 JSONL。"""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 50 条复旦新闻证据 CoT 教学数据")
    parser.add_argument("--source-rl", type=Path, required=True)
    parser.add_argument("--source-val", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_rl = {row["record_id"]: row for row in 读取_jsonl(args.source_rl)}
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    train_annotations = annotations["训练"]
    heldout_annotations = annotations["留出"]
    all_annotations = train_annotations + heldout_annotations
    ids = [item["record_id"] for item in all_annotations]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("annotations.json 必须包含 50 个互不重复的 record_id")

    for item in all_annotations:
        if item["record_id"] not in source_rl:
            raise KeyError(f"RL 源数据中找不到：{item['record_id']}")
        if len(item["evidence_terms"]) != 3 or len(set(item["evidence_terms"])) != 3:
            raise ValueError(f"每条记录必须有三个不同证据词：{item['record_id']}")

    train_sources = [source_rl[item["record_id"]] for item in train_annotations]
    heldout_sources = [source_rl[item["record_id"]] for item in heldout_annotations]
    if Counter(row["label"] for row in train_sources) != Counter({label: 10 for label in 标签}):
        raise ValueError("40 条训练标注必须四类各 10 条")
    if Counter(row["label"] for row in heldout_sources) != Counter(
        {"政治": 3, "财经": 3, "体育": 2, "计算机": 2}
    ):
        raise ValueError("10 条留出标注的类别数量不符合设计")

    sft_rows = [
        构造标注记录(source_rl[item["record_id"]], item, True) for item in train_annotations
    ]
    rl_rows = [
        构造标注记录(source_rl[item["record_id"]], item, False) for item in train_annotations
    ]
    heldout_rows = [
        构造标注记录(source_rl[item["record_id"]], item, True) for item in heldout_annotations
    ]
    smoke_rows = []
    smoke_counts: Counter[str] = Counter()
    for row in rl_rows:
        if smoke_counts[row["label"]] < 2:
            smoke_rows.append(row)
            smoke_counts[row["label"]] += 1
    val_rows = [构造通用验证记录(row) for row in 读取_jsonl(args.source_val)]

    args.output.mkdir(parents=True, exist_ok=True)
    files = {
        "sft_train.jsonl": sft_rows,
        "rl_train.jsonl": rl_rows,
        "rl_smoke.jsonl": smoke_rows,
        "evidence_val.jsonl": heldout_rows,
        "cot_val_320.jsonl": val_rows,
    }
    for name, rows in files.items():
        写入_jsonl(args.output / name, rows)

    manifest = {
        "数据来源": "datasets/fudan_news_4class/rl_train.jsonl 与 val.jsonl",
        "标注方式": "人工筛选语义明确样本并标注三个原文证据词，参考理由由固定模板生成",
        "证据分隔符": 证据分隔符,
        "文件": {
            name: {"样本数": len(rows), "SHA-256": 文件摘要(args.output / name)}
            for name, rows in files.items()
        },
    }
    (args.output / "checksums.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
