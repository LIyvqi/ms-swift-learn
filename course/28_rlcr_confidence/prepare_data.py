#!/usr/bin/env python3
"""构造 RLCR、独立 Verifier 和 OOD 共用的严格拆分数据。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[2]
原分类目录 = 项目根目录 / "datasets/fudan_news_4class"
原始CSV = 项目根目录 / "downloads/datasets/zh_cls_fudan-news/zh_cls_fudan-news.csv"
输出目录 = 项目根目录 / "datasets/confidence_news"

标签 = ("政治", "财经", "体育", "计算机")
源标签映射 = {"Politics": "政治", "Economy": "财经", "Sports": "体育", "Computer": "计算机"}
OOD源标签 = ("Art", "Agriculture", "History", "Space", "Enviornment")
相邻错误 = {"政治": "财经", "财经": "政治", "体育": "政治", "计算机": "财经"}

RLCR系统提示 = (
    "你是会校准置信度的中文新闻分类器。可选类别只有：政治、财经、体育、计算机。"
    "严格输出 <answer>类别</answer><confidence>0.00到1.00</confidence>。"
    "confidence 表示本次类别预测正确的概率，不要输出其他内容。"
)
验证器系统提示 = (
    "你是与分类模型参数独立的新闻候选验证器。"
    "根据新闻与候选类别判断候选是否正确，不得假设候选一定属于四个目标类别。"
)


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """按稳定字段顺序写入 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 摘要(path: Path) -> str:
    """计算文件 SHA256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def 规范正文(text: str, max_chars: int = 600) -> str:
    """复用四分类课程的清洗口径。"""

    text = re.sub(r"^\s*输入:\s*", "", text)
    text = re.sub(r"\s*分类:\s*.*$", "", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def 取新闻用户消息(row: dict[str, Any]) -> str:
    """从旧数据中保留新闻用户问题。"""

    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def RLCR记录(row: dict[str, Any], assistant: bool = False) -> dict[str, Any]:
    """转换为类别与数值置信度联合输出格式。"""

    messages = [
        {"role": "system", "content": RLCR系统提示},
        {"role": "user", "content": 取新闻用户消息(row)},
    ]
    if assistant:
        # 均匀占位值由 record_id 哈希决定，与类别、难度和正确性无关，只教数值语法。
        bucket = int(hashlib.sha256(row["record_id"].encode()).hexdigest()[:8], 16) % 9 + 1
        placeholder = bucket / 10
        messages.append({
            "role": "assistant",
            "content": f"<answer>{row['label']}</answer><confidence>{placeholder:.2f}</confidence>",
        })
    return {
        "messages": messages,
        "label": row["label"],
        "source_label": row.get("source_label"),
        "record_id": row["record_id"],
        "is_ood": False,
    }


def 验证器用户消息(user_text: str, candidate: str) -> str:
    """构造不包含金标签字段的候选校验输入。"""

    return f"{user_text}\n\n待验证的候选类别：{candidate}"


def 验证器成对记录(
    row: dict[str, Any], candidate: str, correct: bool, suffix: str,
) -> dict[str, Any]:
    """构造正确/错误 verdict 互为 chosen/rejected 的 RM 数据。"""

    verdict = "CORRECT" if correct else "INCORRECT"
    rejected = "INCORRECT" if correct else "CORRECT"
    messages = [
        {"role": "system", "content": 验证器系统提示},
        {"role": "user", "content": 验证器用户消息(取新闻用户消息(row), candidate)},
        {"role": "assistant", "content": f"<verdict>{verdict}</verdict>"},
    ]
    return {
        "messages": messages,
        "rejected_response": f"<verdict>{rejected}</verdict>",
        "margin": 1.0,
        "record_id": f"{row['record_id']}-{suffix}",
        "source_record_id": row["record_id"],
        "candidate_label": candidate,
        "candidate_correct": correct,
        "gold_label": row.get("label", "OOD"),
        "is_ood": bool(row.get("is_ood", False)),
    }


def 构造验证器对(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对 ID 样本同时构造正例和困难错类负例。"""

    pairs = []
    for row in rows:
        if row.get("is_ood"):
            candidate = row["candidate_label"]
            pairs.append(验证器成对记录(row, candidate, False, "ood"))
        else:
            gold = row["label"]
            pairs.append(验证器成对记录(row, gold, True, "positive"))
            pairs.append(验证器成对记录(row, 相邻错误[gold], False, "hard-negative"))
    return pairs


def 构造OOD(seed: int = 2026) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """从真实非目标新闻类别构造三份不交叉 OOD。"""

    if not 原始CSV.is_file():
        raise FileNotFoundError(f"缺少已下载的复旦原始数据：{原始CSV}")
    text_labels: dict[str, set[str]] = defaultdict(set)
    with 原始CSV.open(encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            source_label = source["answer"].strip()
            if source_label in OOD源标签:
                text = 规范正文(source["prompt"])
                if text:
                    text_labels[text].add(source_label)
    grouped: dict[str, list[str]] = defaultdict(list)
    for text, labels in text_labels.items():
        if len(labels) == 1:
            grouped[next(iter(labels))].append(text)

    randomizer = random.Random(seed)
    splits = {"train": [], "calibration": [], "test": []}
    for source_label in OOD源标签:
        texts = grouped[source_label].copy()
        randomizer.shuffle(texts)
        if len(texts) < 60:
            raise RuntimeError(f"OOD 类别 {source_label} 可用样本不足 60 条")
        for split_index, split in enumerate(splits):
            for index, text in enumerate(texts[split_index * 20:(split_index + 1) * 20]):
                candidate = 标签[(index + split_index + OOD源标签.index(source_label)) % len(标签)]
                splits[split].append({
                    "messages": [
                        {"role": "system", "content": RLCR系统提示},
                        {"role": "user", "content": f"请判断下面新闻的主要类别。\n\n新闻：{text}"},
                    ],
                    "label": "OOD",
                    "source_label": source_label,
                    "record_id": f"{source_label}-ood-{split}-{index:04d}",
                    "is_ood": True,
                    "candidate_label": candidate,
                })
    for rows in splits.values():
        randomizer.shuffle(rows)
    return splits["train"], splits["calibration"], splits["test"]


def 主程序() -> None:
    """生成全部训练、校准和最终测试文件。"""

    randomizer = random.Random(2026)
    sft_source = 读_jsonl(原分类目录 / "sft_train.jsonl")
    rl_source = 读_jsonl(原分类目录 / "rl_train.jsonl")
    val_source = 读_jsonl(原分类目录 / "val.jsonl")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in val_source:
        grouped[row["label"]].append(row)
    calibration_source, test_source = [], []
    for label in 标签:
        rows = sorted(grouped[label], key=lambda row: row["record_id"])
        randomizer.shuffle(rows)
        calibration_source.extend(rows[:40])
        test_source.extend(rows[40:80])
    randomizer.shuffle(calibration_source)
    randomizer.shuffle(test_source)

    sft_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sft_source:
        sft_grouped[row["label"]].append(row)
    sft_train_source, sft_val_source = [], []
    for label in 标签:
        rows = sorted(sft_grouped[label], key=lambda row: row["record_id"])
        randomizer.shuffle(rows)
        # 格式热身验证集必须带 assistant 标签，不能误用只有问题的 RL 校准集。
        sft_val_source.extend(rows[:10])
        sft_train_source.extend(rows[10:])
    randomizer.shuffle(sft_train_source)
    randomizer.shuffle(sft_val_source)

    rlcr_sft = [RLCR记录(row, assistant=True) for row in sft_train_source]
    rlcr_sft_val = [RLCR记录(row, assistant=True) for row in sft_val_source]
    rlcr_train = [RLCR记录(row) for row in rl_source]
    calibration = [RLCR记录(row) for row in calibration_source]
    test = [RLCR记录(row) for row in test_source]
    smoke_counts = defaultdict(int)
    smoke = []
    for row in rlcr_train:
        if smoke_counts[row["label"]] < 4:
            smoke.append(row)
            smoke_counts[row["label"]] += 1

    ood_train, ood_calibration, ood_test = 构造OOD()
    verifier_train = 构造验证器对([*rlcr_train, *ood_train])
    verifier_val = 构造验证器对([*calibration, *ood_calibration])
    verifier_test = 构造验证器对([*test, *ood_test])
    randomizer.shuffle(verifier_train)
    randomizer.shuffle(verifier_val)
    randomizer.shuffle(verifier_test)

    files = {
        "rlcr_sft.jsonl": rlcr_sft,
        "rlcr_sft_val.jsonl": rlcr_sft_val,
        "rlcr_train.jsonl": rlcr_train,
        "rlcr_smoke.jsonl": smoke,
        "calibration.jsonl": calibration,
        "test.jsonl": test,
        "ood_train.jsonl": ood_train,
        "ood_calibration.jsonl": ood_calibration,
        "ood_test.jsonl": ood_test,
        "verifier_train.jsonl": verifier_train,
        "verifier_val.jsonl": verifier_val,
        "verifier_test.jsonl": verifier_test,
        "verifier_smoke.jsonl": verifier_train[:16],
    }
    for name, rows in files.items():
        写_jsonl(输出目录 / name, rows)
    manifest = {
        "随机种子": 2026,
        "目标标签": list(标签),
        "OOD源标签": list(OOD源标签),
        "文件": {
            name: {"样本数": len(rows), "SHA256": 摘要(输出目录 / name)}
            for name, rows in files.items()
        },
    }
    (输出目录 / "checksums.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
