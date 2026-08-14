#!/usr/bin/env python3
"""独立 Verifier 的成对数据、消息边界和拆分测试。"""

from __future__ import annotations

import json
from pathlib import Path

from evaluate_verifier import 构造verdict消息


项目根目录 = Path(__file__).resolve().parents[2]


def 读(path: Path) -> list[dict]:
    """读取 JSONL。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def test_正负候选的chosen方向相反() -> None:
    """正确候选应偏好 CORRECT，错误候选应偏好 INCORRECT。"""

    rows = 读(项目根目录 / "datasets/confidence_news/verifier_train.jsonl")
    positive = next(row for row in rows if row["candidate_correct"])
    negative = next(row for row in rows if not row["candidate_correct"] and not row["is_ood"])
    assert positive["messages"][-1]["content"] == "<verdict>CORRECT</verdict>"
    assert positive["rejected_response"] == "<verdict>INCORRECT</verdict>"
    assert negative["messages"][-1]["content"] == "<verdict>INCORRECT</verdict>"
    assert negative["rejected_response"] == "<verdict>CORRECT</verdict>"


def test_OOD只能作为错误候选() -> None:
    """非目标类别新闻不能被任意四类候选标为正确。"""

    rows = 读(项目根目录 / "datasets/confidence_news/verifier_train.jsonl")
    ood = [row for row in rows if row["is_ood"]]
    assert len(ood) == 100
    assert all(not row["candidate_correct"] for row in ood)
    assert all(row["gold_label"] == "OOD" for row in ood)


def test_评测消息不把金标签传给Verifier() -> None:
    """线上 Verifier 只看新闻、候选和待打分 verdict。"""

    row = 读(项目根目录 / "datasets/confidence_news/test.jsonl")[0]
    messages = 构造verdict消息(row, "财经", "CORRECT")
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "gold_label" not in serialized
    assert "candidate_correct" not in serialized
    assert "待验证的候选类别：财经" in serialized


def test_验证器三拆分的源新闻不交叉() -> None:
    """一篇新闻的正负候选可成组，但不得跨 train/val/test。"""

    root = 项目根目录 / "datasets/confidence_news"
    partitions = [
        {row["source_record_id"] for row in 读(root / filename)}
        for filename in ("verifier_train.jsonl", "verifier_val.jsonl", "verifier_test.jsonl")
    ]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]


def test_候选集合不含非法分类() -> None:
    """候选标签仅允许四个主任务类别。"""

    rows = 读(项目根目录 / "datasets/confidence_news/verifier_test.jsonl")
    assert {row["candidate_label"] for row in rows} <= {"政治", "财经", "体育", "计算机"}


def test_部署消息只拼接一次候选() -> None:
    """从源新闻构造 verdict 时不得重复附加候选字段。"""

    row = 读(项目根目录 / "datasets/confidence_news/test.jsonl")[0]
    messages = 构造verdict消息(row, "体育", "INCORRECT")
    user = next(message["content"] for message in messages if message["role"] == "user")
    assert user.count("待验证的候选类别：") == 1
    assert user.endswith("待验证的候选类别：体育")
