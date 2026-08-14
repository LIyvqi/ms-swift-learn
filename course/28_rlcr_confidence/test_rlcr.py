#!/usr/bin/env python3
"""RLCR 数据、奖励、校准和拒答阈值测试。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from confidence_metrics import Platt校准器, 汇总置信指标, 选择阈值
from rlcr_rewards import RLCRAccuracy, RLCRBrier, RLCRFormat, RLCRLogScore, 解析答案, 解析置信度


项目根目录 = Path(__file__).resolve().parents[2]


def 读(path: Path) -> list[dict]:
    """读取测试 JSONL。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def test_解析器拒绝越界与非数值置信度() -> None:
    """数值协议不得把字符串或 100 当成概率。"""

    good = "<answer>财经</answer><confidence>0.73</confidence>"
    assert 解析答案(good) == "财经"
    assert 解析置信度(good) == 0.73
    assert 解析置信度("<confidence>100</confidence>") is None
    assert 解析置信度("<confidence>很高</confidence>") is None


def test_Brier和对数奖励鼓励如实置信() -> None:
    """正确答案高置信、错误答案低置信应获得更高奖励。"""

    correct_high = "<answer>财经</answer><confidence>0.90</confidence>"
    correct_low = "<answer>财经</answer><confidence>0.10</confidence>"
    wrong_high = "<answer>政治</answer><confidence>0.90</confidence>"
    wrong_low = "<answer>政治</answer><confidence>0.10</confidence>"
    brier = RLCRBrier()
    log_score = RLCRLogScore()
    assert brier([correct_high], ["财经"])[0] > brier([correct_low], ["财经"])[0]
    assert brier([wrong_low], ["财经"])[0] > brier([wrong_high], ["财经"])[0]
    assert log_score([correct_high], ["财经"])[0] > log_score([correct_low], ["财经"])[0]
    assert log_score([wrong_low], ["财经"])[0] > log_score([wrong_high], ["财经"])[0]


def test_正确性与格式奖励互不冒充() -> None:
    """类别正确但缺少置信度时，只能得正确性奖励。"""

    text = "\\boxed{财经}"
    assert RLCRAccuracy()([text], ["财经"]) == [1.0]
    assert RLCRFormat()([text]) == [0.0]


def test_Platt校准器能纠正单调过度自信() -> None:
    """拟合后应保留高低顺序并降低高分。"""

    raw = [0.60, 0.70, 0.80, 0.90] * 10
    targets = [0, 0, 1, 1] * 10
    calibrator = Platt校准器.拟合(raw, targets, steps=2000)
    values = [calibrator.预测(value) for value in raw[:4]]
    assert all(0 < value < 1 for value in values)
    assert values == sorted(values)
    assert values[-1] < 0.99


def test_Platt全正校准集使用平滑常数退化() -> None:
    """没有错误样本时不能拟合伪斜率，应输出接近一但非一的常数。"""

    calibrator = Platt校准器.拟合([0.2, 0.8, 0.9], [1, 1, 1])
    assert calibrator.weight == 0.0
    assert calibrator.预测(0.1) == calibrator.预测(0.9)
    assert 0.7 < calibrator.预测(0.5) < 1.0


def test_风险阈值只在可行覆盖中选择() -> None:
    """最高置信样本无错时应得到非空选择集。"""

    confidence = [0.95, 0.90, 0.30, 0.20]
    targets = [1, 1, 0, 0]
    threshold = 选择阈值(confidence, targets, maximum_risk=0.0)
    summary = 汇总置信指标(confidence, targets, threshold)
    assert math.isclose(threshold, 0.90)
    assert summary["coverage"] == 0.5
    assert summary["selective_risk"] == 0.0


def test_数据四段不泄漏且类别平衡() -> None:
    """主分类训练、校准、测试与 OOD 必须严格隔离。"""

    root = 项目根目录 / "datasets/confidence_news"
    train = 读(root / "rlcr_train.jsonl")
    sft_train = 读(root / "rlcr_sft.jsonl")
    sft_val = 读(root / "rlcr_sft_val.jsonl")
    calibration = 读(root / "calibration.jsonl")
    test = 读(root / "test.jsonl")
    ood_train = 读(root / "ood_train.jsonl")
    ood_calibration = 读(root / "ood_calibration.jsonl")
    ood_test = 读(root / "ood_test.jsonl")
    assert len(train) == 960 and len(calibration) == len(test) == 160
    assert len(sft_train) == 280 and len(sft_val) == 40
    assert not {row["record_id"] for row in sft_train} & {row["record_id"] for row in sft_val}
    assert all(row["messages"][-1]["role"] == "assistant" for row in sft_val)
    placeholders = {解析置信度(row["messages"][-1]["content"]) for row in sft_train}
    assert len(placeholders) >= 8
    assert len(ood_train) == len(ood_calibration) == len(ood_test) == 100
    ids = [{row["record_id"] for row in rows} for rows in (train, calibration, test, ood_train, ood_calibration, ood_test)]
    assert all(not ids[i] & ids[j] for i in range(len(ids)) for j in range(i + 1, len(ids)))
    for rows in (calibration, test):
        assert {label: sum(row["label"] == label for row in rows) for label in ("政治", "财经", "体育", "计算机")} == {
            "政治": 40, "财经": 40, "体育": 40, "计算机": 40,
        }


def test_验证器训练与测试新闻无交叉() -> None:
    """验证器的一正一负对不得跨拆分复用正文。"""

    root = 项目根目录 / "datasets/confidence_news"
    train = 读(root / "verifier_train.jsonl")
    val = 读(root / "verifier_val.jsonl")
    test = 读(root / "verifier_test.jsonl")
    train_ids = {row["source_record_id"] for row in train}
    val_ids = {row["source_record_id"] for row in val}
    test_ids = {row["source_record_id"] for row in test}
    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids
    assert {row["candidate_correct"] for row in train} == {True, False}


def test_固定数据条数与摘要一致() -> None:
    """提交的数据必须与生成清单中的条数和 SHA-256 完全一致。"""

    root = 项目根目录 / "datasets/confidence_news"
    manifest = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["文件"].items():
        path = root / filename
        assert path.is_file()
        assert len(读(path)) == expected["样本数"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["SHA256"]
