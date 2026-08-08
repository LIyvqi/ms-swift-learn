#!/usr/bin/env python3
"""快速检查模型、数据集和课程前置资产的结构与完整性。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "gsm8k_1k"
MODEL = ROOT / "models" / "Qwen3.5-0.8B-Base"
CLASSIFICATION_DATA = ROOT / "datasets" / "fudan_news_4class"
COT_CLASSIFICATION_DATA = ROOT / "datasets" / "fudan_news_cot_50"
ALIGNMENT_DATA = ROOT / "datasets" / "alignment_news"
REAL_JUDGE_DATA = ROOT / "datasets" / "real_judge_1to5"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


assert MODEL.joinpath("model.safetensors-00001-of-00001.safetensors").stat().st_size > 1_700_000_000
assert not ROOT.joinpath("models/Qwen2.5-0.5B-Instruct").exists()

checksums = json.loads(DATA.joinpath("checksums.json").read_text(encoding="utf-8"))
for name, expected in checksums.items():
    actual = hashlib.sha256(DATA.joinpath(name).read_bytes()).hexdigest()
    assert actual == expected, name

for view in ("cot", "direct", "mixed", "prompts_cot", "prompts_direct", "prompts_multi"):
    assert len(read_rows(DATA / f"{view}_train.jsonl")) == 900
    assert len(read_rows(DATA / f"{view}_val.jsonl")) == 100
    assert len(read_rows(DATA / f"{view}_smoke.jsonl")) == 16

cot = read_rows(DATA / "cot_train.jsonl")[0]
direct = read_rows(DATA / "direct_train.jsonl")[0]
assert "<think>" in cot["messages"][-1]["content"]
assert direct["messages"][-1]["content"].startswith("\\boxed{")
assert {row["teacher_tag"] for row in read_rows(DATA / "prompts_multi_train.jsonl")} == {"cot", "direct"}

classification_manifest = json.loads(
    CLASSIFICATION_DATA.joinpath("checksums.json").read_text(encoding="utf-8")
)
classification_counts = {
    "sft_train.jsonl": 320,
    "rl_train.jsonl": 960,
    "rl_smoke.jsonl": 16,
    "val.jsonl": 320,
}
classification_rows = {}
classification_labels = ("政治", "财经", "体育", "计算机")
for name, expected_count in classification_counts.items():
    path = CLASSIFICATION_DATA / name
    rows = read_rows(path)
    classification_rows[name] = rows
    assert len(rows) == expected_count, name
    assert Counter(row["label"] for row in rows) == Counter(
        {label: expected_count // 4 for label in classification_labels}
    ), name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == classification_manifest["文件"][name]["SHA-256"], name

for name in ("sft_train.jsonl", "val.jsonl"):
    assert all(rows["messages"][-1]["role"] == "assistant" for rows in classification_rows[name])
for name in ("rl_train.jsonl", "rl_smoke.jsonl"):
    assert all(
        all(message["role"] != "assistant" for message in row["messages"])
        for row in classification_rows[name]
    )

classification_texts = {
    name: {row["messages"][1]["content"] for row in rows}
    for name, rows in classification_rows.items()
}
assert classification_texts["sft_train.jsonl"].isdisjoint(classification_texts["rl_train.jsonl"])
assert classification_texts["sft_train.jsonl"].isdisjoint(classification_texts["val.jsonl"])
assert classification_texts["rl_train.jsonl"].isdisjoint(classification_texts["val.jsonl"])
assert classification_texts["rl_smoke.jsonl"] <= classification_texts["rl_train.jsonl"]

cot_manifest = json.loads(
    COT_CLASSIFICATION_DATA.joinpath("checksums.json").read_text(encoding="utf-8")
)
cot_counts = {
    "sft_train.jsonl": 40,
    "rl_train.jsonl": 40,
    "rl_smoke.jsonl": 8,
    "evidence_val.jsonl": 10,
    "cot_val_320.jsonl": 320,
}
cot_rows = {}
for name, expected_count in cot_counts.items():
    path = COT_CLASSIFICATION_DATA / name
    rows = read_rows(path)
    cot_rows[name] = rows
    assert len(rows) == expected_count, name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == cot_manifest["文件"][name]["SHA-256"], name

# SFT 与留出集必须有参考答案，RLOO 数据必须只含提示。
for name in ("sft_train.jsonl", "evidence_val.jsonl", "cot_val_320.jsonl"):
    assert all(row["messages"][-1]["role"] == "assistant" for row in cot_rows[name]), name
for name in ("rl_train.jsonl", "rl_smoke.jsonl"):
    assert all(
        all(message["role"] != "assistant" for message in row["messages"])
        for row in cot_rows[name]
    ), name

cot_train_ids = {row["source_record_id"] for row in cot_rows["rl_train.jsonl"]}
cot_heldout_ids = {row["source_record_id"] for row in cot_rows["evidence_val.jsonl"]}
assert cot_train_ids.isdisjoint(cot_heldout_ids)
assert {row["source_record_id"] for row in cot_rows["rl_smoke.jsonl"]} <= cot_train_ids
assert Counter(row["label"] for row in cot_rows["rl_train.jsonl"]) == Counter(
    {label: 10 for label in classification_labels}
)
assert all(len(row["evidence_terms"].split("|||")) == 3 for row in cot_rows["rl_train.jsonl"])

# 统一对齐数据必须同时满足文件校验、视图条数与算法字段约束。
alignment_manifest = json.loads(ALIGNMENT_DATA.joinpath("checksums.json").read_text(encoding="utf-8"))
alignment_counts = {
    **{f"{view}_train.jsonl": 256 for view in ("sft", "pairwise", "prompts", "opsd")},
    **{f"{view}_val.jsonl": 64 for view in ("sft", "pairwise", "prompts", "opsd")},
    **{f"{view}_smoke.jsonl": 16 for view in ("sft", "pairwise", "prompts", "opsd")},
    "rm_train.jsonl": 512,
    "rm_val.jsonl": 128,
    "rm_smoke.jsonl": 32,
    "kto_train.jsonl": 512,
    "kto_val.jsonl": 128,
    "kto_smoke.jsonl": 32,
}
alignment_rows = {}
for name, expected_count in alignment_counts.items():
    path = ALIGNMENT_DATA / name
    rows = read_rows(path)
    alignment_rows[name] = rows
    assert len(rows) == expected_count, name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == alignment_manifest["文件"][name]["SHA-256"], name

for split in ("train", "val", "smoke"):
    pairwise = alignment_rows[f"pairwise_{split}.jsonl"]
    assert all(row["messages"][-1]["content"] != row["rejected_response"] for row in pairwise)
    rm_rows = alignment_rows[f"rm_{split}.jsonl"]
    assert Counter(row["negative_type"] for row in rm_rows) == Counter(
        {"错误类别": len(rm_rows) // 2, "缺少右花括号": len(rm_rows) // 2}
    )
    prompts = alignment_rows[f"prompts_{split}.jsonl"]
    assert all(all(message["role"] != "assistant" for message in row["messages"]) for row in prompts)
    opsd = alignment_rows[f"opsd_{split}.jsonl"]
    assert all("teacher_prompt" in row for row in opsd)
    kto = alignment_rows[f"kto_{split}.jsonl"]
    assert Counter(row["label"] for row in kto) == Counter({True: len(kto) // 2, False: len(kto) // 2})
    # KTO 的循环错位 KL 对照不能与原回答相同，短分类答案因此带唯一记录编号。
    kto_answers = [row["messages"][-1]["content"] for row in kto]
    assert len(kto_answers) == len(set(kto_answers))

# REAL 回归标签必须覆盖有序的 1～5 分，且在线视图不能带 assistant。
real_manifest = json.loads(REAL_JUDGE_DATA.joinpath("checksums.json").read_text(encoding="utf-8"))
real_split_sizes = {"train": 90, "val": 30, "smoke": 10}
for split, expected_count in real_split_sizes.items():
    for view in ("sft", "prompts"):
        name = f"{view}_{split}.jsonl"
        path = REAL_JUDGE_DATA / name
        rows = read_rows(path)
        assert len(rows) == expected_count, name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == real_manifest["文件"][name]["SHA-256"], name
        assert Counter(row["score"] for row in rows) == Counter({score: expected_count // 5 for score in range(1, 6)})
        if view == "sft":
            assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
        else:
            assert all(all(message["role"] != "assistant" for message in row["messages"]) for row in rows)
print("ASSET_CHECK=PASS")
