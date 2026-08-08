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
print("ASSET_CHECK=PASS")
