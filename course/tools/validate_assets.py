#!/usr/bin/env python3
"""快速检查模型、数据集和课程前置资产的结构与完整性。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "gsm8k_1k"
MODEL = ROOT / "models" / "Qwen3.5-0.8B-Base"


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
print("ASSET_CHECK=PASS")
