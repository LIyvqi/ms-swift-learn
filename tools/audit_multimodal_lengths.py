#!/usr/bin/env python3
"""用真实 Qwen3.5 视觉模板审计全部 200 个源样本的 token 长度。"""

from __future__ import annotations

import json
from pathlib import Path

from swift.model import get_processor
from swift.template import get_template

项目根目录 = Path(__file__).resolve().parents[1]
模型目录 = 项目根目录 / "models/Qwen3.5-0.8B-Base"
数据目录 = 项目根目录 / "datasets/multimodal_200"


def 读取记录(name: str) -> list[dict]:
    """合并指定视图的训练集和验证集，并把图片路径转成绝对路径。"""

    rows = []
    for split in ("train", "val"):
        path = 数据目录 / f"{name}_{split}.jsonl"
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("images"):
                row["images"] = [str(项目根目录 / item) for item in row["images"]]
            rows.append(row)
    return rows


def 分位数(values: list[int], ratio: float) -> int:
    """返回离散最近秩分位数。"""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio + 0.999999) - 1))
    return ordered[index]


def 审计视图(processor, name: str, mode: str, limit: int) -> dict:
    """在不按课程阈值截断的模板中编码，再检查真实长度上限。"""

    template = get_template(processor, max_length=8192)
    template.set_mode(mode)
    lengths = []
    violations = []
    for row in 读取记录(name):
        encoded = template.encode(row)
        input_ids = encoded["input_ids"]
        length = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[-1]
        lengths.append(length)
        if length > limit:
            violations.append(
                {"id": row["id"], "modality": row["modality"], "tokens": length}
            )
    return {
        "视图": name,
        "样本数": len(lengths),
        "阈值": limit,
        "最小": min(lengths),
        "中位数": 分位数(lengths, 0.5),
        "P95": 分位数(lengths, 0.95),
        "最大": max(lengths),
        "超限数": len(violations),
        "超限样本": violations,
    }


def 主程序() -> None:
    """CoT 是最长 SFT/Prompt 视图；它通过即可覆盖更短的 Direct 与 mixed 视图。"""

    processor = get_processor(str(模型目录))
    results = [
        审计视图(processor, "cot", "train", 2048),
        # GRPO 的 Prompt-only JSONL 仍由训练数据模板编码；infer 模式只接受 InferRequest。
        审计视图(processor, "prompts_cot", "train", 1536),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(result["超限数"] for result in results):
        raise RuntimeError("多模态数据存在超过课程 max_length 的样本")
    print("全部 200 个源样本的最长监督视图和最长提示视图均未超限。")


if __name__ == "__main__":
    主程序()
