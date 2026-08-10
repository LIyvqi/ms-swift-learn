#!/usr/bin/env python3
"""用本地 Qwen3.5 处理器验证三种模态能被 ms-swift 模板真实编码。"""

from __future__ import annotations

import json
from pathlib import Path

from swift.model import get_processor
from swift.template import get_template

项目根目录 = Path(__file__).resolve().parents[1]
模型目录 = 项目根目录 / "models/Qwen3.5-0.8B-Base"
冒烟数据 = 项目根目录 / "datasets/multimodal_200/cot_smoke.jsonl"


def 主程序() -> None:
    """只加载 tokenizer 与视觉处理器，不加载模型权重，也不占用 GPU。"""

    processor = get_processor(str(模型目录))
    template = get_template(processor, max_length=2048)
    template.set_mode("train")
    rows = [json.loads(line) for line in 冒烟数据.open(encoding="utf-8")]
    seen = set()
    for row in rows:
        encoded = template.encode(row)
        input_ids = encoded["input_ids"]
        length = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[-1]
        if row["modality"] == "text_only":
            if "pixel_values" in encoded:
                raise ValueError("纯文本样本不应产生 pixel_values")
        else:
            for field in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
                if field not in encoded:
                    raise ValueError(f"视觉样本缺少编码字段：{field}")
        if length > 2048:
            raise ValueError(f"样本 {row['id']} 编码后超过 max_length")
        seen.add(row["modality"])
        print(f"{row['modality']}：{length} token，编码字段={sorted(encoded)}")
    if seen != {"text_only", "image_only", "image_text"}:
        raise ValueError(f"冒烟数据没有覆盖全部模态：{seen}")
    print("ms-swift 多模态模板编码校验通过。")


if __name__ == "__main__":
    主程序()
