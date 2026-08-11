#!/usr/bin/env python3
"""对多模态 GRPO 奖励执行不占 GPU 的边界测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

插件路径 = Path(__file__).resolve().parents[1] / "plugins/multimodal_rewards.py"
模块规格 = importlib.util.spec_from_file_location("课程多模态奖励", 插件路径)
模块 = importlib.util.module_from_spec(模块规格)
assert 模块规格 and 模块规格.loader
模块规格.loader.exec_module(模块)


def 主程序() -> None:
    """覆盖答案、格式、视觉落地和过程一致性。"""

    accuracy = 模块.多模态答案正确奖励()
    assert accuracy(
        ["<answer>AC</answer>", "<answer>1,000</answer>", "<answer>B</answer>"],
        ["AC", "1000", "D"],
    ) == [1.0, 1.0, 0.0]

    direct = 模块.多模态直接格式奖励()
    assert direct(
        [
            "<answer>A</answer>",
            "<think>\n\n</think>\n\n<answer>A</answer>",
            "<think>因为图中箭头向右。</think><answer>A</answer>",
            "说明如下<answer>A</answer>",
        ]
    ) == [1.0, 1.0, 0.0, 0.0]

    cot = 模块.多模态思考结构奖励()
    assert cot(
        [
            "<think>根据图中箭头方向可知应选择 D。</think><answer>D</answer>",
            "<think></think><answer>D</answer>",
        ]
    ) == [1.0, 0.0]

    grounding = 模块.多模态视觉落地奖励()
    assert grounding(
        [
            "<think>根据图中曲线可知答案为 A。</think><answer>A</answer>",
            "<think>无法查看图片，所以猜 A。</think><answer>A</answer>",
            "<think>计算 3×4=12。</think><answer>12</answer>",
        ],
        ["image_text", "image_only", "text_only"],
    ) == [1.0, 0.0, 1.0]

    consistency = 模块.多模态过程答案一致奖励()
    assert consistency(
        [
            "<think>比较四个选项，故选 C。</think><answer>C</answer>",
            "<think>计算得到 42。</think><answer>42</answer>",
            "<think>故选 A。</think><answer>B</answer>",
        ]
    ) == [1.0, 1.0, 0.0]
    print("多模态奖励边界测试通过。")


if __name__ == "__main__":
    主程序()
