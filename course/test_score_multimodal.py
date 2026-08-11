#!/usr/bin/env python3
"""验证多模态生成评测器的答案、协议与分组统计。"""

from __future__ import annotations

from score_multimodal import 提取答案, 规范答案, 评测


def test_answer_normalization() -> None:
    assert 规范答案(" 1,500。") == "1500"
    assert 提取答案("<answer> A C </answer>") == "AC"


def test_grouped_metrics() -> None:
    rows = [
        {
            "response": "<answer>A</answer>",
            "final_answer": "A",
            "modality": "text_only",
            "style": "direct",
        },
        {
            "response": "<think>根据图中曲线可以判断，故选 B。</think><answer>B</answer>",
            "final_answer": "B",
            "modality": "image_text",
            "style": "cot",
        },
        {
            "response": "看不到图片。<answer>C</answer>",
            "final_answer": "D",
            "modality": "image_only",
            "style": "direct",
        },
    ]
    summary = 评测(rows)
    assert summary["总体"]["准确率"] == 2 / 3
    assert summary["按风格"]["direct"]["严格格式率"] == 0.5
    assert summary["按风格"]["cot"]["非空思考率"] == 1.0
    assert summary["按模态"]["image_only"]["视觉读取失败率"] == 1.0
