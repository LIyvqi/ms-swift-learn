#!/usr/bin/env python3
"""验证 GPU 采样按阶段切分和统计。"""

from __future__ import annotations

from summarize_gpu_samples import 汇总阶段, 渲染_markdown


def test_stage_summary() -> None:
    samples = [
        {"时间戳": 1.0, "物理显存GiB": 10.0, "GPU利用率": 20.0},
        {"时间戳": 2.0, "物理显存GiB": 30.0, "GPU利用率": 100.0},
        {"时间戳": 3.0, "物理显存GiB": 40.0, "GPU利用率": 80.0},
        {"时间戳": 4.0, "物理显存GiB": 20.0, "GPU利用率": 40.0},
    ]
    results = 汇总阶段(samples, [(1.0, "阶段甲"), (3.0, "阶段乙")])
    assert len(results) == 2
    assert results[0]["峰值物理显存GiB"] == 30.0
    assert results[0]["平均GPU利用率"] == 60.0
    assert results[1]["P95_GPU利用率"] == 80.0
    assert "阶段甲" in 渲染_markdown(results)
