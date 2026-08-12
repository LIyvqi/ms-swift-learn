#!/usr/bin/env python3
"""把被 Git 忽略的大型输出提炼为可提交的实验摘要。"""

from __future__ import annotations

import json
from pathlib import Path


项目根目录 = Path(__file__).resolve().parents[2]
输出根目录 = 项目根目录 / "outputs/26_memo_rule_memory"
结果目录 = 项目根目录 / "course/26_memo_rule_memory/results"


def 读取(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def 相对模型(value: str | None) -> str | None:
    """移除机器绝对路径，使摘要能在任意克隆目录阅读。"""

    if value is None:
        return None
    marker = "/ms-swift-learn/"
    return value.split(marker, 1)[-1] if marker in value else value


def 主程序() -> None:
    结果目录.mkdir(parents=True, exist_ok=True)
    memory_candidates = list((输出根目录 / "memory_evaluation").rglob("comparison.json"))
    if not memory_candidates:
        raise FileNotFoundError("没有找到 Memory checkpoint 对比结果")
    memory = 读取(max(memory_candidates, key=lambda path: path.stat().st_mtime))
    for row in memory:
        row["model"] = 相对模型(row.get("model"))
        row["adapter"] = 相对模型(row.get("adapter"))

    fragmented = {row["method"]: row for row in 读取(输出根目录 / "audit_evaluation_fragmented/comparison.json")}
    # 当前一键入口把七组方法都写入 fragmented；早期实测只在该目录重跑两种
    # 结构化方法，因此兼容从旧 audit_evaluation 读取其余五组真实基线。
    required_baselines = {"no_memory", "all_rules", "bm25", "memo_single", "oracle_rules"}
    if required_baselines <= set(fragmented):
        baseline = fragmented
    else:
        baseline = {row["method"]: row for row in 读取(输出根目录 / "audit_evaluation/comparison.json")}
    selected = [
        baseline["no_memory"], baseline["all_rules"], baseline["bm25"], baseline["memo_single"],
        fragmented["memo_structured"], fragmented["memo_structured_deterministic"], baseline["oracle_rules"],
    ]
    ablations = {
        "完整新闻直接输入Memory": 读取(输出根目录 / "audit_ablation_full_content/comparison.json")[0],
        "先Grounding并拆分多线索": fragmented["memo_structured_deterministic"],
    }
    scaling = 读取(输出根目录 / "scaling_probe.json")
    scaling["model"] = 相对模型(scaling.get("model"))

    payloads = {
        "memory_checkpoints.json": memory,
        "audit_comparison.json": selected,
        "ablations.json": ablations,
        "scaling_probe.json": scaling,
    }
    for name, payload in payloads.items():
        (结果目录 / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    print(json.dumps({name: len(payload) for name, payload in payloads.items()}, ensure_ascii=False))


if __name__ == "__main__":
    主程序()
