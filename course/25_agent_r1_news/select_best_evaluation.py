"""从统一选择集上的动态评测结果中选择最佳 Agent 检查点。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def 嵌套数值(data: dict[str, Any], *keys: str) -> float:
    """读取缺失时按零处理的嵌套指标。"""

    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return float(current) if isinstance(current, int | float) else 0.0


def 计算选择分数(data: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """三个任务等权；决策任务内部再等权衡量类别、规则和证据。"""

    retrieval = 嵌套数值(data, "summary", "retrieve", "retrieval_f1")
    composition = 嵌套数值(data, "summary", "compose", "composition_f1")
    decision_accuracy = 嵌套数值(data, "summary", "decision", "decision_accuracy")
    decision_rules = 嵌套数值(data, "summary", "decision", "composition_f1")
    evidence = 嵌套数值(data, "summary", "decision", "evidence_coverage")
    decision = (decision_accuracy + decision_rules + evidence) / 3
    score = (retrieval + composition + decision) / 3
    return score, {
        "retrieval_f1": retrieval,
        "composition_f1": composition,
        "decision_accuracy": decision_accuracy,
        "decision_rule_f1": decision_rules,
        "evidence_coverage": evidence,
        "decision_subscore": decision,
    }


def 检查点步数(adapter: str) -> int:
    """从 adapter 路径提取 step；无法识别时放到同分候选末尾。"""

    match = re.search(r"checkpoint-(\d+)(?:/)?$", adapter)
    return int(match.group(1)) if match else 10**12


def 选择最佳结果(results: list[dict[str, Any]]) -> dict[str, Any]:
    """按预注册分数选择；同分时依次偏好协议完整、无效动作少和较早节点。"""

    candidates = []
    for data in results:
        score, metrics = 计算选择分数(data)
        agent = data.get("agent_summary", {})
        completion = 嵌套数值({"agent": agent}, "agent", "completion_rate")
        thinking = 嵌套数值({"agent": agent}, "agent", "thinking_presence_rate")
        invalid = 嵌套数值({"agent": agent}, "agent", "invalid_action_rate")
        step = 检查点步数(str(data.get("adapter", "")))
        candidates.append(
            {
                "result": str(data.get("_result_path", "")),
                "adapter": str(data.get("adapter", "")),
                "step": step,
                "selection_score": score,
                "completion_rate": completion,
                "thinking_presence_rate": thinking,
                "invalid_action_rate": invalid,
                "metrics": metrics,
            }
        )
    if not candidates:
        raise ValueError("至少需要一个评测结果")
    first = results[0]
    first_config = first.get("evaluation_config", {})
    candidates.sort(
        key=lambda item: (
            -item["selection_score"],
            -item["completion_rate"],
            -item["thinking_presence_rate"],
            item["invalid_action_rate"],
            item["step"],
        )
    )
    return {
        "selection_protocol": {
            "dataset": first.get("dataset"),
            "dataset_sha256": first_config.get("dataset_sha256"),
            "sample_offset": first_config.get("sample_offset"),
            "maximum_samples": first_config.get("maximum_samples"),
            "sample_sequence_sha256": first_config.get("sample_sequence_sha256"),
            "formula": "(retrieve_F1 + compose_F1 + mean(decision_accuracy, decision_rule_F1, evidence_coverage)) / 3",
            "tie_breakers": [
                "完成率高",
                "显式思考覆盖率高",
                "无效动作率低",
                "step 较早",
            ],
        },
        "best": candidates[0],
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="选择最佳 Agent-R1 动态评测检查点")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = []
    expected_fingerprint: tuple[Any, ...] | None = None
    for path in args.results:
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("evaluation_config", {})
        fingerprint = (
            data.get("samples"),
            config.get("sample_offset"),
            config.get("dataset_sha256"),
            config.get("knowledge_sha256"),
            config.get("sample_sequence_sha256"),
            config.get("temperature"),
            config.get("max_new_tokens"),
        )
        if expected_fingerprint is None:
            expected_fingerprint = fingerprint
        elif fingerprint != expected_fingerprint:
            raise SystemExit(f"评测协议不一致，不能直接比较：{path}")
        data["_result_path"] = str(path)
        loaded.append(data)

    selected = 选择最佳结果(loaded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
