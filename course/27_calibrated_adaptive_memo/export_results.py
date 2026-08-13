#!/usr/bin/env python3
"""把大型 CA-MeMo 输出提炼为可提交的小型真实结果摘要。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[2]
课程目录 = Path(__file__).resolve().parent


def 解析参数() -> argparse.Namespace:
    """定义输入和提交结果目录。"""

    parser = argparse.ArgumentParser(description="导出 CA-MeMo 可提交结果")
    parser.add_argument(
        "--input-dir", type=Path,
        default=项目根目录 / "outputs/27_calibrated_adaptive_memo/full_experiment",
    )
    parser.add_argument("--output-dir", type=Path, default=课程目录 / "results")
    return parser.parse_args()


def 读_json(path: Path) -> Any:
    """读取 JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 摘要(path: Path) -> str:
    """为原始结果计算 SHA256，建立提交摘要到本地证据的映射。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def 主程序() -> None:
    """导出对照、实验配置和代表性失败。"""

    args = 解析参数()
    comparison_path = args.input_dir / "comparison.json"
    config_path = args.input_dir / "experiment_config.json"
    traces_path = args.input_dir / "calibrated_search_verifier.jsonl"
    for path in (comparison_path, config_path, traces_path):
        if not path.exists():
            raise FileNotFoundError(f"完整实验证据不存在：{path}")

    comparison = 读_json(comparison_path)
    config = 读_json(config_path)
    traces = 读_jsonl(traces_path)
    compact_config = {
        "memory_model_name": Path(config["memory_model"]).name,
        "memory_backend": config["memory_backend"],
        "ms_swift": config.get("ms_swift"),
        "calibration_samples": config["calibration_samples"],
        "test_samples": config["test_samples"],
        "calibration_positive_rate": config["calibration_positive_rate"],
        "route_thresholds": config["route_thresholds"],
        "conformal_alpha": config["conformal_alpha"],
        "conformal_threshold": config["conformal_threshold"],
        "authority_bm25_threshold": config["authority_bm25_threshold"],
        "medium_search_cases_all_splits": config["medium_search_cases_all_splits"],
        "request_logprobs": config["request_logprobs"],
        "batch_size": config["batch_size"],
        "api_concurrency": config.get("api_concurrency"),
        "judge_backend": config.get("judge_backend", "none"),
        "input_price_per_million": config.get("input_price_per_million", 0.0),
        "output_price_per_million": config.get("output_price_per_million", 0.0),
    }
    failures = []
    for trace in traces:
        rule_exact = set(trace["prediction"]["matched_rules"]) == set(trace["gold_rule_ids"])
        decision_correct = trace["prediction"]["decision"] == trace["gold_decision"]
        required_reject = trace.get("requires_abstention", False) and trace["accepted"]
        if rule_exact and decision_correct and not required_reject:
            continue
        verifier = trace.get("verifier_trace") or {}
        failures.append({
            "case_id": trace["case_id"],
            "scenario_type": trace["scenario_type"],
            "is_ood": trace["is_ood"],
            "gold_decision": trace["gold_decision"],
            "predicted_decision": trace["prediction"]["decision"],
            "gold_rule_ids": trace["gold_rule_ids"],
            "predicted_rule_ids": trace["prediction"]["matched_rules"],
            "accepted": trace["accepted"],
            "confidence": trace["confidence"],
            "route": trace["route"],
            "authority_rule_ids": verifier.get("authority_rule_ids", []),
        })

    evidence = {
        "comparison_sha256": 摘要(comparison_path),
        "config_sha256": 摘要(config_path),
        "verifier_traces_sha256": 摘要(traces_path),
        "source_output_dir": "outputs/27_calibrated_adaptive_memo/full_experiment",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "comparison.json": comparison,
        "experiment_config.json": compact_config,
        "failure_cases.json": failures,
        "evidence.json": evidence,
    }
    for name, payload in artifacts.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({name: len(payload) if isinstance(payload, list) else 1 for name, payload in artifacts.items()}, ensure_ascii=False))


if __name__ == "__main__":
    主程序()
