#!/usr/bin/env python3
"""用独立 Reward/Verifier 对 RLCR 分类预测输出正确概率。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


项目根目录 = Path(__file__).resolve().parents[2]
第28课目录 = 项目根目录 / "course/28_rlcr_confidence"
if str(第28课目录) not in sys.path:
    sys.path.insert(0, str(第28课目录))

from confidence_metrics import Platt校准器, 汇总置信指标, 选择阈值
from prepare_data import 标签, 验证器系统提示, 验证器用户消息


@dataclass
class 成对分数:
    """保存两种 verdict 序列的奖励差与资源。"""

    correct_score: float
    incorrect_score: float
    delta: float
    prompt_tokens: int
    elapsed_seconds: float


def 解析参数() -> argparse.Namespace:
    """定义 Verifier 和第 28 课策略轨迹输入。"""

    parser = argparse.ArgumentParser(description="独立置信度 Verifier 真实评测")
    parser.add_argument("--verifier", required=True)
    parser.add_argument(
        "--policy-traces", type=Path,
        default=项目根目录 / "outputs/28_rlcr_confidence/evaluation/brier_rlcr.jsonl",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=项目根目录 / "datasets/confidence_news"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=项目根目录 / "outputs/29_independent_confidence_verifier/evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-regenerate", action="store_true")
    return parser.parse_args()


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 用户消息(row: dict[str, Any]) -> str:
    """取分类数据的用户输入。"""

    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def 构造verdict消息(row: dict[str, Any], candidate: str, verdict: str) -> list[dict[str, str]]:
    """构造 Verifier 的完整待打分序列。"""

    return [
        {"role": "system", "content": 验证器系统提示},
        {"role": "user", "content": 验证器用户消息(用户消息(row), candidate)},
        {"role": "assistant", "content": f"<verdict>{verdict}</verdict>"},
    ]


class 验证器打分器:
    """直接读取 ms-swift RM 的单值 score head。"""

    def __init__(self, checkpoint: str, batch_size: int):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()

    def 序列分数(self, messages: list[list[dict[str, str]]]) -> tuple[list[float], list[int], list[float]]:
        """批量计算序列奖励，返回分数、token 数和均摊延迟。"""

        scores, token_counts, latencies = [], [], []
        for start in range(0, len(messages), self.batch_size):
            chunk = messages[start:start + self.batch_size]
            texts = [
                self.tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=False)
                for item in chunk
            ]
            encoded = self.tokenizer(
                texts, padding=True, truncation=True, max_length=768, return_tensors="pt"
            ).to("cuda:0")
            began = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(**encoded).logits.float().view(-1).cpu().tolist()
            elapsed = (time.perf_counter() - began) / max(1, len(chunk))
            scores.extend(float(value) for value in logits)
            token_counts.extend(int(value) for value in encoded["attention_mask"].sum(dim=1).cpu().tolist())
            latencies.extend([elapsed] * len(chunk))
        return scores, token_counts, latencies

    def 成对打分(
        self, inputs: list[tuple[dict[str, Any], str]],
    ) -> list[成对分数]:
        """对每个候选同时打分 CORRECT 与 INCORRECT 两种声明。"""

        correct_messages = [构造verdict消息(row, candidate, "CORRECT") for row, candidate in inputs]
        incorrect_messages = [构造verdict消息(row, candidate, "INCORRECT") for row, candidate in inputs]
        all_messages = [*correct_messages, *incorrect_messages]
        scores, tokens, latency = self.序列分数(all_messages)
        split = len(inputs)
        return [
            成对分数(
                correct_score=scores[index],
                incorrect_score=scores[index + split],
                delta=scores[index] - scores[index + split],
                prompt_tokens=tokens[index] + tokens[index + split],
                elapsed_seconds=latency[index] + latency[index + split],
            )
            for index in range(split)
        ]


def 加载数据索引(data_dir: Path) -> dict[str, dict[str, Any]]:
    """以 record_id 连接策略轨迹和原始新闻，不使用金标决定打分。"""

    index = {}
    for filename in ("calibration.jsonl", "test.jsonl", "ood_calibration.jsonl", "ood_test.jsonl"):
        for row in 读_jsonl(data_dir / filename):
            index[row["record_id"]] = row
    return index


def 缓存成对分数(
    args: argparse.Namespace,
    scorer: 验证器打分器,
    inputs: list[tuple[dict[str, Any], str]],
    name: str,
) -> list[成对分数]:
    """用模型与完整输入指纹保护奖励分数缓存。"""

    path = args.output_dir / "score_cache" / f"{name}.jsonl"
    meta = path.with_suffix(".meta.json")
    fingerprint = hashlib.sha256(json.dumps({
        "verifier": str(Path(args.verifier).resolve()),
        "inputs": [(row["record_id"], candidate, row["messages"]) for row, candidate in inputs],
    }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if path.exists() and meta.exists() and not args.force_regenerate:
        if json.loads(meta.read_text(encoding="utf-8")).get("sha256") == fingerprint:
            rows = 读_jsonl(path)
            if len(rows) == len(inputs):
                return [成对分数(**row) for row in rows]
    scores = scorer.成对打分(inputs)
    写_jsonl(path, [score.__dict__ for score in scores])
    meta.write_text(json.dumps({"sha256": fingerprint}, indent=2) + "\n", encoding="utf-8")
    return scores


def 评估置信源(
    name: str,
    traces: list[dict[str, Any]],
    confidence_field: str,
) -> dict[str, Any]:
    """在统一校准、ID 测试与 OOD 测试上评分一种置信源。"""

    calibration = [trace for trace in traces if trace["split"] in {"calibration", "ood_calibration"}]
    test = [trace for trace in traces if trace["split"] == "test"]
    ood_test = [trace for trace in traces if trace["split"] == "ood_test"]
    joint_test = [*test, *ood_test]
    threshold = 选择阈值(
        [trace[confidence_field] for trace in calibration],
        [trace["target_correct"] for trace in calibration],
    )
    metrics = 汇总置信指标(
        [trace[confidence_field] for trace in test],
        [trace["target_correct"] for trace in test],
        threshold,
    )
    metrics.update({
        "name": name,
        "ood_false_accept_rate": sum(
            trace[confidence_field] >= threshold for trace in ood_test
        ) / len(ood_test),
        "joint_id_ood": 汇总置信指标(
            [trace[confidence_field] for trace in joint_test],
            [trace["target_correct"] for trace in joint_test],
            threshold,
        ),
    })
    return metrics


def 主程序() -> None:
    """运行策略预测的独立验证和静态候选评测。"""

    args = 解析参数()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = 读_jsonl(args.policy_traces)
    data_index = 加载数据索引(args.data_dir)
    scorer = 验证器打分器(args.verifier, args.batch_size)
    valid_policy = [trace for trace in policy if trace["record_id"] in data_index]
    inputs = [(data_index[trace["record_id"]], trace["predicted_label"]) for trace in valid_policy]
    pair_scores = 缓存成对分数(args, scorer, inputs, "policy_predictions")

    traces = []
    for policy_trace, score in zip(valid_policy, pair_scores):
        raw_verifier = 1 / (1 + math.exp(-max(-30.0, min(30.0, score.delta))))
        traces.append({
            **policy_trace,
            "verifier_correct_score": score.correct_score,
            "verifier_incorrect_score": score.incorrect_score,
            "verifier_delta": score.delta,
            "verifier_raw_confidence": raw_verifier,
            "verifier_prompt_tokens": score.prompt_tokens,
            "verifier_elapsed_seconds": score.elapsed_seconds,
        })
    calibration = [trace for trace in traces if trace["split"] in {"calibration", "ood_calibration"}]
    calibrator = Platt校准器.拟合(
        [trace["verifier_raw_confidence"] for trace in calibration],
        [trace["target_correct"] for trace in calibration],
    )
    for trace in traces:
        trace["verifier_calibrated_confidence"] = calibrator.预测(trace["verifier_raw_confidence"])

    predictors = [
        评估置信源("policy_self_raw", traces, "operational_confidence"),
        评估置信源("policy_self_calibrated", traces, "calibrated_confidence"),
        评估置信源("verifier_raw", traces, "verifier_raw_confidence"),
        评估置信源("verifier_calibrated", traces, "verifier_calibrated_confidence"),
    ]

    static_rows = 读_jsonl(args.data_dir / "verifier_test.jsonl")
    static_inputs = []
    for row in static_rows:
        # 回连不含候选的源新闻，避免把训练行中已有的候选文本重复拼接两次。
        source = data_index[row["source_record_id"]]
        static_inputs.append((source, row["candidate_label"]))
    static_scores = 缓存成对分数(args, scorer, static_inputs, "static_candidates")
    static_correct = [
        (score.delta > 0) == bool(row["candidate_correct"])
        for score, row in zip(static_scores, static_rows)
    ]
    static_summary = {
        "samples": len(static_rows),
        "verdict_accuracy": sum(static_correct) / len(static_correct),
        "id_samples": sum(not row["is_ood"] for row in static_rows),
        "ood_samples": sum(row["is_ood"] for row in static_rows),
        "ood_incorrect_detection_rate": sum(
            score.delta < 0 for score, row in zip(static_scores, static_rows) if row["is_ood"]
        ) / sum(row["is_ood"] for row in static_rows),
    }
    test_traces = [trace for trace in traces if trace["split"] == "test"]
    summary = {
        "policy_classification_accuracy": sum(trace["target_correct"] for trace in test_traces) / len(test_traces),
        "verifier_platt_calibrator": calibrator.转字典(),
        "predictors": predictors,
        "static_candidate_test": static_summary,
        "mean_verifier_calls": 2.0,
        "mean_verifier_prompt_tokens": sum(trace["verifier_prompt_tokens"] for trace in test_traces) / len(test_traces),
        "mean_verifier_latency_seconds": sum(trace["verifier_elapsed_seconds"] for trace in test_traces) / len(test_traces),
    }
    写_jsonl(args.output_dir / "policy_verifier_traces.jsonl", traces)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "verifier": args.verifier,
        "policy_traces": str(args.policy_traces),
        "verifier_and_policy_are_distinct_paths": str(Path(args.verifier).resolve()) != str(args.policy_traces.resolve()),
        "verifier_training_type": "独立全参数成对奖励模型",
        "verifier_uses_policy_reported_confidence": False,
        "verifier_scoring": "sigmoid(score(CORRECT)-score(INCORRECT))",
        "batch_size": args.batch_size,
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
