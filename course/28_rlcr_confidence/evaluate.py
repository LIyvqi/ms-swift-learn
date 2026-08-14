#!/usr/bin/env python3
"""对格式 SFT、正确性 RL、Brier-RLCR 与对数 RLCR 做真实校准评测。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


项目根目录 = Path(__file__).resolve().parents[2]
课程目录 = Path(__file__).resolve().parent
if str(课程目录) not in sys.path:
    sys.path.insert(0, str(课程目录))

from confidence_metrics import Platt校准器, 汇总置信指标, 解析响应, 选择阈值


允许标签 = ("政治", "财经", "体育", "计算机")
基础模型 = 项目根目录 / "models/Qwen3.5-0.8B-Base"
官方Swift目录 = 项目根目录 / "third_party/ms-swift-official-4.4.3"
官方Swift提交 = "e1287928be4451b9ed5e2fb00a24ad3c8f61287b"


@dataclass
class 生成记录:
    """持久化文本与真实资源统计。"""

    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    mean_logprob: float | None


def 解析参数() -> argparse.Namespace:
    """定义四个模型与数据路径。"""

    parser = argparse.ArgumentParser(description="RLCR 四组置信度对照评测")
    parser.add_argument("--format-sft", required=True)
    parser.add_argument("--correctness", required=True)
    parser.add_argument("--brier", required=True)
    parser.add_argument("--log-score", required=True)
    parser.add_argument("--data-dir", type=Path, default=项目根目录 / "datasets/confidence_news")
    parser.add_argument(
        "--output-dir", type=Path,
        default=项目根目录 / "outputs/28_rlcr_confidence/evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=128)
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


def 校验Swift() -> dict[str, str]:
    """强制使用官方 v4.4.3 标签源码。"""

    commit = subprocess.check_output(
        ["git", "-C", str(官方Swift目录), "rev-parse", "HEAD"], text=True
    ).strip()
    tag = subprocess.check_output(
        ["git", "-C", str(官方Swift目录), "describe", "--tags", "--exact-match", "HEAD"], text=True
    ).strip()
    if commit != 官方Swift提交 or tag != "v4.4.3":
        raise RuntimeError(f"ms-swift 版本不符：{tag} {commit}")
    if str(官方Swift目录) not in sys.path:
        sys.path.insert(0, str(官方Swift目录))
    import swift
    return {
        "git_tag": tag,
        "git_commit": commit,
        "package_version_string": swift.__version__,
        "module_path": str(Path(swift.__file__).resolve()),
    }


def 平均对数概率(logprobs: Any) -> float | None:
    """从 OpenAI 兼容结构中取实际生成 token 均值。"""

    if not logprobs:
        return None
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    values = [
        float(item["logprob"]) for item in content or []
        if isinstance(item, dict) and isinstance(item.get("logprob"), (int, float))
    ]
    return sum(values) / len(values) if values else None


def 生成(
    adapter: str, rows: list[dict[str, Any]], batch_size: int,
) -> list[生成记录]:
    """使用本地 TransformersEngine 确定性批量生成。"""

    from swift import InferRequest, RequestConfig, TransformersEngine

    engine = TransformersEngine(
        str(基础模型),
        adapters=[adapter],
        torch_dtype=torch.bfloat16,
        attn_impl="eager",
        device_map="cuda:0",
        max_batch_size=batch_size,
    )
    records = []
    config = RequestConfig(
        max_tokens=64, temperature=0.0, logprobs=True, top_logprobs=5, return_details=True,
    )
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        requests = [InferRequest(messages=row["messages"]) for row in chunk]
        began = time.perf_counter()
        responses = engine.infer(requests, request_config=config, use_tqdm=False)
        elapsed = (time.perf_counter() - began) / max(1, len(chunk))
        for response in responses:
            choice = response.choices[0]
            records.append(生成记录(
                text=str(choice.message.content or ""),
                prompt_tokens=int(response.usage.prompt_tokens),
                completion_tokens=int(response.usage.completion_tokens),
                elapsed_seconds=elapsed,
                mean_logprob=平均对数概率(choice.logprobs),
            ))
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return records


def 宏平均F1(gold: list[str], predicted: list[str]) -> float:
    """计算四类宏平均 F1。"""

    scores = []
    for label in 允许标签:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def 评分方法(name: str, adapter: str, all_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """生成或复用缓存，拟合校准器并评分最终测试。"""

    cache_path = args.output_dir / "generation_cache" / f"{name}.jsonl"
    meta_path = cache_path.with_suffix(".meta.json")
    fingerprint = hashlib.sha256(json.dumps({
        "adapter": str(Path(adapter).resolve()),
        "base_model": str(基础模型.resolve()),
        "records": [row["record_id"] for row in all_rows],
        "messages": [row["messages"] for row in all_rows],
        "batch_size": args.batch_size,
        "max_tokens": 64,
    }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cached = None
    if cache_path.exists() and meta_path.exists() and not args.force_regenerate:
        if json.loads(meta_path.read_text(encoding="utf-8")).get("sha256") == fingerprint:
            rows = 读_jsonl(cache_path)
            if len(rows) == len(all_rows):
                cached = [生成记录(**row) for row in rows]
    records = cached if cached is not None else 生成(adapter, all_rows, args.batch_size)
    if cached is None:
        写_jsonl(cache_path, [asdict(record) for record in records])
        meta_path.write_text(json.dumps({"sha256": fingerprint}, indent=2) + "\n", encoding="utf-8")

    traces = []
    for row, record in zip(all_rows, records):
        parsed = 解析响应(record.text)
        is_ood = bool(row["is_ood"])
        target = int(not is_ood and parsed["predicted_label"] == row["label"])
        # 没有数值置信度的模型等价于未表达任何怀疑，运营口径记为 1。
        raw_confidence = parsed["reported_confidence"]
        operational_confidence = 1.0 if raw_confidence is None else raw_confidence
        traces.append({
            "method": name,
            "adapter": adapter,
            "record_id": row["record_id"],
            "split": row["evaluation_split"],
            "gold_label": row["label"],
            "is_ood": is_ood,
            "response": record.text,
            **parsed,
            "target_correct": target,
            "operational_confidence": operational_confidence,
            "resource": {
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "elapsed_seconds": record.elapsed_seconds,
                "mean_logprob": record.mean_logprob,
            },
        })

    # 自报置信度策略没有见过 OOD，因此只用 ID 校准；OOD 能力作为独立压力测试报告。
    calibration = [trace for trace in traces if trace["split"] == "calibration"]
    platt = Platt校准器.拟合(
        [trace["operational_confidence"] for trace in calibration],
        [trace["target_correct"] for trace in calibration],
    )
    for trace in traces:
        trace["calibrated_confidence"] = platt.预测(trace["operational_confidence"])
    threshold = 选择阈值(
        [trace["calibrated_confidence"] for trace in calibration],
        [trace["target_correct"] for trace in calibration],
    )
    test = [trace for trace in traces if trace["split"] == "test"]
    ood_test = [trace for trace in traces if trace["split"] == "ood_test"]
    gold = [trace["gold_label"] for trace in test]
    predicted = [trace["predicted_label"] for trace in test]
    raw = 汇总置信指标(
        [trace["operational_confidence"] for trace in test],
        [trace["target_correct"] for trace in test],
    )
    calibrated = 汇总置信指标(
        [trace["calibrated_confidence"] for trace in test],
        [trace["target_correct"] for trace in test], threshold,
    )
    summary = {
        "method": name,
        "adapter": adapter,
        "test_samples": len(test),
        "ood_test_samples": len(ood_test),
        "classification_accuracy": sum(trace["target_correct"] for trace in test) / len(test),
        "classification_macro_f1": 宏平均F1(gold, predicted),
        "strict_format_rate": sum(trace["format_valid"] for trace in test) / len(test),
        "numeric_confidence_rate": sum(
            trace["reported_confidence"] is not None for trace in test
        ) / len(test),
        "raw_confidence": raw,
        "platt_calibrator": platt.转字典(),
        "calibrated_confidence": calibrated,
        "ood_false_accept_rate": sum(
            trace["calibrated_confidence"] >= threshold for trace in ood_test
        ) / len(ood_test),
        "mean_prompt_tokens": sum(trace["resource"]["prompt_tokens"] for trace in test) / len(test),
        "mean_completion_tokens": sum(trace["resource"]["completion_tokens"] for trace in test) / len(test),
        "mean_latency_seconds": sum(trace["resource"]["elapsed_seconds"] for trace in test) / len(test),
        "prediction_counts": dict(Counter(predicted)),
    }
    写_jsonl(args.output_dir / f"{name}.jsonl", traces)
    return summary


def 主程序() -> None:
    """运行全部对照并保存配置。"""

    args = 解析参数()
    version = 校验Swift()
    split_files = [
        ("calibration", "calibration.jsonl"),
        ("test", "test.jsonl"),
        ("ood_calibration", "ood_calibration.jsonl"),
        ("ood_test", "ood_test.jsonl"),
    ]
    all_rows = []
    for split, filename in split_files:
        for row in 读_jsonl(args.data_dir / filename):
            all_rows.append({**row, "evaluation_split": split})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = {
        "format_sft": args.format_sft,
        "correctness_rl": args.correctness,
        "brier_rlcr": args.brier,
        "log_rlcr": args.log_score,
    }
    comparison = [评分方法(name, adapter, all_rows, args) for name, adapter in methods.items()]
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "ms_swift": version,
        "batch_size": args.batch_size,
        "methods": methods,
        "base_model": str(基础模型),
        "calibration_samples": sum(row["evaluation_split"] == "calibration" for row in all_rows),
        "test_samples": sum(row["evaluation_split"] == "test" for row in all_rows),
        "ood_calibration_samples": sum(row["evaluation_split"] == "ood_calibration" for row in all_rows),
        "ood_test_samples": sum(row["evaluation_split"] == "ood_test" for row in all_rows),
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for summary in comparison:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    主程序()
