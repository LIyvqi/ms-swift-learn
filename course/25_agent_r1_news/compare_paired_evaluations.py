"""对同一批新闻上的 SFT 与 GRPO 动态轨迹做配对统计比较。"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

指标显示名 = {
    "retrieval_f1": "检索 F1",
    "composition_f1": "组合 F1",
    "decision_accuracy": "决策准确率",
    "decision_rule_f1": "决策规则 F1",
    "evidence_coverage": "证据覆盖率",
    "pipeline_score": "三任务总分",
}


def 轨迹指标(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    """把三条任务轨迹合并成以新闻 record_id 为单位的指标。"""

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for trace in data.get("traces", []):
        record_id = str(trace.get("record_id", ""))
        task = str(trace.get("task", ""))
        if not record_id or task not in {"retrieve", "compose", "decision"}:
            raise ValueError("轨迹缺少合法的 record_id 或 task")
        tasks = grouped.setdefault(record_id, {})
        if task in tasks:
            raise ValueError(f"同一新闻出现重复任务轨迹：{record_id}/{task}")
        tasks[task] = trace

    result: dict[str, dict[str, float]] = {}
    for record_id, tasks in grouped.items():
        missing = {"retrieve", "compose", "decision"} - set(tasks)
        if missing:
            raise ValueError(f"新闻 {record_id} 缺少任务：{sorted(missing)}")
        retrieve = tasks["retrieve"].get("metrics", {})
        compose = tasks["compose"].get("metrics", {})
        decision = tasks["decision"].get("metrics", {})
        values = {
            "retrieval_f1": float(retrieve.get("retrieval_f1", 0.0)),
            "composition_f1": float(compose.get("composition_f1", 0.0)),
            "decision_accuracy": float(decision.get("decision_accuracy", 0.0)),
            "decision_rule_f1": float(decision.get("composition_f1", 0.0)),
            "evidence_coverage": float(decision.get("evidence_coverage", 0.0)),
        }
        decision_subscore = mean(
            values[key]
            for key in (
                "decision_accuracy",
                "decision_rule_f1",
                "evidence_coverage",
            )
        )
        values["pipeline_score"] = mean(
            [values["retrieval_f1"], values["composition_f1"], decision_subscore]
        )
        result[record_id] = values
    return result


def 分位数(sorted_values: list[float], probability: float) -> float:
    """用线性插值计算分位数。"""

    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def 精确_mcnemar(
    baseline: list[float], candidate: list[float]
) -> dict[str, float | int]:
    """对决策正确性的配对变化计算双侧精确 McNemar 检验。"""

    regressions = sum(
        left >= 0.5 and right < 0.5 for left, right in zip(baseline, candidate)
    )
    improvements = sum(
        left < 0.5 and right >= 0.5 for left, right in zip(baseline, candidate)
    )
    discordant = regressions + improvements
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(regressions, improvements)
        probability = sum(
            math.comb(discordant, index) for index in range(lower + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * probability)
    return {
        "improvements": improvements,
        "regressions": regressions,
        "discordant_pairs": discordant,
        "two_sided_exact_p": p_value,
    }


def 配对比较(
    baseline_data: dict[str, Any],
    candidate_data: dict[str, Any],
    bootstrap_samples: int = 10_000,
    seed: int = 20_260_811,
) -> dict[str, Any]:
    """按新闻配对重采样，计算候选模型减基线模型的差值区间。"""

    baseline = 轨迹指标(baseline_data)
    candidate = 轨迹指标(candidate_data)
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))[:10]
        missing_baseline = sorted(set(candidate) - set(baseline))[:10]
        raise ValueError(
            "两份评测的 record_id 不一致；"
            f"候选缺少 {missing_candidate}，基线缺少 {missing_baseline}"
        )
    record_ids = sorted(baseline)
    if not record_ids:
        raise ValueError("没有可比较的新闻轨迹")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples 必须大于 0")

    metric_names = list(指标显示名)
    baseline_values = {
        name: [baseline[record_id][name] for record_id in record_ids]
        for name in metric_names
    }
    candidate_values = {
        name: [candidate[record_id][name] for record_id in record_ids]
        for name in metric_names
    }
    differences = {
        name: [
            right - left
            for left, right in zip(baseline_values[name], candidate_values[name])
        ]
        for name in metric_names
    }

    rng = random.Random(seed)
    bootstrap = {name: [] for name in metric_names}
    sample_size = len(record_ids)
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(sample_size) for _ in range(sample_size)]
        for name in metric_names:
            bootstrap[name].append(mean(differences[name][index] for index in indices))

    metrics = {}
    for name in metric_names:
        sampled = sorted(bootstrap[name])
        metrics[name] = {
            "display_name": 指标显示名[name],
            "baseline": mean(baseline_values[name]),
            "candidate": mean(candidate_values[name]),
            "difference": mean(differences[name]),
            "ci95_low": 分位数(sampled, 0.025),
            "ci95_high": 分位数(sampled, 0.975),
        }

    return {
        "baseline_adapter": baseline_data.get("adapter"),
        "candidate_adapter": candidate_data.get("adapter"),
        "paired_news": len(record_ids),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "difference_direction": "candidate_minus_baseline",
        "metrics": metrics,
        "decision_mcnemar": 精确_mcnemar(
            baseline_values["decision_accuracy"],
            candidate_values["decision_accuracy"],
        ),
        "limitations": "区间描述当前留出新闻集合的配对抽样波动，不替代独立外部测试集。",
    }


def 渲染_markdown(report: dict[str, Any]) -> str:
    """把配对统计结果渲染成中文 Markdown。"""

    lines = [
        "# SFT 与最佳 GRPO 的留出集配对比较",
        "",
        f"配对新闻数：{report['paired_news']}；bootstrap 次数：{report['bootstrap_samples']}；差值方向：候选 GRPO − SFT。",
        "",
        "| 指标 | SFT | 最佳 GRPO | 配对差值 | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in report["metrics"].values():
        lines.append(
            f"| {metric['display_name']} | {metric['baseline']:.4f} | "
            f"{metric['candidate']:.4f} | {metric['difference']:+.4f} | "
            f"[{metric['ci95_low']:+.4f}, {metric['ci95_high']:+.4f}] |"
        )
    mcnemar = report["decision_mcnemar"]
    lines.extend(
        [
            "",
            (
                "决策正确性配对变化："
                f"改善 {mcnemar['improvements']} 篇，退化 {mcnemar['regressions']} 篇，"
                f"双侧精确 McNemar `p={mcnemar['two_sided_exact_p']:.6g}`。"
            ),
            "",
            f"> 限制：{report['limitations']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对两份 Agent 动态评测做新闻级配对统计"
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = 配对比较(baseline, candidate, args.bootstrap_samples, args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rendered = 渲染_markdown(report)
    args.output_md.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
