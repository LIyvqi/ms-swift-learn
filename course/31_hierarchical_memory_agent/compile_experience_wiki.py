#!/usr/bin/env python3
"""把 Agent 轨迹编译为持久经验 Wiki，不生成或安装任何 Skill。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
默认数据目录 = 项目根目录 / "datasets/hierarchical_memory_audit"
默认输出目录 = 课程目录 / "experience_wiki"
动作模式 = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)


def 解析参数() -> argparse.Namespace:
    """真实评测可以重复传入，Wiki 会汇总而不读取模型检查点。"""

    parser = argparse.ArgumentParser(description="编译多源记忆 Agent 的持久经验 Wiki")
    parser.add_argument(
        "--sft-data", type=Path, default=默认数据目录 / "sft_train.jsonl"
    )
    parser.add_argument(
        "--registry", type=Path, default=默认数据目录 / "source_registry.json"
    )
    parser.add_argument(
        "--offline-metrics",
        type=Path,
        default=项目根目录 / "outputs/31_hierarchical_memory_agent/offline_metrics.json",
    )
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=默认输出目录)
    return parser.parse_args()


def 文件_sha256(path: Path) -> str:
    """只记录原始轨迹摘要，Wiki 不复制整份敏感审核文本。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 可移植路径(path: Path) -> str:
    """优先记录仓库相对路径，避免把当前机器的绝对目录写入版本库。"""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(项目根目录))
    except ValueError:
        return str(resolved)


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 导入动作(content: str) -> dict[str, Any] | None:
    """从专家消息中提取结构化动作，非法文本留给失败统计而不猜测修复。"""

    match = 动作模式.search(content)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def 分析专家轨迹(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """统计动作序列、来源、路径和类别，不把原始有害内容写入 Wiki。"""

    sequences = Counter()
    source_usage = Counter()
    path_usage = Counter()
    source_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    direct_by_safety = Counter()
    invalid_actions = 0
    for row in rows:
        tools = []
        for message in row["messages"]:
            if message.get("role") != "assistant":
                continue
            payload = 导入动作(str(message.get("content", "")))
            if payload is None:
                invalid_actions += 1
                continue
            tool = str(payload.get("tool", ""))
            arguments = payload.get("arguments", {})
            arguments = arguments if isinstance(arguments, dict) else {}
            tools.append(tool)
            if tool == "search":
                source_id = str(arguments.get("source_id", "未知来源"))
                path = str(arguments.get("path", "未知路径"))
                source_usage[source_id] += 1
                path_usage[f"{source_id}::{path}"] += 1
                for category in row.get("categories", []):
                    source_by_category[str(category)][source_id] += 1
        sequences[" → ".join(tools)] += 1
        if tools == ["finish"]:
            direct_by_safety["SAFE" if row["is_safe"] else "UNSAFE"] += 1
    return {
        "samples": len(rows),
        "invalid_actions": invalid_actions,
        "sequences": dict(sequences.most_common()),
        "source_usage": dict(source_usage.most_common()),
        "path_usage": dict(path_usage.most_common()),
        "source_by_category": {
            category: dict(counts.most_common())
            for category, counts in sorted(source_by_category.items())
        },
        "direct_by_safety": dict(direct_by_safety),
    }


def 分析真实评测(paths: list[Path]) -> dict[str, Any]:
    """从真实 rollout 提取失败记录，未来可由人工复核后进入 Case 候选区。"""

    runs = []
    failures: dict[str, set[str]] = defaultdict(set)
    aggregate_metrics: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        runs.append(
            {
                "path": 可移植路径(path),
                "sha256": 文件_sha256(path),
                "samples": payload.get("samples", 0),
                "adapter": payload.get("adapter"),
                "summary": summary,
            }
        )
        for name, value in summary.items():
            if isinstance(value, (int, float)):
                aggregate_metrics[name].append(float(value))
        for trace in payload.get("traces", []):
            record_id = str(trace.get("record_id", "未知记录"))
            metrics = trace.get("metrics", {})
            events = [str(step.get("event", "")) for step in trace.get("trace", [])]
            if float(metrics.get("safety_accuracy", 0.0)) < 1.0:
                failures["安全结论错误"].add(record_id)
            if float(metrics.get("category_f1", 0.0)) < 1.0:
                failures["风险类别不完整"].add(record_id)
            if float(metrics.get("source_selection_score", 0.0)) < 1.0:
                failures["库或目录定位不足"].add(record_id)
            if any(event.startswith("invalid") for event in events):
                failures["动作协议错误"].add(record_id)
            if not events or events[-1] != "finish":
                failures["未正常结束"].add(record_id)
    return {
        "runs": runs,
        "aggregate_metrics": {
            name: mean(values) for name, values in sorted(aggregate_metrics.items())
        },
        "failures": {name: sorted(ids) for name, ids in sorted(failures.items())},
    }


def 导航文档(expert: dict[str, Any]) -> str:
    """生成人可读的导航行为页面。"""

    lines = [
        "# 导航模式",
        "",
        "本页来自训练集专家轨迹的确定性统计，不包含原始审核正文，也不是线上提示词。",
        "",
        "## 动作序列",
        "",
        "| 序列 | 样本数 |",
        "|---|---:|",
    ]
    lines.extend(f"| `{sequence}` | {count} |" for sequence, count in expert["sequences"].items())
    lines.extend(["", "## 独立库使用", "", "| 来源 | search 次数 |", "|---|---:|"])
    lines.extend(f"| `{source}` | {count} |" for source, count in expert["source_usage"].items())
    lines.extend(["", "## 高频深层路径", "", "| 来源与路径 | 次数 |", "|---|---:|"])
    lines.extend(
        f"| `{path}` | {count} |"
        for path, count in list(expert["path_usage"].items())[:20]
    )
    return "\n".join(lines).rstrip() + "\n"


def 失败文档(evaluation: dict[str, Any]) -> str:
    """记录失败类型和 record_id，不自动把模型结论提升为事实。"""

    lines = [
        "# 失败模式",
        "",
        "这里只保存真实评测中可由金标签或环境确定的失败。记录不会自动进入正式 Case 库。",
        "",
    ]
    if not evaluation["runs"]:
        lines.append("尚未导入真实模型评测；完成 SFT/GRPO 冒烟后重新运行编译脚本。")
        return "\n".join(lines).rstrip() + "\n"
    for name, record_ids in evaluation["failures"].items():
        lines.extend([f"## {name}", "", f"数量：{len(record_ids)}", ""])
        lines.extend(f"- `{record_id}`" for record_id in record_ids[:50])
        if len(record_ids) > 50:
            lines.append(f"- 其余 {len(record_ids) - 50} 条保存在 `state.json`。")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def 来源文档(registry: dict[str, Any]) -> str:
    """展示源目录而不是复制三个业务库的具体内容。"""

    lines = [
        "# 来源地图",
        "",
        "经验 Wiki 只记录如何访问各库，不复制规则、Case 或知识正文。",
        "",
        "| 来源 | 后端 | 层级 | 职责 |",
        "|---|---|---|---|",
    ]
    for source in registry["sources"]:
        hierarchy = " → ".join(source["hierarchy_schema"])
        lines.append(
            f"| `{source['source_id']}` | `{source['connector']}` | {hierarchy} | {source['description']} |"
        )
    return "\n".join(lines) + "\n"


def 建议列表(offline: dict[str, Any], evaluation: dict[str, Any]) -> list[str]:
    """只给出可审计建议，不自动修改奖励、索引或训练数据。"""

    recommendations = []
    retrieval = offline.get("retrieval", {}) if isinstance(offline, dict) else {}
    if float(retrieval.get("top3_support_recall", 1.0)) < 0.8:
        recommendations.append("Top-3 支持召回低于 0.8：优先改进 parent-child 路由或真实语义向量，不扩大提示词。")
    if float(retrieval.get("top3_candidate_fraction", 0.0)) > 0.4:
        recommendations.append("Top-3 子树仍覆盖过多记录：提高最小搜索深度或训练 Agent 减少无效路径。")
    metrics = evaluation.get("aggregate_metrics", {})
    if metrics and float(metrics.get("invalid_action_rate", 0.0)) > 0.05:
        recommendations.append("无效动作率超过 5%：增加协议修复轨迹，而不是提高最终分类奖励。")
    if metrics and float(metrics.get("source_selection_score", 1.0)) < 0.8:
        recommendations.append("真实模型库选择不足：增加同一输入在不同来源上的对比轨迹。")
    if not recommendations:
        recommendations.append("当前自动阈值未发现明显退化；继续通过独立测试集和真实轨迹验证。")
    return recommendations


def 主程序() -> None:
    """生成稳定 Markdown 页面、机器状态和原始来源清单。"""

    args = 解析参数()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "patterns").mkdir(parents=True, exist_ok=True)
    rows = 读取_jsonl(args.sft_data)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    offline = (
        json.loads(args.offline_metrics.read_text(encoding="utf-8"))
        if args.offline_metrics.exists()
        else {}
    )
    expert = 分析专家轨迹(rows)
    evaluation = 分析真实评测(args.evaluation)
    recommendations = 建议列表(offline, evaluation)
    raw_manifest = {
        "sft_data": {
            "path": 可移植路径(args.sft_data),
            "sha256": 文件_sha256(args.sft_data),
            "samples": len(rows),
        },
        "offline_metrics": {
            "path": 可移植路径(args.offline_metrics),
            "sha256": 文件_sha256(args.offline_metrics) if args.offline_metrics.exists() else None,
        },
        "evaluations": evaluation["runs"],
    }
    state = {
        "schema_version": 1,
        "kind": "experience_wiki_not_skill",
        "raw_manifest": raw_manifest,
        "expert_patterns": expert,
        "offline_metrics": offline,
        "model_evaluation": evaluation,
        "recommendations": recommendations,
        "candidate_record_ids": sorted(
            {
                record_id
                for record_ids in evaluation["failures"].values()
                for record_id in record_ids
            }
        ),
    }
    (args.output_dir / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "source_map.md").write_text(来源文档(registry), encoding="utf-8")
    (args.output_dir / "patterns/navigation.md").write_text(
        导航文档(expert), encoding="utf-8"
    )
    (args.output_dir / "failures.md").write_text(
        失败文档(evaluation), encoding="utf-8"
    )
    recommendation_lines = ["# 下一轮建议", ""]
    recommendation_lines.extend(f"- {item}" for item in recommendations)
    (args.output_dir / "recommendations.md").write_text(
        "\n".join(recommendation_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "wiki": 可移植路径(args.output_dir),
                "raw_sources": raw_manifest,
                "recommendations": recommendations,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    主程序()
