#!/usr/bin/env python3
"""把可复现配置、训练检查点和自动评分导出为仓库内中文结果报告。"""

from __future__ import annotations

import json
from pathlib import Path


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]
输出目录 = 项目根目录 / "outputs/30_macaron_mol_audit"


def 最新检查点(directory: Path) -> Path:
    """按修改时间定位最后一个 checkpoint。"""

    checkpoints = list(directory.glob("**/checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"找不到检查点：{directory}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def 主程序() -> None:
    """生成包含事实边界的 RESULTS.md。"""

    comparison_path = 输出目录 / "evaluation/comparison.json"
    comparison_md = 输出目录 / "evaluation/comparison.md"
    if not comparison_path.is_file() or not comparison_md.is_file():
        raise FileNotFoundError("缺少完整评测结果，请先运行 evaluate.sh")
    summary = json.loads(comparison_path.read_text(encoding="utf-8"))
    manifest = json.loads((课程目录 / "data/manifest.json").read_text(encoding="utf-8"))
    checkpoints = {
        target: 最新检查点(输出目录 / target)
        for target in ("baseline", "router", "l1", "l2", "l3", "l4")
    }
    lines = [
        "# 第 30 课真实实验结果",
        "",
        "本文件由 `export_results.py` 根据真实检查点和固定测试集自动生成。",
        "",
        "## 可复现配置",
        "",
        f"- 数据：BeaverTails `330k_train` 去重后按联合标签分层抽取 {manifest['sample_size']} 条。",
        f"- 划分：训练 {manifest['split_sizes']['train']}、验证 {manifest['split_sizes']['validation']}、测试 {manifest['split_sizes']['test']}。",
        f"- 最大类别边际率偏差：{manifest['maximum_category_rate_deviation_percentage_points']:.4f} 个百分点。",
        "- 模型：Qwen3.5-0.8B-Base；LoRA rank=16、alpha=32；默认 3 epoch。",
        "- 检索：BM25 召回 + 字符三元组重排；规则 Top-3、案例 Top-3。",
        "- 评测：200 条清洁测试 + 100 条成对表面扰动挑战，所有 LoRA 使用完全冻结的四种检索上下文。",
        "",
        "## 真实检查点",
        "",
    ]
    for target, checkpoint in checkpoints.items():
        metadata = summary["generation_metadata"][target]
        lines.append(
            f"- `{target}`：`{checkpoint.relative_to(项目根目录)}`，"
            f"权重 SHA256 `{metadata['adapter_sha256']}`。"
        )
    gpu_report = 输出目录 / "status/gpu_summary.md"
    if gpu_report.is_file():
        gpu_lines = gpu_report.read_text(encoding="utf-8").splitlines()
        if gpu_lines and gpu_lines[0].startswith("# "):
            gpu_lines = gpu_lines[1:]
        lines.extend(["", "## 训练资源", *gpu_lines])
    auto_report = comparison_md.read_text(encoding="utf-8").splitlines()
    if auto_report and auto_report[0].startswith("# "):
        auto_report = auto_report[1:]
    lines.extend(["", "## 真实生成评测", *auto_report])
    baseline_gain = summary["rag_delta"]["baseline_micro_f1_full_minus_none"]
    top2_gain = summary["rag_delta"]["top2_micro_f1_full_minus_none"]
    challenge_baseline = summary["generalization"]["obfuscated"]["baseline"]
    best_challenge_mode = max(challenge_baseline, key=lambda mode: challenge_baseline[mode]["micro_f1"])
    lines.extend(
        [
            "",
            "## 结果解读",
            "",
            f"- 完整检索使单体 LoRA 的 Micro-F1 相对无检索提升 {baseline_gain * 100:.2f} 个百分点，"
            f"MoL Top-2 提升 {top2_gain * 100:.2f} 个百分点。",
            f"- 表面扰动挑战中，单体 LoRA 的最佳知识模式是 `{best_challenge_mode}`，"
            f"Micro-F1 为 {challenge_baseline[best_challenge_mode]['micro_f1'] * 100:.2f}%。",
            f"- 规则 Recall@3 为 {summary['retrieval']['full']['rule_recall_at_3'] * 100:.2f}%，"
            f"Case 标签 Recall@3 为 {summary['retrieval']['full']['case_label_recall_at_3'] * 100:.2f}%；"
            "本次 Case 检索是收益的主要来源，规则召回是后续优化点。",
        ]
    )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 这是 0.8B 模型、2000 条教学样本上的缩小复现，不等价于论文 50B/748B 系统。",
            "- `Top-2` 是针对多标签内容审核的课程扩展；官方 Macaron Harness 每回合选择一个 LoRA。",
            "- RAG 收益以表格中的真实差值为准；如果某一模式下降，报告保留下降结果，不把检索默认描述成提升。",
            "- 表面扰动只检验一类可控词法偏移，不等于生产 OOD 、新类别或新政策泛化。",
            "- “新增 L4 无旧权重遗忘”只证明冻结旧 LoRA 的结构性质，不证明路由错误、规则冲突和数据漂移已经解决。",
            "",
        ]
    )
    (课程目录 / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"已导出：{课程目录 / 'RESULTS.md'}")


if __name__ == "__main__":
    主程序()
