#!/usr/bin/env python3
"""把多组多模态评测汇总为便于课程比较的 Markdown 表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def 百分比(value: float) -> str:
    """把零到一之间的指标渲染成百分比。"""

    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="比较多组多模态生成评测")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.summaries):
        raise RuntimeError("labels 数量必须与评测文件数量一致")
    labels = args.labels or [path.stem for path in args.summaries]
    rows = []
    for label, path in zip(labels, args.summaries):
        data = json.loads(path.read_text(encoding="utf-8"))
        overall = data["总体"]
        modalities = data["按模态"]
        rows.append(
            [
                label,
                str(overall["样本数"]),
                百分比(overall["准确率"]),
                百分比(modalities["text_only"]["准确率"]),
                百分比(modalities["image_only"]["准确率"]),
                百分比(modalities["image_text"]["准确率"]),
                百分比(overall["严格格式率"]),
                百分比(overall["非空思考率"]),
                百分比(overall["视觉读取失败率"]),
                f"{overall['平均输出字符数']:.1f}",
            ]
        )

    header = [
        "模型/协议",
        "样本",
        "总体准确率",
        "纯文本",
        "纯图像",
        "图文混合",
        "严格格式率",
        "非空思考率",
        "视觉失败率",
        "平均字符",
    ]
    lines = [
        "# 多模态固定验证集对比",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(
        [
            "",
            "说明：固定验证集只有 40 条，分模态结果用于链路排错和教学比较，不能当作公开 benchmark 成绩。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"多模态对比已写入：{args.output}")


if __name__ == "__main__":
    main()
