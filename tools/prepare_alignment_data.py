#!/usr/bin/env python3
"""从复旦新闻四分类数据构造一套可复用的人类偏好对齐教学数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


类别顺序 = ("政治", "财经", "体育", "计算机")


def 读取_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def 写入_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 错误类别(row: dict) -> str:
    """确定性挑选一个错误类别，使每次重新生成的数据完全一致。"""

    正确位置 = 类别顺序.index(row["label"])
    偏移 = int(hashlib.sha256(row["record_id"].encode()).hexdigest()[:8], 16) % 3 + 1
    return 类别顺序[(正确位置 + 偏移) % len(类别顺序)]


def 去除回答(messages: list[dict]) -> list[dict]:
    return [dict(message) for message in messages if message["role"] != "assistant"]


def 构造视图(rows: list[dict]) -> dict[str, list[dict]]:
    views: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        prompt = 去除回答(row["messages"])
        正确回答 = f"\\boxed{{{row['label']}}}"
        错误回答 = f"\\boxed{{{错误类别(row)}}}"
        公共字段 = {
            "label": row["label"],
            "record_id": row["record_id"],
        }

        views["sft"].append({"messages": prompt + [{"role": "assistant", "content": 正确回答}], **公共字段})
        views["pairwise"].append({
            "messages": prompt + [{"role": "assistant", "content": 正确回答}],
            "rejected_response": 错误回答,
            "margin": 1.0,
            **公共字段,
        })
        # RM 额外加入“类别正确但格式残缺”的困难负例。
        # 仅用错误类别负例时，奖励模型可能只学会识别类别词，而忽略右花括号等协议细节。
        views["rm"].extend([
            {
                "messages": prompt + [{"role": "assistant", "content": 正确回答}],
                "rejected_response": 错误回答,
                "margin": 1.0,
                **公共字段,
                "record_id": row["record_id"] + "-类别负例",
                "negative_type": "错误类别",
            },
            {
                "messages": prompt + [{"role": "assistant", "content": 正确回答}],
                "rejected_response": f"\\boxed{{{row['label']}",
                "margin": 1.0,
                **公共字段,
                "record_id": row["record_id"] + "-格式负例",
                "negative_type": "缺少右花括号",
            },
        ])
        # ms-swift 会在 batch 内循环错位回答来构造 KTO 的 KL 对照。
        # 只有四种短答案时，不同样本可能拥有完全相同的回答，模板会拒绝这种 KL 对照；
        # 因此 KTO 视图要求模型同时抄写唯一记录编号，让每条回答在字符串层面唯一。
        kto_prompt = [dict(message) for message in prompt]
        kto_prompt[-1]["content"] += f"\n\n回答末尾另起一行抄写：记录编号：{row['record_id']}"
        kto_chosen = f"{正确回答}\n记录编号：{row['record_id']}"
        kto_rejected = f"{错误回答}\n记录编号：{row['record_id']}"
        views["kto"].extend([
            {
                "messages": kto_prompt + [{"role": "assistant", "content": kto_chosen}],
                "label": True,
                "gold_label": row["label"],
                "record_id": row["record_id"] + "-偏好",
            },
            {
                "messages": kto_prompt + [{"role": "assistant", "content": kto_rejected}],
                "label": False,
                "gold_label": row["label"],
                "record_id": row["record_id"] + "-拒绝",
            },
        ])
        views["prompts"].append({"messages": prompt, **公共字段})
        用户问题 = next(message["content"] for message in reversed(prompt) if message["role"] == "user")
        views["opsd"].append({
            "messages": prompt,
            "teacher_prompt": (
                f"{用户问题}\n\n"
                f"特权参考信息：这条新闻的人工标准类别是“{row['label']}”。"
                "请利用参考信息，只输出一个形如\\boxed{类别}的答案。"
            ),
            **公共字段,
        })
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description="构造统一的人类偏好对齐教学数据")
    parser.add_argument("--source", type=Path, required=True, help="复旦分类 SFT 源数据")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    args = parser.parse_args()

    分组: dict[str, list[dict]] = defaultdict(list)
    for row in 读取_jsonl(args.source):
        分组[row["label"]].append(row)

    for label in 类别顺序:
        if len(分组[label]) < 80:
            raise ValueError(f"类别 {label} 至少需要 80 条，实际只有 {len(分组[label])} 条")

    train = [row for label in 类别顺序 for row in 分组[label][:64]]
    val = [row for label in 类别顺序 for row in 分组[label][64:80]]
    smoke = [row for label in 类别顺序 for row in 分组[label][:4]]

    args.output.mkdir(parents=True, exist_ok=True)
    文件摘要: dict[str, dict[str, object]] = {}
    for split, rows in (("train", train), ("val", val), ("smoke", smoke)):
        for view, view_rows in 构造视图(rows).items():
            path = args.output / f"{view}_{split}.jsonl"
            写入_jsonl(path, view_rows)
            文件摘要[path.name] = {
                "条数": len(view_rows),
                "SHA-256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    manifest = {
        "来源": str(args.source),
        "划分": "每类前 64 条训练、后 16 条验证；冒烟集取每类前 4 条",
        "类别": list(类别顺序),
        "文件": 文件摘要,
    }
    (args.output / "checksums.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已生成 {len(文件摘要)} 个数据文件：{args.output}")


if __name__ == "__main__":
    main()
