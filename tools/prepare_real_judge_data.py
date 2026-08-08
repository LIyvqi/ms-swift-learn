#!/usr/bin/env python3
"""用 GSM8K 标准答案构造 1～5 分的回归式 LLM-as-a-Judge 教学数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


系统提示 = (
    "你是数值解题回答评分员。根据题目、参考答案和候选回答，按 1 到 5 分评分："
    "5=完全正确，4=结论非常接近但有轻微误差，3=部分正确，2=主要错误但与问题相关，1=完全错误。"
    "先在 <think> 中给出简短、可核对的误差依据，然后严格以 <score>N</score> 结束。"
)


def 读取_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def 写入_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 提取答案(text: str) -> float:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"无法提取 GSM8K 答案：{text[-80:]}")
    return float(match.group(1).replace(",", ""))


def 格式化数值(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def 候选答案(gold: float, score: int, index: int) -> float:
    方向 = -1 if index % 2 else 1
    基础步长 = max(1.0, abs(gold) * 0.02)
    if score == 5:
        return gold
    if score == 4:
        return gold + 方向 * 基础步长
    if score == 3:
        return gold + 方向 * max(2.0, abs(gold) * 0.15)
    if score == 2:
        return gold + 方向 * max(5.0, abs(gold) * 0.60)
    return -abs(gold) - 37 - index


def 构造问题(question: str, gold: float, candidate: float) -> str:
    return (
        f"题目：{question}\n\n"
        f"参考数值答案：{格式化数值(gold)}\n"
        f"候选回答：计算结果是 {格式化数值(candidate)}。\n\n"
        "请依据评分量表给出 1 到 5 分。"
    )


def 构造记录(source: dict, question_index: int, score: int) -> tuple[dict, dict]:
    gold = 提取答案(source["answer"])
    candidate = 候选答案(gold, score, question_index)
    user = 构造问题(source["question"], gold, candidate)
    误差 = abs(candidate - gold)
    rationale = (
        f"参考答案为 {格式化数值(gold)}，候选值为 {格式化数值(candidate)}，"
        f"绝对误差为 {格式化数值(误差)}；按给定量表对应 {score} 分。"
    )
    公共字段 = {
        "score": score,
        "question_id": f"gsm8k-{question_index:04d}",
        "reference_answer": 格式化数值(gold),
        "candidate_answer": 格式化数值(candidate),
    }
    prompt_messages = [
        {"role": "system", "content": 系统提示},
        {"role": "user", "content": user},
    ]
    sft = {
        "messages": prompt_messages + [{
            "role": "assistant",
            "content": f"<think>{rationale}</think><score>{score}</score>",
        }],
        **公共字段,
    }
    prompt = {"messages": prompt_messages, **公共字段}
    return sft, prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="构造回归感知 REAL 教学数据")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = 读取_jsonl(args.source)[:24]
    splits = {
        "train": source[:18],
        "val": source[18:24],
        "smoke": source[:2],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for split, questions in splits.items():
        sft_rows: list[dict] = []
        prompt_rows: list[dict] = []
        起点 = 0 if split != "val" else 18
        for offset, question in enumerate(questions):
            for score in range(1, 6):
                sft, prompt = 构造记录(question, 起点 + offset, score)
                sft_rows.append(sft)
                prompt_rows.append(prompt)
        for view, rows in (("sft", sft_rows), ("prompts", prompt_rows)):
            path = args.output / f"{view}_{split}.jsonl"
            写入_jsonl(path, rows)
            manifest[path.name] = {
                "条数": len(rows),
                "SHA-256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    (args.output / "checksums.json").write_text(
        json.dumps({"文件": manifest, "分数量表": "1 到 5，每个题目各生成一条"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"REAL 教学数据已生成：{args.output}")


if __name__ == "__main__":
    main()
