#!/usr/bin/env python3
"""为 ms-swift 课程创建可确定复现的 1000 条 GSM8K 数据。

数据来自 ModelScope 托管的 GSM8K。脚本使用固定随机种子抽取 1000 条，
划分为 900 条训练数据和 100 条验证数据，并生成三类视图：带显式推理过程的
监督微调数据、只保留最终答案的监督微调数据，以及只含提示词的强化学习数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from modelscope.msdatasets import MsDataset


SEED = 42
SYSTEM_COT = (
    "You are a helpful math assistant. Solve the problem step by step and "
    "put the final answer within \\boxed{}."
)
SYSTEM_DIRECT = (
    "You are a helpful math assistant. Return only the final answer within "
    "\\boxed{}, without showing reasoning."
)


def parse_answer(answer: str) -> tuple[str, str]:
    match = re.search(r"\n####\s*(.+?)\s*$", answer)
    if not match:
        raise ValueError(f"GSM8K answer has no final marker: {answer!r}")
    rationale = answer[: match.start()].strip()
    final = match.group(1).strip()
    return rationale, final


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_views(records: list[dict]) -> dict[str, list[dict]]:
    result = {
        "cot": [],
        "direct": [],
        "mixed": [],
        "prompts_cot": [],
        "prompts_direct": [],
        "prompts_multi": [],
    }
    for index, record in enumerate(records):
        question = record["question"].strip()
        original = record["answer"].strip()
        rationale, final = parse_answer(original)
        common = {
            "id": f"gsm8k-{index:04d}",
            "question": question,
            "solution": original,
            "final_answer": final,
        }
        cot = {
            **common,
            "teacher_tag": "cot",
            "messages": [
                {"role": "system", "content": SYSTEM_COT},
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": f"<think>\n{rationale}\n</think>\n\\boxed{{{final}}}",
                },
            ],
        }
        direct = {
            **common,
            "teacher_tag": "direct",
            "messages": [
                {"role": "system", "content": SYSTEM_DIRECT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"\\boxed{{{final}}}"},
            ],
        }
        prompt_cot = {
            **common,
            "teacher_tag": "cot",
            "messages": [
                {"role": "system", "content": SYSTEM_COT},
                {"role": "user", "content": question},
            ],
        }
        prompt_direct = {
            **common,
            "teacher_tag": "direct",
            "messages": [
                {"role": "system", "content": SYSTEM_DIRECT},
                {"role": "user", "content": question},
            ],
        }
        result["cot"].append(cot)
        result["direct"].append(direct)
        result["mixed"].append(cot if index % 2 == 0 else direct)
        result["prompts_cot"].append(prompt_cot)
        result["prompts_direct"].append(prompt_direct)
        result["prompts_multi"].append(prompt_cot if index % 2 == 0 else prompt_direct)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/gsm8k_1k"))
    args = parser.parse_args()

    dataset = MsDataset.load(
        "modelscope/gsm8k",
        subset_name="main",
        split="train",
        trust_remote_code=True,
    )
    sample = dataset.shuffle(seed=SEED).select(range(1000))
    raw = [sample[i] for i in range(len(sample))]
    views = build_views(raw)

    out = args.output
    dump_jsonl(out / "source_1k.jsonl", raw)
    for name, rows in views.items():
        dump_jsonl(out / f"{name}_train.jsonl", rows[:900])
        dump_jsonl(out / f"{name}_val.jsonl", rows[900:])
        dump_jsonl(out / f"{name}_smoke.jsonl", rows[:16])

    checksums = {}
    for path in sorted(out.glob("*.jsonl")):
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "checksums.json").write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已在 {out.resolve()} 生成 {len(raw)} 条可确定复现的 GSM8K 数据")
    for name, rows in views.items():
        print(f"{name}：训练={len(rows[:900])}，验证={len(rows[900:])}，冒烟测试=16")


if __name__ == "__main__":
    main()
