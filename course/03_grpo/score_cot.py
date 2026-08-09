#!/usr/bin/env python3
"""统计 GSM8K 显式 CoT 生成结果的答案、结构与过程代理指标。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from importlib import import_module
from pathlib import Path

项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

奖励模块 = import_module("course.plugins.gsm8k_rewards")
GSM8KCoTCalculation = 奖励模块.GSM8KCoTCalculation
GSM8KCoTConsistency = 奖励模块.GSM8KCoTConsistency
GSM8KCoTGrounding = 奖励模块.GSM8KCoTGrounding
GSM8KCoTStructure = 奖励模块.GSM8KCoTStructure

思考模式 = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def 提取框选答案(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else ""


def 规范数值(text: str) -> Decimal | None:
    cleaned = text.replace(",", "").replace("$", "").strip()
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def 答案正确(response: str, expected: str) -> bool:
    predicted = 提取框选答案(response)
    predicted_number, expected_number = 规范数值(predicted), 规范数值(expected)
    if predicted_number is not None and expected_number is not None:
        return abs(predicted_number - expected_number) <= Decimal("1e-8")
    return bool(predicted and predicted == expected.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="评测显式 CoT-GRPO 生成结果")
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=项目根目录 / "datasets/gsm8k_1k/prompts_cot_explicit_val.jsonl",
        help="包含 question 和 final_answer 的参考数据；推理输出缺少这些列时按行合并",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in args.result.open(encoding="utf-8") if line.strip()
    ]
    if not rows:
        raise RuntimeError("推理结果为空")

    if "question" not in rows[0] or "final_answer" not in rows[0]:
        references = [
            json.loads(line)
            for line in args.reference.open(encoding="utf-8")
            if line.strip()
        ]
        if len(references) != len(rows):
            raise RuntimeError(
                f"推理结果与参考数据行数不一致：{len(rows)} != {len(references)}"
            )
        for row, reference in zip(rows, references):
            row.setdefault("question", reference["question"])
            row.setdefault("final_answer", reference["final_answer"])

    responses = [row["response"] for row in rows]
    questions = [row["question"] for row in rows]
    final_answers = [row["final_answer"] for row in rows]
    correct = sum(
        答案正确(response, expected)
        for response, expected in zip(responses, final_answers)
    )
    opened = sum("<think>" in response for response in responses)
    closed = sum(bool(思考模式.search(response)) for response in responses)
    nonempty = sum(
        any(content.strip() for content in 思考模式.findall(response))
        for response in responses
    )
    boxed = sum(bool(提取框选答案(response)) for response in responses)

    structure_scores = GSM8KCoTStructure()(responses)
    calculation_scores = GSM8KCoTCalculation()(responses, questions, final_answers)
    grounding_scores = GSM8KCoTGrounding()(responses, questions)
    consistency_scores = GSM8KCoTConsistency()(responses)

    summary = {
        "文件": str(args.result),
        "样本数": len(rows),
        "答案正确数": correct,
        "答案正确率": correct / len(rows),
        "思考开始率": opened / len(rows),
        "思考闭合率": closed / len(rows),
        "非空思考率": nonempty / len(rows),
        "严格格式率": sum(structure_scores) / len(rows),
        "框选答案率": boxed / len(rows),
        "平均计算过程奖励": sum(calculation_scores) / len(rows),
        "平均题目数值覆盖": sum(grounding_scores) / len(rows),
        "过程答案一致率": sum(consistency_scores) / len(rows),
        "平均输出字符数": sum(map(len, responses)) / len(rows),
        "最大输出字符数": max(map(len, responses)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
