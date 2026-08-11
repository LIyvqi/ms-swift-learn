#!/usr/bin/env python3
"""按输入模态和输出协议统计多模态生成结果。"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

答案模式 = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
思考模式 = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
严格直接格式 = re.compile(
    r"^\s*<answer>\s*.+?\s*</answer>\s*$", re.DOTALL | re.IGNORECASE
)
严格思考格式 = re.compile(
    r"^\s*<think>\s*(?P<reason>.+?)\s*</think>\s*"
    r"<answer>\s*(?P<answer>.+?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
视觉失败词 = ("看不到", "无法查看", "没有图片", "未提供图片", "图片缺失")
模态顺序 = ("text_only", "image_only", "image_text")


def 规范答案(text: str) -> str:
    """与训练奖励使用同一套答案规范化规则。"""

    value = str(text).strip().upper().replace(",", "")
    value = re.sub(r"\s+", "", value)
    return value.strip("。.!！?？:：;；")


def 提取答案(text: str) -> str:
    """读取最后一个非空 answer 块。"""

    for value in reversed(答案模式.findall(text)):
        if value.strip():
            return 规范答案(value)
    return ""


def 加载_jsonl(path: Path) -> list[dict]:
    """读取非空 JSONL 记录。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 合并参考(rows: list[dict], references: list[dict]) -> None:
    """推理结果缺少教学元数据时，按固定验证集行序补齐。"""

    if len(rows) != len(references):
        raise RuntimeError(
            f"推理结果与参考数据行数不一致：{len(rows)} != {len(references)}"
        )
    for row, reference in zip(rows, references):
        for key in ("id", "source_id", "modality", "style", "final_answer"):
            if key in reference:
                row.setdefault(key, reference[key])


def 空统计() -> dict[str, float | int]:
    """创建可累加的内部计数器。"""

    return {
        "样本数": 0,
        "正确数": 0,
        "答案块数": 0,
        "严格格式数": 0,
        "非空思考数": 0,
        "视觉读取失败数": 0,
        "输出字符数": 0,
        "最大输出字符数": 0,
    }


def 汇总计数(counts: dict[str, float | int]) -> dict[str, float | int]:
    """把内部计数转换为对外指标。"""

    total = int(counts["样本数"])

    def rate(key: str) -> float:
        return float(counts[key]) / total if total else 0.0

    return {
        "样本数": total,
        "正确数": int(counts["正确数"]),
        "准确率": rate("正确数"),
        "答案块率": rate("答案块数"),
        "严格格式率": rate("严格格式数"),
        "非空思考率": rate("非空思考数"),
        "视觉读取失败率": rate("视觉读取失败数"),
        "平均输出字符数": float(counts["输出字符数"]) / total if total else 0.0,
        "最大输出字符数": int(counts["最大输出字符数"]),
    }


def 评测(rows: list[dict]) -> dict:
    """计算总体、按模态和按输出风格拆分的指标。"""

    counters: dict[tuple[str, str], dict[str, float | int]] = defaultdict(空统计)
    for row in rows:
        response = str(row.get("response", ""))
        expected = 规范答案(row.get("final_answer", ""))
        predicted = 提取答案(response)
        modality = str(row.get("modality", "unknown"))
        style = str(row.get("style", "unknown"))
        thinking = [
            value.strip() for value in 思考模式.findall(response) if value.strip()
        ]
        if style == "direct":
            strict = (
                bool(严格直接格式.fullmatch(response))
                and "<think>" not in response.lower()
            )
        elif style == "cot":
            match = 严格思考格式.fullmatch(response)
            strict = bool(match and 12 <= len(match.group("reason").strip()) <= 4000)
        else:
            strict = bool(
                严格直接格式.fullmatch(response) or 严格思考格式.fullmatch(response)
            )

        groups = (("overall", "all"), ("modality", modality), ("style", style))
        for group in groups:
            counts = counters[group]
            counts["样本数"] += 1
            counts["正确数"] += int(bool(predicted) and predicted == expected)
            counts["答案块数"] += int(bool(predicted))
            counts["严格格式数"] += int(strict)
            counts["非空思考数"] += int(bool(thinking))
            counts["视觉读取失败数"] += int(
                modality != "text_only" and any(word in response for word in 视觉失败词)
            )
            counts["输出字符数"] += len(response)
            counts["最大输出字符数"] = max(int(counts["最大输出字符数"]), len(response))

    return {
        "总体": 汇总计数(counters[("overall", "all")]),
        "按模态": {
            modality: 汇总计数(counters[("modality", modality)])
            for modality in 模态顺序
            if ("modality", modality) in counters
        },
        "按风格": {
            style: 汇总计数(counters[("style", style)])
            for style in ("direct", "cot")
            if ("style", style) in counters
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评测多模态 Direct/CoT 生成结果")
    parser.add_argument("result", type=Path, help="swift infer 生成的 JSONL")
    parser.add_argument(
        "--reference", type=Path, required=True, help="对应验证集 JSONL"
    )
    parser.add_argument("--output", type=Path, help="可选的 JSON 汇总输出")
    args = parser.parse_args()

    rows = 加载_jsonl(args.result)
    if not rows:
        raise RuntimeError("推理结果为空")
    合并参考(rows, 加载_jsonl(args.reference))
    summary = {"文件": str(args.result), **评测(rows)}
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
