#!/usr/bin/env python3
"""从原始 GSM8K Prompt 视图生成显式思考版 GRPO 数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

项目根目录 = Path(__file__).resolve().parents[2]
显式思考系统提示 = (
    "你是数学推理助手。请先在 <think> 和 </think> 之间给出非空、可核验的分步计算，"
    "至少写出一个形如“算式=结果”的等式，并使用题目中的数值条件。"
    "推理应简洁，最多六步；得到结果后立即闭合 </think>，不要反复检查或复述题目。"
    "然后在思考块之后仅输出 \\boxed{最终答案}。不要省略思考块，也不要在框选答案后追加内容。"
)


def 转换记录(record: dict) -> dict:
    """替换系统提示并确认在线强化学习数据没有预填 assistant。"""

    converted = dict(record)
    messages = [dict(message) for message in record["messages"]]
    if any(message.get("role") == "assistant" for message in messages):
        raise ValueError(f"GRPO 数据不能包含 assistant：{record.get('id', '未知编号')}")
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = 显式思考系统提示
    else:
        messages.insert(0, {"role": "system", "content": 显式思考系统提示})
    converted["messages"] = messages
    converted["teacher_tag"] = "explicit_cot"
    return converted


def 转换文件(source: Path, target: Path) -> int:
    """逐行转换 JSONL，并保留奖励函数需要的全部顶层字段。"""

    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number} 不是合法 JSON") from error
            for field in ("messages", "question", "solution", "final_answer"):
                if field not in record:
                    raise ValueError(f"{source}:{line_number} 缺少字段 {field}")
            rows.append(转换记录(record))

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成显式思考版 GSM8K GRPO 数据")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=项目根目录 / "datasets" / "gsm8k_1k",
        help="原始数据与输出数据所在目录",
    )
    args = parser.parse_args()

    generated = []
    for split in ("train", "val", "smoke"):
        source = args.data_dir / f"prompts_cot_{split}.jsonl"
        target = args.data_dir / f"prompts_cot_explicit_{split}.jsonl"
        count = 转换文件(source, target)
        generated.append(target)
        print(f"已生成 {target}：{count} 条")

    # 与课程原有数据保持相同的完整性记录方式，便于发现意外改动。
    checksum_path = args.data_dir / "checksums.json"
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    for path in generated:
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path.write_text(
        json.dumps(dict(sorted(checksums.items())), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"已更新 {checksum_path}")


if __name__ == "__main__":
    main()
