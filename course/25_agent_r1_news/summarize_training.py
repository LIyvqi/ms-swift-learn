"""从 ms-swift 日志提取首尾指标和最优验证指标，便于记录实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def 读取日志(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 ms-swift 训练日志")
    parser.add_argument("logging_jsonl", type=Path)
    args = parser.parse_args()

    rows = 读取日志(args.logging_jsonl)
    training = [row for row in rows if "loss" in row and "global_step/max_steps" in row]
    evaluations = [row for row in rows if "eval_loss" in row]
    final = next(
        (
            row
            for row in reversed(rows)
            if "train_runtime" in row or "global_step" in row
        ),
        rows[-1] if rows else {},
    )
    summary = {
        "日志": str(args.logging_jsonl),
        "首个训练点": training[0] if training else {},
        "最后训练点": training[-1] if training else {},
        "最佳验证点": min(evaluations, key=lambda row: row["eval_loss"])
        if evaluations
        else {},
        "最终摘要": final,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
