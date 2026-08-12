#!/usr/bin/env bash

# 对 Base 和每轮全参 Memory 做独立的留出改写问答，避免只按 token loss 选模型。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

RUN_DIR="${1:?用法：bash evaluate_memory_checkpoints.sh 全参训练运行目录}"
RUN_NAME="$(basename "$(realpath "${RUN_DIR}")")"
# 每次训练使用独立目录，避免旧 checkpoint 摘要混入新的动态选模。
RESULT_ROOT="${MEMORY_EVAL_OUTPUT:-${ROOT}/outputs/26_memo_rule_memory/memory_evaluation/${RUN_NAME}}"
mkdir -p "${RESULT_ROOT}"

python "${ROOT}/course/26_memo_rule_memory/evaluate_memory.py" \
  --output "${RESULT_ROOT}/base.jsonl"

for checkpoint in "${RUN_DIR}"/checkpoint-*; do
  [[ -d "${checkpoint}" ]] || continue
  step="$(basename "${checkpoint}" | sed 's/checkpoint-//')"
  python "${ROOT}/course/26_memo_rule_memory/evaluate_memory.py" \
    --model "${checkpoint}" \
    --output "${RESULT_ROOT}/checkpoint_${step}.jsonl"
done

python - "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.summary.json")):
    rows.append(json.loads(path.read_text(encoding="utf-8")))
best = max(rows, key=lambda row: (row["rule_f1"] + row["decision_accuracy"] + row["format_rate"], row["rule_f1"]))
(root / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(root / "best_model.txt").write_text((best["adapter"] or best["model"]) + "\n", encoding="utf-8")
print(json.dumps({"全部结果": rows, "最佳": best}, ensure_ascii=False, indent=2))
PY
