#!/usr/bin/env bash

# 从确定性数据生成开始，依次完成训练、选模、七组审核、消融、规模探针和结果导出。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

python "${ROOT}/course/26_memo_rule_memory/prepare_data.py"
python "${ROOT}/course/26_memo_rule_memory/audit_data.py"
PYTHONPATH="${ROOT}/course/26_memo_rule_memory" \
  python -m pytest -q "${ROOT}/course/26_memo_rule_memory/test_memo.py"

bash "${ROOT}/course/26_memo_rule_memory/run_memory_full.sh"
run_dir="$(find "${ROOT}/outputs/26_memo_rule_memory/full_sft" -mindepth 1 -maxdepth 1 -type d -name 'v*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -z "${run_dir}" ]]; then
  printf '没有找到正式训练运行目录。\n' >&2
  exit 1
fi

eval_dir="${ROOT}/outputs/26_memo_rule_memory/memory_evaluation/$(basename "${run_dir}")"
bash "${ROOT}/course/26_memo_rule_memory/evaluate_memory_checkpoints.sh" "${run_dir}"
memory_model="$(<"${eval_dir}/best_model.txt")"

python "${ROOT}/course/26_memo_rule_memory/run_audit_experiment.py" \
  --memory-model "${memory_model}" \
  --output-dir "${ROOT}/outputs/26_memo_rule_memory/audit_evaluation_fragmented"

python "${ROOT}/course/26_memo_rule_memory/run_audit_experiment.py" \
  --memory-model "${memory_model}" \
  --methods memo_structured_deterministic \
  --memory-grounding full \
  --output-dir "${ROOT}/outputs/26_memo_rule_memory/audit_ablation_full_content"

python "${ROOT}/course/26_memo_rule_memory/probe_rule_scaling.py"
python "${ROOT}/course/26_memo_rule_memory/export_results.py"
printf '第 26 课完整实验已完成，最佳 Memory：%s\n' "${memory_model}"
