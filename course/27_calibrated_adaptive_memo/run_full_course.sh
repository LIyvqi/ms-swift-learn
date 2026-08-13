#!/usr/bin/env bash

# 从数据生成、测试到完整真实实验的一键复现入口。
set -euo pipefail

课程目录="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
项目根目录="$(cd -- "${课程目录}/../.." && pwd)"
Swift目录="${项目根目录}/third_party/ms-swift-official-4.4.3"
Memory检查点="${MEMORY_MODEL:-${项目根目录}/outputs/26_memo_rule_memory/full_sft/v0-20260813-021836/checkpoint-33}"
输出目录="${OUTPUT_DIR:-${项目根目录}/outputs/27_calibrated_adaptive_memo/full_experiment}"

source "${项目根目录}/activate.sh"

# PyPI 尚未提供 4.4.3；用官方标签及固定提交作为权威版本证据。
if [[ ! -d "${Swift目录}/.git" ]]; then
  git clone --depth 1 --branch v4.4.3 \
    https://github.com/modelscope/ms-swift.git "${Swift目录}"
fi

export PYTHONPATH="${课程目录}:${项目根目录}/course/26_memo_rule_memory:${PYTHONPATH:-}"

python "${课程目录}/prepare_data.py"
python -m pytest -q "${课程目录}/test_ca_memo.py"

python "${课程目录}/run_experiment.py" \
  --memory-model "${Memory检查点}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --output-dir "${输出目录}"

python "${课程目录}/export_results.py" --input-dir "${输出目录}"
