#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

python "${ROOT}/tools/prepare_real_judge_data.py" \
  --source "${ROOT}/datasets/gsm8k_1k/source_1k.jsonl" \
  --output "${ROOT}/datasets/real_judge_1to5"
