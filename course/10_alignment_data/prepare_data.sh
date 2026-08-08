#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

python "${ROOT}/tools/prepare_alignment_data.py" \
  --source "${ROOT}/datasets/fudan_news_4class/sft_train.jsonl" \
  --output "${ROOT}/datasets/alignment_news"
