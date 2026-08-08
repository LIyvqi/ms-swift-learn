#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

python "${ROOT}/tools/prepare_fudan_cot.py" \
  --source-rl "${ROOT}/datasets/fudan_news_4class/rl_train.jsonl" \
  --source-val "${ROOT}/datasets/fudan_news_4class/val.jsonl" \
  --annotations "${ROOT}/datasets/fudan_news_cot_50/annotations.json" \
  --output "${ROOT}/datasets/fudan_news_cot_50"
