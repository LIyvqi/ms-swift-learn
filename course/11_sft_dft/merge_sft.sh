#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/alignment_common.sh"

SFT_ADAPTER="$(alignment_sft_checkpoint)"
if [[ -n "${SFT_MERGED_MODEL:-}" ]]; then
  OUTPUT="${SFT_MERGED_MODEL}"
elif [[ "${SMOKE:-0}" == "1" ]]; then
  OUTPUT="${ROOT}/outputs/11_sft_dft/sft_smoke_merged"
else
  OUTPUT="${ROOT}/models/alignment-news-sft-merged"
fi

swift export \
  --adapters "${SFT_ADAPTER}" \
  --merge_lora true \
  --output_dir "${OUTPUT}"
