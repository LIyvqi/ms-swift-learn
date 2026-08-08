#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

SOURCE_DIR="${ROOT}/downloads/datasets/zh_cls_fudan-news"
SOURCE_FILE="${SOURCE_DIR}/zh_cls_fudan-news.csv"
REVISION="1810dce2722d76e714db8290c9a4de3f6c8340f2"

if [[ ! -f "${SOURCE_FILE}" ]]; then
  modelscope download \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${SOURCE_DIR}" \
    --include 'zh_cls_fudan-news.csv' README.md \
    damo/zh_cls_fudan-news
fi

python "${ROOT}/tools/prepare_fudan_classification.py" \
  --source "${SOURCE_FILE}" \
  --output "${ROOT}/datasets/fudan_news_4class"
