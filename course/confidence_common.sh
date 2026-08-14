#!/usr/bin/env bash

# 第 28、29 课共用的持久化环境与检查点工具。

CONFIDENCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIDENCE_SWIFT_DIR="${CONFIDENCE_ROOT}/third_party/ms-swift-official-4.4.3"
CONFIDENCE_SWIFT_COMMIT="e1287928be4451b9ed5e2fb00a24ad3c8f61287b"

activate_confidence_env() {
  source "${CONFIDENCE_ROOT}/activate.sh"
  if [[ ! -d "${CONFIDENCE_SWIFT_DIR}/.git" ]]; then
    git clone --depth 1 --branch v4.4.3 \
      https://github.com/modelscope/ms-swift.git "${CONFIDENCE_SWIFT_DIR}"
  fi
  local actual_commit actual_tag
  actual_commit="$(git -C "${CONFIDENCE_SWIFT_DIR}" rev-parse HEAD)"
  actual_tag="$(git -C "${CONFIDENCE_SWIFT_DIR}" describe --tags --exact-match HEAD)"
  if [[ "${actual_commit}" != "${CONFIDENCE_SWIFT_COMMIT}" || "${actual_tag}" != "v4.4.3" ]]; then
    echo "ms-swift 版本不符：${actual_tag} ${actual_commit}" >&2
    return 1
  fi
  export PYTHONPATH="${CONFIDENCE_SWIFT_DIR}:${PYTHONPATH:-}"
}

latest_confidence_checkpoint() {
  local directory="$1"
  local checkpoint
  checkpoint="$({ find "${directory}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "找不到检查点：${directory}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}
