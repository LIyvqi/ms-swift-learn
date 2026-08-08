#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "缺少 uv，请先在基础镜像中安装 uv" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  uv venv --system-site-packages .venv
fi

mkdir -p third_party
SWIFT_ROOT="third_party/ms-swift-v4.4.3"
EXPECTED_COMMIT="e1287928be4451b9ed5e2fb00a24ad3c8f61287b"

if [[ ! -d "${SWIFT_ROOT}/.git" ]]; then
  git clone --branch v4.4.3 --depth 1 \
    https://github.com/modelscope/ms-swift.git "${SWIFT_ROOT}"
fi

ACTUAL_COMMIT="$(git -C "${SWIFT_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "ms-swift 提交不匹配：期望 ${EXPECTED_COMMIT}，实际 ${ACTUAL_COMMIT}" >&2
  exit 1
fi

# 必须禁用依赖解析，避免普通 CUDA 软件包覆盖基础镜像中的 ROCm PyTorch。
uv pip install --python .venv/bin/python --no-deps -r requirements-local.txt

echo "环境叠加层安装完成。请执行：source ./activate.sh"
