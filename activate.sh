#!/usr/bin/env bash

# 激活位于工作区中的持久化 ms-swift/ROCm 训练环境。
_MS_SWIFT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 模型缓存和编译缓存必须位于 /mnt/workspace 下，重启后才能继续保留。
export MODELSCOPE_CACHE="${_MS_SWIFT_ROOT}/.cache/modelscope"
export MODELSCOPE_HOME="${_MS_SWIFT_ROOT}/.cache/modelscope_home"
export HF_HOME="${_MS_SWIFT_ROOT}/.cache/huggingface"
export TORCH_HOME="${_MS_SWIFT_ROOT}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${_MS_SWIFT_ROOT}/.cache/torch_extensions"
export TRITON_CACHE_DIR="${_MS_SWIFT_ROOT}/.cache/triton"
export XDG_CACHE_HOME="${_MS_SWIFT_ROOT}/.cache/xdg"

# 这台机器可直接访问 ModelScope，下载时不需要代理。
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PIP_INDEX_URL="https://pypi.org/simple"
export PYTHONNOUSERSITE=1

mkdir -p \
  "${MODELSCOPE_CACHE}" \
  "${MODELSCOPE_HOME}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${XDG_CACHE_HOME}"

# 这里复用机器镜像自带的 ROCm 版 PyTorch，同时把 ms-swift 及其兼容依赖
# 保存在当前持久化项目目录中，避免普通 CUDA 软件包覆盖平台定制环境。
source "${_MS_SWIFT_ROOT}/.venv/bin/activate"

unset _MS_SWIFT_ROOT
