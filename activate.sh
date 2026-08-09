#!/usr/bin/env bash

# 激活位于工作区中的持久化 ms-swift/ROCm 训练环境。
_MS_SWIFT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_MS_SWIFT_PERSIST_ROOT="$(cd -- "${_MS_SWIFT_ROOT}/.." && pwd)"

# GitHub CLI、登录状态与 Git 全局配置也放在持久化工作区中。
export PATH="${_MS_SWIFT_PERSIST_ROOT}/.local/bin:${PATH}"
export GH_CONFIG_DIR="${_MS_SWIFT_PERSIST_ROOT}/.config/gh"
export GIT_CONFIG_GLOBAL="${_MS_SWIFT_PERSIST_ROOT}/.config/git/config"

# 模型缓存和编译缓存必须位于 /mnt/workspace 下，重启后才能继续保留。
export MODELSCOPE_CACHE="${_MS_SWIFT_ROOT}/.cache/modelscope"
export MODELSCOPE_HOME="${_MS_SWIFT_ROOT}/.cache/modelscope_home"
export HF_HOME="${_MS_SWIFT_ROOT}/.cache/huggingface"
export TORCH_HOME="${_MS_SWIFT_ROOT}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${_MS_SWIFT_ROOT}/.cache/torch_extensions"
export TRITON_CACHE_DIR="${_MS_SWIFT_ROOT}/.cache/triton"
export XDG_CACHE_HOME="${_MS_SWIFT_ROOT}/.cache/xdg"

# ModelScope 与阿里云模型仓库直接下载；保留平台代理供 GitHub 等站点使用。
_MS_SWIFT_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
for _MS_SWIFT_DIRECT_HOST in modelscope.cn .modelscope.cn .aliyuncs.com; do
  if [[ ",${_MS_SWIFT_NO_PROXY}," != *",${_MS_SWIFT_DIRECT_HOST},"* ]]; then
    _MS_SWIFT_NO_PROXY="${_MS_SWIFT_NO_PROXY:+${_MS_SWIFT_NO_PROXY},}${_MS_SWIFT_DIRECT_HOST}"
  fi
done
export NO_PROXY="${_MS_SWIFT_NO_PROXY}"
export no_proxy="${_MS_SWIFT_NO_PROXY}"
export PIP_INDEX_URL="https://pypi.org/simple"
export PYTHONNOUSERSITE=1

mkdir -p \
  "${MODELSCOPE_CACHE}" \
  "${MODELSCOPE_HOME}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${GH_CONFIG_DIR}" \
  "$(dirname -- "${GIT_CONFIG_GLOBAL}")"

chmod 700 "${GH_CONFIG_DIR}" "$(dirname -- "${GIT_CONFIG_GLOBAL}")"

# 这里复用机器镜像自带的 ROCm 版 PyTorch，同时把 ms-swift 及其兼容依赖
# 保存在当前持久化项目目录中，避免普通 CUDA 软件包覆盖平台定制环境。
source "${_MS_SWIFT_ROOT}/.venv/bin/activate"

unset _MS_SWIFT_ROOT _MS_SWIFT_PERSIST_ROOT _MS_SWIFT_NO_PROXY _MS_SWIFT_DIRECT_HOST
