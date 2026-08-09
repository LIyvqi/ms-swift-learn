#!/usr/bin/env bash
set -euo pipefail

# 从任意目录运行时都切换到仓库根目录，并激活持久化环境。
COURSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${COURSE_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source ./activate.sh

# 先验证与模型无关的闭式解，再运行真实 Qwen 基础 logits 实验。
python course/23_jitrl/test_closed_form.py
python course/23_jitrl/test_api_policy.py
python course/23_jitrl/run_experiment.py "$@"
