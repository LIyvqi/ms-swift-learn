#!/usr/bin/env bash

# 等其他 swift 任务彻底释放 GPU 后，自动启动多模态课程冒烟链路。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_DIR="${PROJECT_ROOT}/outputs/multimodal_course_status"
mkdir -p "${STATUS_DIR}"
printf '等待开始：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/waiting.txt"

# 连续三次检查都没有 swift 训练进程才启动，避免旧任务短暂退出后又被监控器拉起。
idle_checks=0
while (( idle_checks < 3 )); do
  if pgrep -f '[/]swift (sft|rlhf)' >/dev/null; then
    idle_checks=0
    printf '[%s] 检测到其他 swift 训练，继续等待。\n' "$(date --iso-8601=seconds)" >>"${STATUS_DIR}/wait.log"
  else
    idle_checks=$((idle_checks + 1))
    printf '[%s] GPU 训练空闲检查 %s/3。\n' "$(date --iso-8601=seconds)" "${idle_checks}" >>"${STATUS_DIR}/wait.log"
  fi
  sleep 60
done

rm -f "${STATUS_DIR}/waiting.txt"
printf '自动启动：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/started.txt"
exec bash "${PROJECT_ROOT}/course/run_multimodal_smoke.sh"
