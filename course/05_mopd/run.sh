#!/usr/bin/env bash

set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"

bash "${HERE}/serve_cot.sh" >"${HERE}/logs/cot.log" 2>&1 &
COT_PID=$!
bash "${HERE}/serve_direct.sh" >"${HERE}/logs/direct.log" 2>&1 &
DIRECT_PID=$!
cleanup() {
  kill "${COT_PID}" "${DIRECT_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

TEACHER_PIDS="${COT_PID},${DIRECT_PID}" \
  python "${HERE}/wait_for_teachers.py" http://127.0.0.1:8001/v1/models http://127.0.0.1:8002/v1/models
bash "${HERE}/train.sh"
