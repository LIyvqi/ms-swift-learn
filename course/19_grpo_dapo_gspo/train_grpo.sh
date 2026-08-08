#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export ALGO=grpo
exec bash "${ROOT}/course/19_grpo_dapo_gspo/train.sh"
