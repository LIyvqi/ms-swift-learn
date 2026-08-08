#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

python "${PROJECT_ROOT}/verify_environment.py"
python "${COURSE_DIR}/tools/validate_assets.py"
