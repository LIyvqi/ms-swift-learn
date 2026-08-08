#!/usr/bin/env bash
STYLE=cot PORT="${PORT:-8001}" exec bash "$(dirname -- "$0")/serve_teacher.sh"
