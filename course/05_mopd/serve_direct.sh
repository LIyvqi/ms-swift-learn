#!/usr/bin/env bash
STYLE=direct PORT="${PORT:-8002}" exec bash "$(dirname -- "$0")/serve_teacher.sh"
