#!/usr/bin/env bash

# RiT 主实验：用可验证 Agent 行为量规，并由最终精确正确性硬门控。
set -euo pipefail
AGENT_METHOD=rit exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_train_agent_grpo.sh"
