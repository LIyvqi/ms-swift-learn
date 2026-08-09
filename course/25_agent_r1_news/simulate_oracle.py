"""用确定性专家策略跑完整环境，验证多轮状态转移和奖励闭环。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from statistics import mean
from typing import Any

项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

知识模块 = import_module("course.25_agent_r1_news.knowledge_pipeline")
环境模块 = import_module("course.25_agent_r1_news.agent_system")
RuleKnowledgeBase = 知识模块.RuleKnowledgeBase
NewsPolicyEnvironment = 环境模块.NewsPolicyEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟 Agent-R1 新闻专家轨迹")
    parser.add_argument(
        "--rules",
        type=Path,
        default=项目根目录 / "datasets/agent_r1_news/knowledge_rules.jsonl",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=项目根目录 / "datasets/agent_r1_news/rl_smoke.jsonl",
    )
    args = parser.parse_args()

    knowledge = RuleKnowledgeBase.from_jsonl(args.rules)
    rows = [
        json.loads(line) for line in args.dataset.open(encoding="utf-8") if line.strip()
    ]
    rewards: dict[str, list[float]] = defaultdict(list)
    turns: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        env = NewsPolicyEnvironment(knowledge, row["env_config"])
        env.reset()
        total_reward = 0.0
        last_info: dict[str, Any] = {}
        while not env.done:
            action = env.expert_action()
            _, reward, _, last_info = env.step(action)
            total_reward += reward
        rewards[row["task"]].append(total_reward)
        turns[row["task"]].append(len(last_info["trace"]))

    summary = {
        task: {
            "samples": len(rewards[task]),
            "mean_reward": mean(rewards[task]),
            "mean_turns": mean(turns[task]),
        }
        for task in sorted(rewards)
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
