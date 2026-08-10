"""注册 Agent-R1 新闻环境、多轮调度器和多任务奖励。"""

from __future__ import annotations

import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any

from swift.infer_engine.protocol import RolloutInferRequest
from swift.rewards import ORM, orms
from swift.rollout.gym_env import Env, envs
from swift.rollout.multi_turn import GYMScheduler, multi_turns
from swift.template import Messages

项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

知识模块 = import_module("course.25_agent_r1_news.knowledge_pipeline")
环境模块 = import_module("course.25_agent_r1_news.agent_system")
RuleKnowledgeBase = 知识模块.RuleKnowledgeBase
NewsPolicyEnvironment = 环境模块.NewsPolicyEnvironment
默认规则路径 = 环境模块.default_knowledge_path

知识库缓存: dict[str, Any] = {}


def 获取知识库(path: str | Path) -> Any:
    resolved = str(Path(path).resolve())
    if resolved not in 知识库缓存:
        知识库缓存[resolved] = RuleKnowledgeBase.from_jsonl(resolved)
    return 知识库缓存[resolved]


class AgentR1NewsEnv(Env):
    """将纯 Python 状态机适配到 ms-swift 的异步 GYM Env 接口。"""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)
        knowledge_path = env_config.get("knowledge_path", 默认规则路径())
        self.core = NewsPolicyEnvironment(获取知识库(knowledge_path), env_config)

    async def reset(
        self, config: RolloutInferRequest
    ) -> tuple[str, dict[str, Any], str]:
        return self.core.reset()

    async def step(self, action: Messages) -> tuple[str, float, bool, dict[str, Any]]:
        completion = action[-1].get("content", "") if action else ""
        return self.core.step(completion)

    async def close(self):
        return None


class AgentR1NewsScheduler(GYMScheduler):
    """保留每步环境信息，使过程奖励能够读取完整 Agent 轨迹。"""

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        uuid = infer_request.uuid
        env = self._envs.get(uuid)
        if env is None:
            return {"done": True, "rollout_infos": {}}

        next_obs, reward, done, info = await env.step(deepcopy(infer_request.messages))
        self._total_rewards[uuid] = self._total_rewards.get(uuid, 0.0) + float(reward)
        self._step_rewards.setdefault(uuid, []).append(float(reward))
        self._pending_obs[uuid] = None if done else next_obs

        rollout_infos = {
            "total_reward": self._total_rewards[uuid],
            "step_rewards": list(self._step_rewards.get(uuid, [])),
            "gym_done": done,
            "task": info.get("task"),
            "agent_trace": info.get("trace", []),
            "task_metrics": info.get("metrics", {}),
            "final": info.get("final", {}),
            "last_event": info.get("event"),
        }
        if done:
            await self._close_and_remove(uuid)
        return {"done": done, "rollout_infos": rollout_infos}


class TaskMetricReward(ORM):
    """只为指定任务返回对应指标，其他任务返回 None。"""

    task_name = ""
    metric_name = ""

    def __call__(
        self, completions, task, rollout_infos, **kwargs
    ) -> list[float | None]:
        rewards: list[float | None] = []
        for current_task, info in zip(task, rollout_infos):
            if current_task != self.task_name:
                rewards.append(None)
                continue
            metrics = info.get("task_metrics", {}) if isinstance(info, dict) else {}
            rewards.append(float(metrics.get(self.metric_name, 0.0)))
        return rewards


class AgentNewsRetrievalReward(TaskMetricReward):
    task_name = "retrieve"
    metric_name = "retrieval_f1"


class AgentNewsCompositionReward(TaskMetricReward):
    task_name = "compose"
    metric_name = "composition_f1"


class AgentNewsDecisionReward(ORM):
    """决策任务以分类正确为主，同时检查规则和证据。"""

    def __call__(
        self, completions, task, rollout_infos, **kwargs
    ) -> list[float | None]:
        rewards: list[float | None] = []
        for current_task, info in zip(task, rollout_infos):
            if current_task != "decision":
                rewards.append(None)
                continue
            metrics = info.get("task_metrics", {}) if isinstance(info, dict) else {}
            reward = (
                float(metrics.get("decision_accuracy", 0.0))
                + 0.3 * float(metrics.get("rule_compliance", 0.0))
                + 0.2 * float(metrics.get("evidence_coverage", 0.0))
            )
            rewards.append(reward / 1.5)
        return rewards


class AgentNewsProtocolReward(ORM):
    """奖励所有任务遵守结构化动作协议并减少无效工具调用。"""

    def __call__(self, completions, task, rollout_infos, **kwargs) -> list[float]:
        rewards = []
        expected_turns = {"retrieve": 3, "compose": 3, "decision": 4}
        for current_task, info in zip(task, rollout_infos):
            metrics = info.get("task_metrics", {}) if isinstance(info, dict) else {}
            protocol = float(metrics.get("protocol_score", 0.0))
            thinking = float(metrics.get("thinking_score", 0.0))
            task_schema = float(metrics.get("task_schema_score", 0.0))
            trace = info.get("agent_trace", []) if isinstance(info, dict) else []
            invalid_count = sum(
                str(step.get("event", "")).startswith("invalid") for step in trace
            )
            extra_turns = max(len(trace) - expected_turns.get(current_task, 4), 0)
            efficiency = max(0.0, 1.0 - 0.2 * invalid_count - 0.05 * extra_turns)
            rewards.append(
                0.35 * protocol
                + 0.20 * thinking
                + 0.15 * efficiency
                + 0.30 * task_schema
            )
        return rewards


class AgentNewsReflectionReward(ORM):
    """奖励三个任务通过查询改写提高召回。"""

    def __call__(
        self, completions, task, rollout_infos, **kwargs
    ) -> list[float | None]:
        rewards: list[float | None] = []
        for current_task, info in zip(task, rollout_infos):
            if current_task not in {"retrieve", "compose", "decision"}:
                rewards.append(None)
                continue
            metrics = info.get("task_metrics", {}) if isinstance(info, dict) else {}
            gain = float(metrics.get("reflection_gain", 0.0))
            success = float(metrics.get("reflection_success", 0.0))
            rewards.append(max(0.0, min(1.0, 0.7 * max(gain, 0.0) + 0.3 * success)))
        return rewards


envs["course_agent_r1_news"] = AgentR1NewsEnv
multi_turns["course_agent_r1_news_scheduler"] = AgentR1NewsScheduler
orms["course_agent_news_retrieval"] = AgentNewsRetrievalReward
orms["course_agent_news_composition"] = AgentNewsCompositionReward
orms["course_agent_news_decision"] = AgentNewsDecisionReward
orms["course_agent_news_protocol"] = AgentNewsProtocolReward
orms["course_agent_news_reflection"] = AgentNewsReflectionReward
