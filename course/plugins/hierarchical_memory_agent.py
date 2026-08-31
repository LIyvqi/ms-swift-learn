"""注册多源分层记忆审核环境、调度器和确定性奖励。"""

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

网关模块 = import_module("course.31_hierarchical_memory_agent.memory_gateway")
环境模块 = import_module("course.31_hierarchical_memory_agent.agent_environment")
分层记忆网关 = 网关模块.分层记忆网关
分层记忆审核环境 = 环境模块.分层记忆审核环境
默认注册表路径 = 环境模块.默认注册表路径

网关缓存: dict[str, Any] = {}


def 获取网关(path: str | Path) -> Any:
    """按绝对注册表路径缓存只读连接器和索引。"""

    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else 项目根目录 / candidate
    key = str(resolved.resolve())
    if key not in 网关缓存:
        网关缓存[key] = 分层记忆网关(Path(key))
    return 网关缓存[key]


class HierarchicalMemoryAuditEnv(Env):
    """把纯 Python 环境适配为 ms-swift 异步 GYM 接口。"""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)
        registry_path = env_config.get("registry_path", 默认注册表路径())
        self.core = 分层记忆审核环境(获取网关(registry_path), env_config)

    async def reset(
        self, config: RolloutInferRequest
    ) -> tuple[str, dict[str, Any], str]:
        return self.core.reset()

    async def step(self, action: Messages) -> tuple[str, float, bool, dict[str, Any]]:
        completion = action[-1].get("content", "") if action else ""
        return self.core.step(completion)

    async def close(self):
        return None


class HierarchicalMemoryScheduler(GYMScheduler):
    """把逐步环境奖励、轨迹和指标完整写入 rollout_infos。"""

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        uuid = infer_request.uuid
        env = self._envs.get(uuid)
        if env is None:
            return {"done": True, "rollout_infos": {}}
        next_observation, reward, done, info = await env.step(
            deepcopy(infer_request.messages)
        )
        self._total_rewards[uuid] = self._total_rewards.get(uuid, 0.0) + float(reward)
        self._step_rewards.setdefault(uuid, []).append(float(reward))
        self._pending_obs[uuid] = None if done else next_observation
        rollout_infos = {
            "total_reward": self._total_rewards[uuid],
            "step_rewards": list(self._step_rewards.get(uuid, [])),
            "gym_done": done,
            "task": "decision",
            "agent_trace": info.get("trace", []),
            "task_metrics": info.get("metrics", {}),
            "final": info.get("final", {}),
            "last_event": info.get("event"),
        }
        if done:
            await self._close_and_remove(uuid)
        return {"done": done, "rollout_infos": rollout_infos}


class DecisionQualityReward(ORM):
    """以安全结论和多标签 F1 为主，证据覆盖为辅。"""

    def __call__(self, completions, rollout_infos, **kwargs) -> list[float]:
        rewards = []
        for info in rollout_infos:
            metrics = info.get("task_metrics", {}) if isinstance(info, dict) else {}
            value = (
                float(metrics.get("safety_accuracy", 0.0))
                + 0.5 * float(metrics.get("category_f1", 0.0))
                + 0.15 * float(metrics.get("evidence_grounding", 0.0))
            ) / 1.65
            rewards.append(value)
        return rewards


class MemoryNavigationReward(ORM):
    """奖励选中相关知识源、目录并召回支持类别。"""

    def __call__(self, completions, rollout_infos, **kwargs) -> list[float]:
        rewards = []
        for info in rollout_infos:
            metrics = info.get("task_metrics", {}) if isinstance(info, dict) else {}
            rewards.append(
                0.45 * float(metrics.get("source_selection_score", 0.0))
                + 0.55 * float(metrics.get("memory_category_recall", 0.0))
            )
        return rewards


class GroundingCalibrationReward(ORM):
    """奖励有效记忆引用、合法协议和与正确性一致的置信度。"""

    def __call__(self, completions, rollout_infos, **kwargs) -> list[float]:
        rewards = []
        for info in rollout_infos:
            metrics = info.get("task_metrics", {}) if isinstance(info, dict) else {}
            rewards.append(
                0.45 * float(metrics.get("memory_grounding", 0.0))
                + 0.30 * float(metrics.get("confidence_score", 0.0))
                + 0.25 * float(metrics.get("task_schema_score", 0.0))
            )
        return rewards


class ProtocolEfficiencyReward(ORM):
    """鼓励短规划、有效动作和按需而不是强制检索。"""

    def __call__(self, completions, rollout_infos, **kwargs) -> list[float]:
        rewards = []
        for info in rollout_infos:
            metrics = info.get("task_metrics", {}) if isinstance(info, dict) else {}
            trace = info.get("agent_trace", []) if isinstance(info, dict) else []
            invalid = sum(
                str(step.get("event", "")).startswith("invalid") for step in trace
            )
            value = (
                0.35 * float(metrics.get("protocol_score", 0.0))
                + 0.20 * float(metrics.get("thinking_score", 0.0))
                + 0.45 * float(metrics.get("tool_efficiency", 0.0))
                - 0.10 * invalid
            )
            rewards.append(max(0.0, min(1.0, value)))
        return rewards


envs["course_hierarchical_memory_audit"] = HierarchicalMemoryAuditEnv
multi_turns["course_hierarchical_memory_scheduler"] = HierarchicalMemoryScheduler
orms["course_hierarchical_decision"] = DecisionQualityReward
orms["course_hierarchical_navigation"] = MemoryNavigationReward
orms["course_hierarchical_grounding"] = GroundingCalibrationReward
orms["course_hierarchical_efficiency"] = ProtocolEfficiencyReward
