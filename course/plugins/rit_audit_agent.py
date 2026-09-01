"""注册极简 RiT 安全审核 Agent、调度器和三路可诊断奖励。"""

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
环境模块 = import_module("course.32_rit_rubric_rl.agent_environment")
记忆模块 = import_module("course.32_rit_rubric_rl.agent_memory")
极简RiT审核环境 = 环境模块.极简RiT审核环境
极简审核记忆 = 记忆模块.极简审核记忆

记忆缓存: dict[tuple[str, str], Any] = {}


def _解析路径(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else 项目根目录 / candidate


def 获取记忆(rules_path: str | Path, cases_path: str | Path) -> Any:
    """按两个只读文件的绝对路径缓存检索索引。"""

    rules = _解析路径(rules_path).resolve()
    cases = _解析路径(cases_path).resolve()
    key = (str(rules), str(cases))
    if key not in 记忆缓存:
        记忆缓存[key] = 极简审核记忆(rules, cases)
    return 记忆缓存[key]


class RitAuditAgentEnv(Env):
    """把纯 Python 环境适配为 ms-swift 异步 GYM 接口。"""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)
        memory = 获取记忆(env_config["rules_path"], env_config["cases_path"])
        self.core = 极简RiT审核环境(memory, env_config)

    async def reset(
        self, config: RolloutInferRequest
    ) -> tuple[str, dict[str, Any], str]:
        return self.core.reset()

    async def step(self, action: Messages) -> tuple[str, float, bool, dict[str, Any]]:
        completion = action[-1].get("content", "") if action else ""
        return self.core.step(completion)

    async def close(self):
        return None


class RitAuditAgentScheduler(GYMScheduler):
    """把完整轨迹和 RiT 三层奖励写入 rollout_infos。"""

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
            "task": "rit_audit_agent",
            "agent_trace": info.get("trace", []),
            "task_metrics": info.get("metrics", {}),
            "rubric_scores": info.get("rubric_scores", {}),
            "final": info.get("final", {}),
            "last_event": info.get("event"),
        }
        if done:
            await self._close_and_remove(uuid)
        return {"done": done, "rollout_infos": rollout_infos}


class _MetricReward(ORM):
    """从环境终态读取一个确定性指标，异常轨迹统一记零。"""

    metric_name = ""

    def __call__(self, completions, rollout_infos, **kwargs) -> list[float]:
        rewards = []
        for info in rollout_infos:
            metrics = info.get("task_metrics", {}) if isinstance(info, dict) else {}
            value = float(metrics.get(self.metric_name, 0.0))
            rewards.append(max(0.0, min(1.0, value)))
        return rewards


class RitAgentResponseReward(_MetricReward):
    """只看最终安全结论和完整多标签是否精确正确。"""

    metric_name = "response_reward"


class RitAgentProcessReward(_MetricReward):
    """报告动作、证据、规则、案例、边界和短链效率六项均分。"""

    metric_name = "process_reward"


class RitAgentGatedReward(_MetricReward):
    """返回 min(过程量规, 精确结果)，错误结论不能靠漂亮轨迹获益。"""

    metric_name = "gated_reward"


envs["course_rit_audit_agent"] = RitAuditAgentEnv
multi_turns["course_rit_audit_agent_scheduler"] = RitAuditAgentScheduler
orms["course_rit_agent_response"] = RitAgentResponseReward
orms["course_rit_agent_process"] = RitAgentProcessReward
orms["course_rit_agent_gated"] = RitAgentGatedReward
