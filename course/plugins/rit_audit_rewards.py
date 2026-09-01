"""注册 RiT 内容审核的结果奖励、thinking rubric 与硬门控奖励。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import aiohttp
from swift.rewards import AsyncORM, ORM, orms


日志 = logging.getLogger(__name__)
项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))
核心模块 = import_module("course.32_rit_rubric_rl.rit_core")
结构化模块 = import_module("course.32_rit_rubric_rl.structured_core")
计算响应奖励 = 核心模块.计算响应奖励
计算思考rubric = 核心模块.计算思考rubric
计算本地RiT奖励 = 核心模块.计算本地RiT奖励
融合奖励 = 核心模块.融合奖励
解析评审分数 = 核心模块.解析评审分数
构造评审消息 = 核心模块.构造评审消息
计算结构化响应奖励 = 结构化模块.计算结构化响应奖励
计算结构化RiT奖励 = 结构化模块.计算结构化RiT奖励


def _训练参数() -> tuple[float, str, str]:
    """从环境读取门控消融设置，默认对齐论文 reasoning 配置。"""

    alpha = float(os.getenv("RIT_ALPHA", "1.0"))
    gate = os.getenv("RIT_GATE", "min")
    outcome_mode = os.getenv("RIT_OUTCOME_MODE", "strict")
    if not 0.0 <= alpha <= 1.0:
        raise RuntimeError("RIT_ALPHA 必须在 [0,1]")
    if gate not in {"min", "none", "max", "conditional"}:
        raise RuntimeError("RIT_GATE 只能是 min、none、max 或 conditional")
    if outcome_mode not in {"strict", "dense"}:
        raise RuntimeError("RIT_OUTCOME_MODE 只能是 strict 或 dense")
    return alpha, gate, outcome_mode


class RiTOutcomeReward(ORM):
    """只评价最终 SAFE/UNSAFE 与多标签，是论文 ORM 对照组。"""

    def __call__(
        self, completions, gold_is_safe, gold_categories, **kwargs
    ) -> list[float]:
        _, _, outcome_mode = _训练参数()
        return [
            计算响应奖励(text, safe, categories, mode=outcome_mode)
            for text, safe, categories in zip(
                completions, gold_is_safe, gold_categories
            )
        ]


class RiTThinkingReward(ORM):
    """报告六项本地可执行 thinking rubric 的平均分。"""

    def __call__(
        self,
        completions,
        prompt_text,
        response_text,
        gold_categories,
        **kwargs,
    ) -> list[float]:
        rewards = []
        for text, prompt, response, categories in zip(
            completions, prompt_text, response_text, gold_categories
        ):
            scores = 计算思考rubric(text, prompt, response, categories)
            rewards.append(sum(scores.values()) / len(scores))
        return rewards


class RiTGatedReward(ORM):
    """把本地 thinking rubric 与结果分按论文公式融合并硬门控。"""

    def __call__(
        self,
        completions,
        prompt_text,
        response_text,
        gold_is_safe,
        gold_categories,
        **kwargs,
    ) -> list[float]:
        alpha, gate, outcome_mode = _训练参数()
        rewards = []
        for text, prompt, response, safe, categories in zip(
            completions,
            prompt_text,
            response_text,
            gold_is_safe,
            gold_categories,
        ):
            detail = 计算本地RiT奖励(
                text,
                prompt,
                response,
                safe,
                categories,
                alpha=alpha,
                gate=gate,
                outcome_mode=outcome_mode,
            )
            rewards.append(detail.最终奖励)
        return rewards


class RiTStructuredOutcomeReward(ORM):
    """无自由 think 对照：只评价结构化输出中的最终审核结论。"""

    def __call__(
        self, completions, gold_is_safe, gold_categories, **kwargs
    ) -> list[float]:
        _, _, outcome_mode = _训练参数()
        return [
            计算结构化响应奖励(text, safe, categories, mode=outcome_mode)
            for text, safe, categories in zip(
                completions, gold_is_safe, gold_categories
            )
        ]


class RiTStructuredGatedReward(ORM):
    """评价证据、规则和边界短字段，再由正确结果执行最小值门控。"""

    def __call__(
        self,
        completions,
        prompt_text,
        response_text,
        gold_is_safe,
        gold_categories,
        **kwargs,
    ) -> list[float]:
        alpha, gate, outcome_mode = _训练参数()
        rewards = []
        for text, prompt, response, safe, categories in zip(
            completions,
            prompt_text,
            response_text,
            gold_is_safe,
            gold_categories,
        ):
            detail = 计算结构化RiT奖励(
                text,
                prompt,
                response,
                safe,
                categories,
                alpha=alpha,
                gate=gate,
                outcome_mode=outcome_mode,
            )
            rewards.append(detail.最终奖励)
        return rewards


class RiTAPIGatedReward(AsyncORM):
    """调用 OpenAI 兼容评审模型逐项打分，再执行相同硬门控。"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.api_base = os.getenv("RIT_JUDGE_API_BASE", "").rstrip("/")
        self.api_key = os.getenv("RIT_JUDGE_API_KEY", "")
        self.model = os.getenv("RIT_JUDGE_MODEL", "")
        self.timeout = float(os.getenv("RIT_JUDGE_TIMEOUT", "90"))
        self.concurrency = int(os.getenv("RIT_JUDGE_CONCURRENCY", "16"))
        if not self.api_base or not self.api_key or not self.model:
            raise RuntimeError(
                "API rubric 模式必须设置 RIT_JUDGE_API_BASE、"
                "RIT_JUDGE_API_KEY 和 RIT_JUDGE_MODEL"
            )
        if self.timeout <= 0 or self.concurrency <= 0:
            raise RuntimeError("评审超时和最大并发必须大于零")
        self.cache: dict[tuple[str, ...], float] = {}

    async def _评价单条(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        rubric_prompt: str,
        prompt_text: str,
        response_text: str,
        completion: str,
        gold_is_safe: Any,
        gold_categories: Any,
    ) -> float:
        cache_key = tuple(
            map(
                str,
                (
                    rubric_prompt,
                    prompt_text,
                    response_text,
                    completion,
                    gold_is_safe,
                    gold_categories,
                ),
            )
        )
        if cache_key in self.cache:
            return self.cache[cache_key]
        payload = {
            "model": self.model,
            "messages": 构造评审消息(
                rubric_prompt,
                prompt_text,
                response_text,
                completion,
                gold_is_safe,
                gold_categories,
            ),
            "temperature": 0,
            "max_tokens": 512,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with semaphore:
            for attempt in range(2):
                try:
                    async with session.post(
                        f"{self.api_base}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as response:
                        if response.status != 200:
                            error = await response.text()
                            日志.warning(
                                "RiT 评审 API 状态码 %s：%s",
                                response.status,
                                error[:160],
                            )
                            continue
                        body = await response.json()
                        content = body["choices"][0]["message"]["content"]
                        score = 解析评审分数(content)
                        self.cache[cache_key] = score
                        return score
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    json.JSONDecodeError,
                    KeyError,
                    OSError,
                    TypeError,
                    ValueError,
                ) as error:
                    日志.warning("第 %s 次 RiT rubric 评审失败：%s", attempt + 1, error)
        return 0.0

    async def __call__(
        self,
        completions,
        prompt_text,
        response_text,
        gold_is_safe,
        gold_categories,
        thinking_rubrics_prompt,
        **kwargs,
    ) -> list[float]:
        alpha, gate, outcome_mode = _训练参数()
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        semaphore = asyncio.Semaphore(self.concurrency)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            jobs = [
                self._评价单条(
                    session,
                    semaphore,
                    rubric,
                    prompt,
                    response,
                    completion,
                    safe,
                    categories,
                )
                for completion, prompt, response, safe, categories, rubric in zip(
                    completions,
                    prompt_text,
                    response_text,
                    gold_is_safe,
                    gold_categories,
                    thinking_rubrics_prompt,
                )
            ]
            thinking_scores = await asyncio.gather(*jobs)
        response_scores = [
            计算响应奖励(text, safe, categories, mode=outcome_mode)
            for text, safe, categories in zip(
                completions, gold_is_safe, gold_categories
            )
        ]
        return [
            融合奖励(thinking, response, alpha=alpha, gate=gate)
            for thinking, response in zip(thinking_scores, response_scores)
        ]


orms["course_rit_outcome"] = RiTOutcomeReward
orms["course_rit_thinking"] = RiTThinkingReward
orms["course_rit_gated"] = RiTGatedReward
orms["course_rit_api_gated"] = RiTAPIGatedReward
orms["course_rit_structured_outcome"] = RiTStructuredOutcomeReward
orms["course_rit_structured_gated"] = RiTStructuredGatedReward
