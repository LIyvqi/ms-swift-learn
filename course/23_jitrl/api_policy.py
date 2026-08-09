#!/usr/bin/env python3
"""从 OpenAI 兼容推理 API 获取 JitRL 所需的候选动作分数。"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from protocol_env import 构造提示

支持的打分模式 = ("constrained_logprobs", "top_logprobs", "verbalized")


def _解析_json_object(text: str) -> dict:
    """容忍 Markdown 代码块，只截取第一个完整 JSON 对象。"""

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"API 没有返回 JSON 对象：{text!r}")
    return json.loads(cleaned[start:end + 1])


class OpenAI动作策略:
    """把 OpenAI 兼容 API 的候选概率转换为可加优势的对数分数。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        score_mode: str = "constrained_logprobs",
        top_logprobs: int = 20,
        timeout: float = 120.0,
        max_retries: int = 3,
        client: Any | None = None,
    ):
        if score_mode not in 支持的打分模式:
            raise ValueError(f"未知打分模式 {score_mode!r}，可选值：{支持的打分模式}")
        if top_logprobs <= 0:
            raise ValueError("top_logprobs 必须大于 0")
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        self.client = client
        self.base_url = base_url
        self.model = model
        self.score_mode = score_mode
        self.top_logprobs = top_logprobs

    def 检查服务(self) -> list[str]:
        """读取 OpenAI 兼容服务端公开的模型列表。"""

        response = self.client.models.list()
        return [item.id for item in response.data]

    @staticmethod
    def _候选编号(count: int) -> list[str]:
        if not 2 <= count <= 9:
            raise ValueError("当前单 token 编号实现要求候选动作数位于 2～9")
        return [str(index) for index in range(1, count + 1)]

    def _读取_top_logprobs(self, response: Any, digits: Sequence[str]) -> list[float]:
        """从第一个生成位置读取所有候选数字的对数概率。"""

        try:
            positions = response.choices[0].logprobs.content
            candidates = positions[0].top_logprobs
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError("API 响应不含 Chat Completions logprobs") from error

        found: dict[str, float] = {}
        for candidate in candidates:
            token = str(candidate.token).strip()
            if token in digits:
                found[token] = max(found.get(token, -math.inf), float(candidate.logprob))
        missing = [digit for digit in digits if digit not in found]
        if missing:
            raise RuntimeError(
                f"API 的 top_logprobs 缺少候选 {missing}。"
                "ms-swift/vLLM 请使用 constrained_logprobs；普通第三方 API 可提高 top_logprobs，"
                "仍缺失时改用 verbalized。"
            )
        return [found[digit] for digit in digits]

    def _概率模式打分(self, prompt: str, digits: Sequence[str], constrained: bool) -> list[float]:
        """请求一个动作编号，并读取该位置上的候选 logprobs。"""

        extra_body = None
        if constrained:
            # ms-swift 会把该正则传给 vLLM 结构化输出，使 top-k 只包含合法编号。
            extra_body = {"structured_outputs_regex": f"[1-{len(digits)}]"}
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": True,
            "top_logprobs": len(digits) if constrained else self.top_logprobs,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self.client.chat.completions.create(**kwargs)
        return self._读取_top_logprobs(response, digits)

    def _文本置信度打分(self, prompt: str, digits: Sequence[str]) -> list[float]:
        """让不提供 logprobs 的 API 显式报告概率，再转换成 log(probability)。"""

        example_scores = ",".join(str(len(digits) - index) for index in range(len(digits)))
        scoring_prompt = (
            f"{prompt}\n\n不要直接选择一个答案。请评估每个编号成功的相对可能性，"
            f"只输出 JSON；格式示例为 {{\"scores\":[{example_scores}]}}，示例数值不代表答案。"
            f"scores 必须恰有 {len(digits)} 个非负数，顺序对应编号 {','.join(digits)}，总和应大于 0。"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": scoring_prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        row = _解析_json_object(content)
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != len(digits):
            raise RuntimeError(f"API 返回的 scores 数量错误：{scores!r}")
        try:
            values = [max(float(value), 0.0) for value in scores]
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"API 返回的 scores 不是数值：{scores!r}") from error
        denominator = sum(values)
        if denominator <= 0:
            raise RuntimeError("API 返回的置信度总和必须大于 0")
        epsilon = 1e-12
        return [math.log(value / denominator + epsilon) for value in values]

    def 打分(self, state) -> list[float]:
        """返回与 state.candidates 顺序一致的基础动作对数分数。"""

        digits = self._候选编号(len(state.candidates))
        prompt = 构造提示(state)
        if self.score_mode == "constrained_logprobs":
            return self._概率模式打分(prompt, digits, constrained=True)
        if self.score_mode == "top_logprobs":
            return self._概率模式打分(prompt, digits, constrained=False)
        return self._文本置信度打分(prompt, digits)

    def 批量打分(self, states: Sequence, concurrency: int) -> dict[tuple, list[float]]:
        """并发请求有限状态集合；缓存键与本地模型实验完全一致。"""

        if concurrency <= 0:
            raise ValueError("concurrency 必须大于 0")

        def score_one(state):
            key = (state.phase_index, state.batch_tag, state.candidates)
            return key, self.打分(state)

        if concurrency == 1:
            pairs = map(score_one, states)
            return dict(pairs)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            return dict(executor.map(score_one, states))
