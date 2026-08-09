#!/usr/bin/env python3
"""用伪造 OpenAI 响应验证三种 API 候选动作打分模式。"""

from __future__ import annotations

import math
from types import SimpleNamespace

from api_policy import OpenAI动作策略
from protocol_env import 所有决策状态


def 对象(**kwargs):
    return SimpleNamespace(**kwargs)


class 伪造完成接口:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class 伪造客户端:
    def __init__(self, responses):
        self.completions = 伪造完成接口(responses)
        self.chat = 对象(completions=self.completions)
        self.models = 对象(list=lambda: 对象(data=[对象(id="教学模型")]))


def logprobs_response():
    candidates = [
        对象(token="1", logprob=-0.2),
        对象(token="2", logprob=-1.7),
        对象(token="3", logprob=-2.4),
    ]
    return 对象(choices=[对象(logprobs=对象(content=[对象(top_logprobs=candidates)]))])


def 测试约束概率模式() -> None:
    client = 伪造客户端([logprobs_response()])
    policy = OpenAI动作策略(
        base_url="http://test/v1",
        api_key="不使用",
        model="教学模型",
        score_mode="constrained_logprobs",
        client=client,
    )
    scores = policy.打分(所有决策状态()[0])
    assert scores == [-0.2, -1.7, -2.4]
    request = client.completions.requests[0]
    assert request["extra_body"]["structured_outputs_regex"] == "[1-3]"
    assert request["max_tokens"] == 1 and request["logprobs"] is True
    assert policy.检查服务() == ["教学模型"]


def 测试普通概率模式缺项报错() -> None:
    response = logprobs_response()
    response.choices[0].logprobs.content[0].top_logprobs.pop()
    client = 伪造客户端([response])
    policy = OpenAI动作策略(
        base_url="http://test/v1",
        api_key="不使用",
        model="教学模型",
        score_mode="top_logprobs",
        client=client,
    )
    try:
        policy.打分(所有决策状态()[0])
    except RuntimeError as error:
        assert "缺少候选" in str(error)
    else:
        raise AssertionError("候选 logprob 缺失时必须明确报错")


def 测试文本置信度模式() -> None:
    response = 对象(choices=[对象(message=对象(content='```json\n{"scores":[60,30,10]}\n```'))])
    client = 伪造客户端([response])
    policy = OpenAI动作策略(
        base_url="http://test/v1",
        api_key="不使用",
        model="教学模型",
        score_mode="verbalized",
        client=client,
    )
    scores = policy.打分(所有决策状态()[0])
    probabilities = [math.exp(value) for value in scores]
    assert max(abs(left - right) for left, right in zip(probabilities, [0.6, 0.3, 0.1])) < 1e-10


if __name__ == "__main__":
    测试约束概率模式()
    测试普通概率模式缺项报错()
    测试文本置信度模式()
    print("JITRL_API_POLICY_TEST=PASS")
