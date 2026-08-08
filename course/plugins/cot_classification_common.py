"""CoT 新闻分类奖励与离线评分器共用的纯文本解析逻辑。"""

from __future__ import annotations

import re
from dataclasses import dataclass


允许标签 = ("政治", "财经", "体育", "计算机")
标签模式 = "|".join(map(re.escape, 允许标签))
证据分隔符 = "|||"

# Qwen3.5 的模板可能在模型正文前附加一个空思考块，因此严格格式也兼容该前缀。
严格模式 = re.compile(
    rf"^\s*(?:<think>\s*</think>\s*)?<think>\s*(?P<reason>.+?)\s*</think>"
    rf"\s*\\boxed\{{(?P<label>{标签模式})\}}\s*$",
    re.DOTALL,
)
思考模式 = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
框选模式 = re.compile(rf"\\boxed\{{({标签模式})\}}")


@dataclass(frozen=True)
class 解析结果:
    """保存生成文本中的推理、最终标签和严格格式状态。"""

    推理: str
    标签: str
    严格格式: bool


def 解析回答(text: str) -> 解析结果:
    """解析最后一个非空思考块和最后一个合法框选标签。"""
    strict = 严格模式.fullmatch(text)
    if strict:
        return 解析结果(strict.group("reason").strip(), strict.group("label"), True)

    reasons = [item.strip() for item in 思考模式.findall(text) if item.strip()]
    labels = 框选模式.findall(text)
    return 解析结果(reasons[-1] if reasons else "", labels[-1] if labels else "", False)
