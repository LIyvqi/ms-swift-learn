"""供 01～04 多模态课程使用的本地结果与显式过程奖励。"""

from __future__ import annotations

import re

from swift.rewards import ORM, orms

答案模式 = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
思考模式 = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
严格直接格式 = re.compile(
    r"^\s*<answer>\s*.+?\s*</answer>\s*$", re.DOTALL | re.IGNORECASE
)
严格思考格式 = re.compile(
    r"^\s*<think>\s*(?P<reason>.+?)\s*</think>\s*"
    r"<answer>\s*(?P<answer>.+?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
视觉证据词 = (
    "图",
    "图片",
    "图中",
    "图示",
    "曲线",
    "坐标",
    "表格",
    "区域",
    "箭头",
    "颜色",
    "形状",
)
视觉失败词 = ("看不到", "无法查看", "没有图片", "未提供图片", "图片缺失")


def 规范答案(text: str) -> str:
    """统一选择题字母、数值逗号、空白和常见外围标点。"""

    value = str(text).strip().upper().replace(",", "")
    value = re.sub(r"\s+", "", value)
    return value.strip("。.!！?？:：;；")


def 提取答案(text: str) -> str:
    """读取最后一个非空 answer 块。"""

    for value in reversed(答案模式.findall(text)):
        if value.strip():
            return 规范答案(value)
    return ""


def 提取思考(text: str) -> str:
    """读取最后一个非空 think 块。"""

    for value in reversed(思考模式.findall(text)):
        if value.strip():
            return value.strip()
    return ""


class 多模态答案正确奖励(ORM):
    """比较结构化最终答案，兼容 A、AC 和普通数值。"""

    def __call__(self, completions, final_answer, **kwargs) -> list[float]:
        return [
            float(
                bool(提取答案(completion)) and 提取答案(completion) == 规范答案(target)
            )
            for completion, target in zip(completions, final_answer)
        ]


class 多模态直接格式奖励(ORM):
    """直接回答只能包含一个非空 answer 块，不能泄漏 think。"""

    def __call__(self, completions, **kwargs) -> list[float]:
        return [
            float(
                bool(严格直接格式.fullmatch(completion))
                and "<think>" not in completion.lower()
            )
            for completion in completions
        ]


class 多模态思考结构奖励(ORM):
    """显式 CoT 必须先给非空、适度长度的思考，再给答案。"""

    def __call__(self, completions, **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            match = 严格思考格式.fullmatch(completion)
            reasoning = match.group("reason").strip() if match else ""
            rewards.append(float(bool(match) and 12 <= len(reasoning) <= 4000))
        return rewards


class 多模态视觉落地奖励(ORM):
    """检查视觉题是否引用可见证据，并惩罚声称无法读取图片。"""

    def __call__(self, completions, modality, **kwargs) -> list[float]:
        rewards = []
        for completion, mode in zip(completions, modality):
            reasoning = 提取思考(completion)
            if mode == "text_only":
                # 纯文本题不要求虚构视觉观察，只要求存在有效过程。
                rewards.append(float(bool(reasoning)))
                continue
            has_evidence = any(word in reasoning for word in 视觉证据词)
            reports_failure = any(word in reasoning for word in 视觉失败词)
            rewards.append(float(has_evidence and not reports_failure))
        return rewards


class 多模态过程答案一致奖励(ORM):
    """检查显式过程末段是否明确导向最终选项或数值。"""

    def __call__(self, completions, **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            answer = 提取答案(completion)
            reasoning = 提取思考(completion)
            tail = 规范答案(reasoning[-240:])
            if not answer or not reasoning:
                rewards.append(0.0)
                continue
            if re.fullmatch(r"[A-D]+", answer):
                pattern = rf"(?:答案|故选|选择|选项)[^A-D]{{0,8}}{re.escape(answer)}"
                consistent = bool(re.search(pattern, reasoning[-240:], re.IGNORECASE))
            else:
                consistent = answer in tail
            rewards.append(float(consistent))
        return rewards


orms["course_multimodal_accuracy"] = 多模态答案正确奖励
orms["course_multimodal_direct_format"] = 多模态直接格式奖励
orms["course_multimodal_cot_structure"] = 多模态思考结构奖励
orms["course_multimodal_visual_grounding"] = 多模态视觉落地奖励
orms["course_multimodal_consistency"] = 多模态过程答案一致奖励
