"""供 CoT-RLOO 新闻分类课程使用的结果奖励与过程代理奖励。"""

from __future__ import annotations

from typing import List

from swift.rewards import ORM, orms

try:
    # 从仓库根目录导入时使用完整包路径。
    from course.plugins.cot_classification_common import 解析回答, 证据分隔符
except ModuleNotFoundError:
    # ms-swift 动态加载单个插件文件时，使用同目录模块名。
    from cot_classification_common import 解析回答, 证据分隔符


class CoTLabelAccuracy(ORM):
    """只检查最终框选标签是否正确，这是任务的主要结果奖励。"""

    def __call__(self, completions, label, **kwargs) -> List[float]:
        return [
            float(解析回答(completion).标签 == expected.strip())
            for completion, expected in zip(completions, label)
        ]


class CoTStructure(ORM):
    """奖励严格格式和适中的非空推理长度，避免空 CoT 与整篇复制。"""

    def __call__(self, completions, **kwargs) -> List[float]:
        scores = []
        for completion in completions:
            parsed = 解析回答(completion)
            scores.append(float(parsed.严格格式 and 15 <= len(parsed.推理) <= 220))
        return scores


class CoTEvidenceCoverage(ORM):
    """计算人工标注证据词在思考块中的覆盖比例，不检查最终答案区。"""

    def __call__(self, completions, evidence_terms, **kwargs) -> List[float]:
        scores = []
        for completion, packed_terms in zip(completions, evidence_terms):
            reason = 解析回答(completion).推理
            terms = [term.strip() for term in packed_terms.split(证据分隔符) if term.strip()]
            scores.append(sum(term in reason for term in terms) / len(terms) if terms else 0.0)
        return scores


class CoTConsistency(ORM):
    """检查思考块是否明确提到最终标签，只衡量自洽性而非事实正确性。"""

    def __call__(self, completions, **kwargs) -> List[float]:
        scores = []
        for completion in completions:
            parsed = 解析回答(completion)
            scores.append(float(bool(parsed.标签) and parsed.标签 in parsed.推理))
        return scores


orms["course_cot_label_accuracy"] = CoTLabelAccuracy
orms["course_cot_structure"] = CoTStructure
orms["course_cot_evidence"] = CoTEvidenceCoverage
orms["course_cot_consistency"] = CoTConsistency
