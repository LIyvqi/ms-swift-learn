"""实现 Agent-R1 新闻任务的状态、动作、工具反馈与奖励状态机。"""

from __future__ import annotations

import json
import re
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any

项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

知识模块 = import_module("course.25_agent_r1_news.knowledge_pipeline")
RuleKnowledgeBase = 知识模块.RuleKnowledgeBase
render_composition = 知识模块.render_composition
render_search_results = 知识模块.render_search_results
集合指标 = 知识模块.集合指标


共同系统提示 = """你是一个基于规则知识库工作的中文新闻分类智能体。
你不能直接访问知识库，只能通过环境工具获得规则。每轮先在 <think>...</think> 中进行不超过两句话的简短规划，然后只输出一个 <action>JSON</action>。

可用动作：
1. {"tool":"search_rules","arguments":{"query":"检索词","top_k":8}}
2. {"tool":"reflect","arguments":{"diagnosis":"首轮召回问题","new_query":"改写后的检索词","top_k":8}}
3. {"tool":"compose_rules","arguments":{"rule_ids":["规则ID"]}}
4. {"tool":"finish","arguments":{...}}

当前任务是 {task}。只允许使用下面这一种 finish 参数：
{finish_schema}
不要使用其他任务的 finish 字段。

不要编造工具返回，不要在 <action> 之后增加文字。"""

任务结束格式 = {
    "retrieve": '{"selected_rules":["canonical规则ID"]}',
    "compose": '{"canonical_rules":["canonical规则ID"]}',
    "decision": (
        '{"decision":"政治|财经|体育|计算机",'
        '"matched_rules":["canonical规则ID"],'
        '"evidence":["新闻原文证据"],"unmet_conditions":[],"reason":"简短理由"}'
    ),
}


def 构造系统提示(task: str) -> str:
    """只展示当前任务的 finish 格式，防止多任务字段串线。"""

    return 共同系统提示.replace("{task}", task).replace(
        "{finish_schema}", 任务结束格式[task]
    )


动作模式 = re.compile(r"<action>\s*(\{.*\})\s*</action>", re.DOTALL)
严格动作模式 = re.compile(
    r"^\s*(?:<think>\s*.+?\s*</think>\s*)?<action>\s*\{.*\}\s*</action>\s*$",
    re.DOTALL,
)


def 导入动作(text: str) -> tuple[dict[str, Any] | None, float, str]:
    """解析模型动作，同时返回协议格式分和错误信息。"""

    match = 动作模式.search(text)
    raw = match.group(1) if match else ""
    if not raw:
        fallback = re.search(r"\{.*\}", text, re.DOTALL)
        raw = fallback.group(0) if fallback else ""
    if not raw:
        return None, 0.0, "没有找到 JSON 动作"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, 0.0, f"JSON 无法解析：{error.msg}"
    if not isinstance(payload, dict):
        return None, 0.0, "动作必须是 JSON 对象"
    protocol_score = 1.0 if 严格动作模式.fullmatch(text) else 0.5
    return payload, protocol_score, ""


def 动作文本(tool: str, arguments: dict[str, Any], thought: str) -> str:
    """生成与在线环境完全一致的专家动作文本。"""

    payload = json.dumps(
        {"tool": tool, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"<think>{thought}</think><action>{payload}</action>"


class NewsPolicyEnvironment:
    """把检索、组合和决策建模为可重复执行的多轮环境。"""

    def __init__(self, knowledge: RuleKnowledgeBase, config: dict[str, Any]):
        self.knowledge = knowledge
        self.config = deepcopy(config)
        self.task = str(config.get("task", "decision"))
        if self.task not in {"retrieve", "compose", "decision"}:
            raise ValueError(f"不支持的任务：{self.task}")
        self.article = str(config["article"])
        self.label = str(config["label"])
        self.gold_rule_ids = list(config["gold_rule_ids"])
        self.gold_evidence = list(config.get("gold_evidence", []))
        self.max_steps = int(config.get("max_steps", 4))
        self.steps = 0
        self.done = False
        self.search_results: list[dict[str, Any]] = []
        self.search_count = 0
        self.reflection_count = 0
        self.composition: dict[str, Any] = {}
        self.trace: list[dict[str, Any]] = []
        self.metrics: dict[str, float] = {
            "retrieval_precision": 0.0,
            "retrieval_recall": 0.0,
            "retrieval_f1": 0.0,
            "composition_precision": 0.0,
            "composition_recall": 0.0,
            "composition_f1": 0.0,
            "decision_accuracy": 0.0,
            "rule_compliance": 0.0,
            "evidence_coverage": 0.0,
            "protocol_score": 0.0,
            "task_schema_score": 0.0,
            "reflection_gain": 0.0,
            "reflection_success": 0.0,
        }

    def reset(self) -> tuple[str, dict[str, Any], str]:
        """返回首轮观察，不向模型泄漏标签和 gold rule。"""

        if self.task == "compose":
            self.search_results = self.knowledge.search(self.article, top_k=10)
            observation = (
                "任务类型：compose。请把候选规则去重、合并为 canonical rules，处理冲突后提交。\n"
                f"新闻：{self.article}\n候选规则：{render_search_results(self.search_results)}"
            )
        elif self.task == "retrieve":
            observation = (
                "任务类型：retrieve。请检索规则并提交最相关的 canonical rule ID。\n"
                f"新闻：{self.article}"
            )
        else:
            observation = (
                "任务类型：decision。请按 Retrieve → Rerank → Compose → Execute 完成分类。\n"
                f"新闻：{self.article}"
            )
        return (
            observation,
            {"task": self.task, "record_id": self.config.get("record_id")},
            构造系统提示(self.task),
        )

    @staticmethod
    def _arguments(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        tool = str(payload.get("tool", ""))
        arguments = payload.get("arguments", {})
        return tool, arguments if isinstance(arguments, dict) else {}

    @staticmethod
    def _ids(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(str(item) for item in value if str(item).strip()))

    @staticmethod
    def _top_k(value: Any, default: int = 8) -> int:
        """容错解析模型生成的 top_k，并限制工具调用规模。"""

        try:
            return max(1, min(int(value), 20))
        except (TypeError, ValueError):
            return default

    def _retrieval_metrics(self, ids: list[str]) -> dict[str, float]:
        canonical = []
        for rule_id in ids:
            rule = self.knowledge.by_id.get(rule_id)
            canonical.append(rule.canonical_id if rule else rule_id)
        return 集合指标(canonical, self.gold_rule_ids)

    def _evidence_score(self, evidence: Any) -> float:
        if not isinstance(evidence, list):
            return 0.0
        evidence_text = " ".join(str(item) for item in evidence)
        if self.gold_evidence:
            return sum(term in evidence_text for term in self.gold_evidence) / len(
                self.gold_evidence
            )
        valid = [
            str(item)
            for item in evidence
            if len(str(item)) >= 2 and str(item) in self.article
        ]
        return min(len(valid) / 2, 1.0)

    def _finish_schema_error(self, arguments: dict[str, Any]) -> str:
        """校验当前任务唯一允许的 finish 字段与类型。"""

        if self.task == "retrieve":
            if not isinstance(arguments.get("selected_rules"), list):
                return "retrieve 必须提交 selected_rules 列表"
            return ""
        if self.task == "compose":
            if not isinstance(arguments.get("canonical_rules"), list):
                return "compose 必须提交 canonical_rules 列表"
            return ""

        required_lists = ("matched_rules", "evidence", "unmet_conditions")
        if arguments.get("decision") not in {"政治", "财经", "体育", "计算机"}:
            return "decision 必须是四个合法类别之一"
        for key in required_lists:
            if not isinstance(arguments.get(key), list):
                return f"decision 必须提交 {key} 列表"
        if not str(arguments.get("reason", "")).strip():
            return "decision 必须提交非空 reason"
        return ""

    def _finish(self, arguments: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
        if self.task == "retrieve":
            selected = self._ids(arguments.get("selected_rules"))
            available = {row["canonical_id"] for row in self.search_results}
            grounded = [rule_id for rule_id in selected if rule_id in available]
            result = 集合指标(grounded, self.gold_rule_ids)
            self.metrics.update(
                {f"retrieval_{key}": value for key, value in result.items()}
            )
            reward = 0.8 * result["f1"]
            summary = {
                "selected_rules": selected,
                "grounded_rules": grounded,
                "gold_rule_ids": self.gold_rule_ids,
            }
        elif self.task == "compose":
            selected = self._ids(arguments.get("canonical_rules"))
            available = set(self.composition.get("canonical_rule_ids", []))
            grounded = [rule_id for rule_id in selected if rule_id in available]
            result = 集合指标(grounded, self.gold_rule_ids)
            self.metrics.update(
                {f"composition_{key}": value for key, value in result.items()}
            )
            reward = 0.8 * result["f1"]
            summary = {
                "canonical_rules": selected,
                "grounded_rules": grounded,
                "gold_rule_ids": self.gold_rule_ids,
            }
        else:
            decision = str(arguments.get("decision", ""))
            matched_rules = self._ids(arguments.get("matched_rules"))
            available = set(self.composition.get("canonical_rule_ids", []))
            grounded_rules = [
                rule_id for rule_id in matched_rules if rule_id in available
            ]
            rule_result = 集合指标(grounded_rules, self.gold_rule_ids)
            accuracy = float(decision == self.label)
            evidence_coverage = self._evidence_score(arguments.get("evidence"))
            self.metrics["decision_accuracy"] = accuracy
            self.metrics["rule_compliance"] = rule_result["f1"]
            self.metrics["evidence_coverage"] = evidence_coverage
            reward = accuracy + 0.3 * rule_result["f1"] + 0.2 * evidence_coverage
            summary = {
                "decision": decision,
                "label": self.label,
                "matched_rules": matched_rules,
                "grounded_rules": grounded_rules,
                "gold_rule_ids": self.gold_rule_ids,
                "evidence": arguments.get("evidence", []),
                "unmet_conditions": arguments.get("unmet_conditions", []),
                "reason": arguments.get("reason", ""),
            }
        return "任务结束。", reward, summary

    def _record(
        self,
        event: str,
        reward: float,
        protocol_score: float,
        detail: dict[str, Any],
    ) -> dict[str, Any]:
        self.metrics["protocol_score"] = (
            self.metrics["protocol_score"] * (self.steps - 1) + protocol_score
        ) / self.steps
        self.trace.append(
            {
                "step": self.steps,
                "event": event,
                "reward": reward,
                "protocol_score": protocol_score,
                "detail": detail,
            }
        )
        return {
            "task": self.task,
            "event": event,
            "trace": deepcopy(self.trace),
            "metrics": deepcopy(self.metrics),
            "final": deepcopy(detail) if event == "finish" else {},
        }

    def step(self, completion: str) -> tuple[str, float, bool, dict[str, Any]]:
        """执行一次模型动作并返回下一观察、过程奖励和轨迹信息。"""

        if self.done:
            raise RuntimeError("环境已经结束，不能继续 step")
        self.steps += 1
        payload, protocol_score, error = 导入动作(completion)
        if payload is None:
            reward = -0.2
            self.done = self.steps >= self.max_steps
            info = self._record("invalid", reward, protocol_score, {"error": error})
            return (
                f"动作无效：{error}。请严格输出 <action>JSON</action>。",
                reward,
                self.done,
                info,
            )

        tool, arguments = self._arguments(payload)
        detail: dict[str, Any]
        if tool == "search_rules":
            query = str(arguments.get("query", "")).strip()
            top_k = self._top_k(arguments.get("top_k", 8))
            if len(query) < 2:
                reward = -0.15
                observation = "检索词过短，请从新闻主题和实体中提取有效检索词。"
                detail = {"error": "query_too_short"}
                event = "invalid_search"
            else:
                self.search_results = self.knowledge.search(query, top_k=top_k)
                self.search_count += 1
                ids = [row["rule_id"] for row in self.search_results]
                result = self._retrieval_metrics(ids)
                self.metrics.update(
                    {f"retrieval_{key}": value for key, value in result.items()}
                )
                reward = 0.15 * result["recall"]
                observation = render_search_results(self.search_results)
                detail = {"query": query, "rule_ids": ids, **result}
                event = "search_rules"
        elif tool == "reflect":
            diagnosis = str(arguments.get("diagnosis", "")).strip()
            new_query = str(arguments.get("new_query", "")).strip()
            top_k = self._top_k(arguments.get("top_k", 8))
            if not self.search_results:
                reward = -0.15
                observation = "反思前必须至少完成一次 search_rules。"
                detail = {"error": "reflect_before_search"}
                event = "invalid_reflect"
            elif len(new_query) < 2 or not diagnosis:
                reward = -0.15
                observation = "反思必须同时说明首轮召回问题，并给出有效的新检索词。"
                detail = {"error": "incomplete_reflection"}
                event = "invalid_reflect"
            else:
                old_ids = [row["rule_id"] for row in self.search_results]
                old_result = self._retrieval_metrics(old_ids)
                self.search_results = self.knowledge.search(new_query, top_k=top_k)
                self.search_count += 1
                self.reflection_count += 1
                new_ids = [row["rule_id"] for row in self.search_results]
                new_result = self._retrieval_metrics(new_ids)
                gain = new_result["f1"] - old_result["f1"]
                self.metrics.update(
                    {f"retrieval_{key}": value for key, value in new_result.items()}
                )
                self.metrics["reflection_gain"] = gain
                self.metrics["reflection_success"] = float(gain > 0)
                reward = 0.2 * max(gain, 0.0) + 0.05 * new_result["recall"]
                observation = (
                    f"反思已执行。诊断：{diagnosis}\n"
                    f"改写后的检索结果：{render_search_results(self.search_results)}"
                )
                detail = {
                    "diagnosis": diagnosis,
                    "new_query": new_query,
                    "old_rule_ids": old_ids,
                    "new_rule_ids": new_ids,
                    "old_f1": old_result["f1"],
                    "new_f1": new_result["f1"],
                    "gain": gain,
                }
                event = "reflect"
        elif tool == "compose_rules":
            rule_ids = self._ids(arguments.get("rule_ids"))
            if not rule_ids:
                reward = -0.15
                observation = "rule_ids 不能为空。"
                detail = {"error": "empty_rule_ids"}
                event = "invalid_compose"
            elif not self.search_results:
                reward = -0.2
                observation = "组合前必须先检索，不能凭空编造规则 ID。"
                detail = {"error": "compose_before_search"}
                event = "invalid_compose"
            else:
                available_ids = {row["rule_id"] for row in self.search_results}
                invented_ids = [
                    rule_id for rule_id in rule_ids if rule_id not in available_ids
                ]
                if invented_ids:
                    reward = -0.2
                    observation = "只能组合本轮检索实际返回的物理 rule_id。"
                    detail = {"error": "invented_rule_ids", "rule_ids": invented_ids}
                    event = "invalid_compose"
                else:
                    self.composition = self.knowledge.compose(rule_ids)
                    result = 集合指标(
                        self.composition["canonical_rule_ids"], self.gold_rule_ids
                    )
                    self.metrics.update(
                        {f"composition_{key}": value for key, value in result.items()}
                    )
                    reward = 0.2 * result["f1"]
                    observation = render_composition(self.composition)
                    detail = {
                        "input_rule_ids": rule_ids,
                        "canonical_rule_ids": self.composition["canonical_rule_ids"],
                        **result,
                    }
                    event = "compose_rules"
        elif tool == "finish":
            missing_stage = (self.task == "retrieve" and not self.search_results) or (
                self.task in {"compose", "decision"} and not self.composition
            )
            schema_error = self._finish_schema_error(arguments)
            if missing_stage:
                reward = -0.2
                observation = "不能跳过必要阶段：retrieve 先检索，compose/decision 先生成组合规则。"
                detail = {"error": "required_stage_missing", "task": self.task}
                event = "invalid_finish"
            elif schema_error:
                protocol_score = 0.0
                reward = -0.35
                observation = f"finish 参数与当前任务不匹配：{schema_error}。"
                detail = {
                    "error": "finish_schema_mismatch",
                    "task": self.task,
                    "message": schema_error,
                }
                event = "invalid_finish_schema"
            else:
                self.metrics["task_schema_score"] = 1.0
                observation, reward, detail = self._finish(arguments)
                reward += 0.1 * protocol_score
                event = "finish"
                self.done = True
        else:
            reward = -0.2
            observation = f"未知工具：{tool}。只能使用 search_rules、reflect、compose_rules 或 finish。"
            detail = {"error": "unknown_tool", "tool": tool}
            event = "invalid_tool"

        if not self.done and self.steps >= self.max_steps:
            reward -= 0.25
            self.done = True
            observation += "\n已达到最大步数，轨迹结束。"
            detail["stop_reason"] = "max_steps"
        info = self._record(event, reward, protocol_score, detail)
        return observation, reward, self.done, info

    def expert_action(self) -> str:
        """根据当前环境状态生成下一条确定性专家动作。"""

        query_terms = self.gold_evidence[:3]
        root_rule = self.knowledge.by_id.get(self.gold_rule_ids[0])
        if not query_terms:
            query_terms = [
                self.label,
                *(list(root_rule.keywords[:3]) if root_rule else []),
            ]
        # 单字证据不满足环境的最短查询约束；补入类别和 ROOT 关键词，避免专家反复提交非法反思。
        elif len("".join(query_terms).strip()) < 2:
            query_terms = [
                *query_terms,
                self.label,
                *(list(root_rule.keywords[:3]) if root_rule else []),
            ]
        focused_query = " ".join(query_terms)
        broad_query = re.sub(r"\s+", " ", self.article[:72]).strip()
        relevant_candidate_ids = [
            row["rule_id"]
            for row in self.search_results
            if row["canonical_id"] in self.gold_rule_ids
        ]
        grounded_retrieved_ids = list(
            dict.fromkeys(
                row["canonical_id"]
                for row in self.search_results
                if row["canonical_id"] in self.gold_rule_ids
            )
        )
        grounded_composed_ids = [
            rule_id
            for rule_id in self.composition.get("canonical_rule_ids", [])
            if rule_id in self.gold_rule_ids
        ]
        if self.task == "retrieve":
            if not self.search_results:
                return 动作文本(
                    "search_rules",
                    {"query": broad_query, "top_k": 8},
                    "先根据标题附近的信息完成一轮宽召回。",
                )
            if self.reflection_count == 0:
                return 动作文本(
                    "reflect",
                    {
                        "diagnosis": "首轮查询混入来源元数据，候选类别较分散，需要围绕正文证据改写。",
                        "new_query": focused_query,
                        "top_k": 8,
                    },
                    "检查首轮候选后，改用正文中的主题证据重新检索。",
                )
            return 动作文本(
                "finish",
                {"selected_rules": grounded_retrieved_ids},
                "提交去重后的相关 canonical 规则。",
            )
        if self.task == "compose":
            if self.reflection_count == 0:
                return 动作文本(
                    "reflect",
                    {
                        "diagnosis": "初始候选来自整篇长文，可能混入背景规则，需要用核心证据收紧候选。",
                        "new_query": focused_query,
                        "top_k": 8,
                    },
                    "先检查初始候选，再围绕正文核心证据重新检索。",
                )
            if not self.composition:
                return 动作文本(
                    "compose_rules",
                    {"rule_ids": relevant_candidate_ids},
                    "先过滤跨类噪声，再合并相关规则的重复版本。",
                )
            return 动作文本(
                "finish",
                {"canonical_rules": grounded_composed_ids},
                "提交与新闻主焦点匹配的 canonical 规则。",
            )
        if not self.search_results:
            return 动作文本(
                "search_rules",
                {"query": broad_query, "top_k": 8},
                "先做宽召回，观察候选类别和规则版本。",
            )
        if self.reflection_count == 0:
            return 动作文本(
                "reflect",
                {
                    "diagnosis": "首轮宽召回可能受元数据或背景词干扰，需要聚焦正文主题证据。",
                    "new_query": focused_query,
                    "top_k": 8,
                },
                "根据首轮候选反思检索偏差并改写查询。",
            )
        if not self.composition:
            return 动作文本(
                "compose_rules",
                {"rule_ids": relevant_candidate_ids},
                "过滤跨类噪声，再对相关候选去重并绑定例外。",
            )
        return 动作文本(
            "finish",
            {
                "decision": self.label,
                "matched_rules": grounded_composed_ids,
                "evidence": self.gold_evidence,
                "unmet_conditions": [],
                "reason": f"新闻主焦点符合{self.label}类规则。",
            },
            "依据组合后的规则给出结构化结论。",
        )


def default_knowledge_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "datasets/agent_r1_news/knowledge_rules.jsonl"
    )
