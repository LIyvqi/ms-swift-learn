#!/usr/bin/env python3
"""比较无知识、全规则、BM25、单轮 Memory 与结构化多轮 Memory。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from inference_backend import 创建后端
from memo_core import (
    BM25索引,
    Executive消息,
    冲突记忆问题,
    单轮记忆问题,
    确定性执行,
    汇总审核,
    集合指标,
    写_jsonl,
    规则上下文,
    解析审核,
    解析记忆,
    读_jsonl,
    记忆消息,
    规范化记忆编号,
    确认记忆问题,
    提取审核线索列表,
)


项目根目录 = Path(__file__).resolve().parents[2]
默认方法 = "no_memory,all_rules,bm25,memo_single,memo_structured,memo_structured_deterministic,oracle_rules"


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MeMo 新闻内容审核完整对比实验")
    parser.add_argument("--memory-model", required=True, help="训练后的完整 Memory 检查点或 API 模型名")
    parser.add_argument("--executive-model", default=str(项目根目录 / "models/Qwen3.5-0.8B-Base"))
    parser.add_argument("--memory-backend", choices=("local", "api"), default="local")
    parser.add_argument("--executive-backend", choices=("local", "api"), default="local")
    parser.add_argument("--memory-base-url")
    parser.add_argument("--executive-base-url")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--methods", default=默认方法)
    parser.add_argument("--maximum-samples", type=int, default=0, help="0 表示全部 120 条")
    parser.add_argument("--bm25-top-k", type=int, default=5)
    parser.add_argument(
        "--memory-grounding", choices=("span", "full"), default="span",
        help="span 先取发布片段；full 把完整新闻交给 Memory，仅建议用于干扰消融",
    )
    parser.add_argument(
        "--id-resolution", choices=("registry", "none"), default="registry",
        help="registry 用不含规则正文的稳定 ID 注册表修正生成式序号；none 保留原始编号",
    )
    parser.add_argument("--rules", type=Path, default=项目根目录 / "datasets/memo_rule_memory/rules.jsonl")
    parser.add_argument("--cases", type=Path, default=项目根目录 / "datasets/memo_rule_memory/audit_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=项目根目录 / "outputs/26_memo_rule_memory/audit_evaluation")
    return parser.parse_args()


def 批量记忆(memory_backend, questions: list[str]) -> tuple[list[str], list[dict[str, Any]], float]:
    """批量调用 Memory 并解析结果。"""

    started = time.perf_counter()
    raw = memory_backend.生成([记忆消息(question) for question in questions], max_tokens=512)
    elapsed = time.perf_counter() - started
    return raw, [解析记忆(text) for text in raw], elapsed


def 记忆上下文(memories: list[dict[str, Any]]) -> str:
    """保留规则编号、处置、事实和例外，供 Executive 引用而不暴露原始规则库。"""

    blocks = []
    for memory in memories:
        blocks.append(json.dumps({
            "rule_ids": memory.get("rule_ids", []),
            "decision": memory.get("decision", ""),
            "facts": memory.get("facts", []),
            "exceptions": memory.get("exceptions", []),
            "priority": memory.get("priority", 0),
        }, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(blocks)


def 最终Executive(executive_backend, cases: list[dict[str, Any]], contexts: list[str], method: str) -> tuple[list[str], list[dict[str, Any]], float, list[int]]:
    """让冻结 Executive 基于不同政策上下文做最终决策。"""

    messages = [Executive消息(case, context, method) for case, context in zip(cases, contexts)]
    started = time.perf_counter()
    raw = executive_backend.生成(messages, max_tokens=384)
    elapsed = time.perf_counter() - started
    return raw, [解析审核(text) for text in raw], elapsed, [sum(len(message["content"]) for message in item) for item in messages]


def 基础上下文(method: str, cases: list[dict[str, Any]], rules: list[dict[str, Any]], index: BM25索引, top_k: int) -> list[str]:
    """构造不使用 Memory 模型的三种基线上下文。"""

    if method == "no_memory":
        return [""] * len(cases)
    if method == "all_rules":
        context = 规则上下文(rules, compact=True)
        return [context] * len(cases)
    if method == "bm25":
        return [规则上下文([pair[0] for pair in index.检索(case["content"], top_k=top_k, category=case["category"])]) for case in cases]
    if method == "oracle_rules":
        by_id = {rule["rule_id"]: rule for rule in rules}
        return [规则上下文([by_id[rule_id] for rule_id in case["gold_rule_ids"]]) for case in cases]
    raise ValueError(method)


def 保存结果(output_dir: Path, method: str, cases: list[dict[str, Any]], predictions: list[dict[str, Any]], raw: list[str], contexts: list[str], prompt_chars: list[int], elapsed: float, extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """保存逐条轨迹和汇总。"""

    traces = []
    for index, (case, prediction, response, context, chars) in enumerate(zip(cases, predictions, raw, contexts, prompt_chars)):
        traces.append({
            **case,
            "method": method,
            "prediction": prediction,
            "response": response,
            "policy_context": context,
            "prompt_chars": chars,
            "elapsed_seconds": elapsed / len(cases),
            "memory_trace": extra[index] if extra else None,
        })
    summary = {"method": method, **汇总审核(traces)}
    if extra:
        memory_scores = []
        raw_memory_scores = []
        memory_valid = []
        for case, item in zip(cases, extra):
            memories = item.get("stages") or [item.get("stage1", {})]
            recalled = list(dict.fromkeys(
                rule_id
                for memory in memories
                for rule_id in memory.get("rule_ids", [])
            ))
            memory_scores.append(集合指标(recalled, case["gold_rule_ids"]))
            raw_recalled = list(dict.fromkeys(
                rule_id
                for memory in memories
                for rule_id in memory.get("raw_rule_ids", memory.get("rule_ids", []))
            ))
            raw_memory_scores.append(集合指标(raw_recalled, case["gold_rule_ids"]))
            memory_valid.extend(bool(memory.get("valid")) for memory in memories)
        summary.update({
            "memory_rule_precision": sum(score[0] for score in memory_scores) / len(memory_scores),
            "memory_rule_recall": sum(score[1] for score in memory_scores) / len(memory_scores),
            "memory_rule_f1": sum(score[2] for score in memory_scores) / len(memory_scores),
            "memory_raw_rule_precision": sum(score[0] for score in raw_memory_scores) / len(raw_memory_scores),
            "memory_raw_rule_recall": sum(score[1] for score in raw_memory_scores) / len(raw_memory_scores),
            "memory_raw_rule_f1": sum(score[2] for score in raw_memory_scores) / len(raw_memory_scores),
            "memory_format_rate": sum(memory_valid) / len(memory_valid),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    写_jsonl(output_dir / f"{method}.jsonl", traces)
    (output_dir / f"{method}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def 主程序() -> None:
    args = 解析参数()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    known = set(默认方法.split(","))
    if not methods or not set(methods) <= known:
        raise ValueError(f"methods 只能来自：{默认方法}")
    rules = 读_jsonl(args.rules)
    cases = 读_jsonl(args.cases)
    if args.maximum_samples:
        cases = cases[:args.maximum_samples]
    index = BM25索引(rules)
    rules_by_id = {rule["rule_id"]: rule for rule in rules}

    def 规范化批次(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按案例顺序应用可关闭的 ID 注册表解析。"""

        if args.id_resolution == "none":
            return [{**memory, "raw_rule_ids": list(memory.get("rule_ids", []))} for memory in parsed]
        return [
            规范化记忆编号(memory, rules_by_id, case["category"])
            for case, memory in zip(cases, parsed)
        ]

    def 分片批量记忆() -> tuple[list[list[str]], list[list[dict[str, Any]]], float]:
        """按案例拆分多个审核线索，批量回忆后再恢复案例分组。"""

        owners: list[int] = []
        questions: list[str] = []
        for case_index, case in enumerate(cases):
            for clue in 提取审核线索列表(case["content"], args.memory_grounding):
                clue_case = {**case, "content": clue}
                owners.append(case_index)
                questions.append(单轮记忆问题(clue_case, "full"))
        raw, parsed, elapsed = 批量记忆(memory_backend, questions)
        grouped_raw: list[list[str]] = [[] for _ in cases]
        grouped_memory: list[list[dict[str, Any]]] = [[] for _ in cases]
        for owner, response, memory in zip(owners, raw, parsed):
            grouped_raw[owner].append(response)
            if args.id_resolution == "none":
                normalized = {**memory, "raw_rule_ids": list(memory.get("rule_ids", []))}
            else:
                normalized = 规范化记忆编号(memory, rules_by_id, cases[owner]["category"])
            grouped_memory[owner].append(normalized)
        return grouped_raw, grouped_memory, elapsed

    need_memory = any(method.startswith("memo_") for method in methods)
    need_executive = any(method != "memo_structured_deterministic" for method in methods)
    memory_backend = 创建后端(
        args.memory_backend, args.memory_model, batch_size=args.batch_size,
        base_url=args.memory_base_url, api_key_env=args.api_key_env, concurrency=args.api_concurrency,
    ) if need_memory else None
    executive_backend = 创建后端(
        args.executive_backend, args.executive_model, batch_size=args.batch_size,
        base_url=args.executive_base_url, api_key_env=args.api_key_env, concurrency=args.api_concurrency,
    ) if need_executive else None

    summaries = []
    structured_cache: tuple[list[list[dict[str, Any]]], list[dict[str, Any]], list[str], float] | None = None
    for method in methods:
        if method in {"no_memory", "all_rules", "bm25", "oracle_rules"}:
            contexts = 基础上下文(method, cases, rules, index, args.bm25_top_k)
            raw, predictions, elapsed, prompt_chars = 最终Executive(executive_backend, cases, contexts, method)
            summaries.append(保存结果(args.output_dir, method, cases, predictions, raw, contexts, prompt_chars, elapsed))
            continue

        if method == "memo_single":
            memory_raw, memories, memory_elapsed = 批量记忆(
                memory_backend,
                [单轮记忆问题(case, args.memory_grounding) for case in cases],
            )
            memories = 规范化批次(memories)
            contexts = [记忆上下文([memory]) for memory in memories]
            raw, predictions, executive_elapsed, prompt_chars = 最终Executive(executive_backend, cases, contexts, method)
            extra = [{"stage1_raw": text, "stage1": memory} for text, memory in zip(memory_raw, memories)]
            summaries.append(保存结果(args.output_dir, method, cases, predictions, raw, contexts, prompt_chars, memory_elapsed + executive_elapsed, extra))
            continue

        if structured_cache is None:
            raw1_grouped, stage1_grouped, elapsed1 = 分片批量记忆()
            # 后续确认问题需要同时看到每个片段回忆出的所有候选，而不是只看第一条。
            stage1 = [
                {
                    "rule_ids": list(dict.fromkeys(
                        rule_id for memory in memories for rule_id in memory.get("rule_ids", [])
                    )),
                    "raw_rule_ids": list(dict.fromkeys(
                        rule_id for memory in memories for rule_id in memory.get("raw_rule_ids", [])
                    )),
                    "decision": max(
                        (memory.get("decision", "") for memory in memories),
                        key=lambda value: {"": -1, "PASS": 0, "REVIEW": 1, "REJECT": 2}.get(value, -1),
                        default="",
                    ),
                    "facts": [fact for memory in memories for fact in memory.get("facts", [])],
                    "exceptions": [item for memory in memories for item in memory.get("exceptions", [])],
                    "priority": max((memory.get("priority", 0) for memory in memories), default=0),
                    "valid": all(memory.get("valid", False) for memory in memories),
                    "fragment_memories": memories,
                }
                for memories in stage1_grouped
            ]
            raw2, stage2, elapsed2 = 批量记忆(
                memory_backend,
                [确认记忆问题(case, memory, args.memory_grounding) for case, memory in zip(cases, stage1)],
            )
            stage2 = 规范化批次(stage2)
            combined = [[first, second] for first, second in zip(stage1, stage2)]
            raw3, stage3, elapsed3 = 批量记忆(
                memory_backend,
                [冲突记忆问题(case, memories, args.memory_grounding) for case, memories in zip(cases, combined)],
            )
            stage3 = 规范化批次(stage3)
            all_memories = [memories + [third] for memories, third in zip(combined, stage3)]
            contexts = [记忆上下文(memories) for memories in all_memories]
            trace = [
                {"stage1_raw": a, "stage2_raw": b, "stage3_raw": c, "stages": memories}
                for a, b, c, memories in zip(raw1_grouped, raw2, raw3, all_memories)
            ]
            structured_cache = (all_memories, trace, contexts, elapsed1 + elapsed2 + elapsed3)

        all_memories, trace, contexts, memory_elapsed = structured_cache
        if method == "memo_structured_deterministic":
            predictions = [确定性执行(memories, rules_by_id, case["content"]) for case, memories in zip(cases, all_memories)]
            raw = [f"<audit>{json.dumps(prediction, ensure_ascii=False, separators=(',', ':'))}</audit>" for prediction in predictions]
            prompt_chars = [len(context) + len(case["content"]) for case, context in zip(cases, contexts)]
            summaries.append(保存结果(args.output_dir, method, cases, predictions, raw, contexts, prompt_chars, memory_elapsed, trace))
        else:
            raw, predictions, executive_elapsed, prompt_chars = 最终Executive(executive_backend, cases, contexts, method)
            summaries.append(保存结果(args.output_dir, method, cases, predictions, raw, contexts, prompt_chars, memory_elapsed + executive_elapsed, trace))

    (args.output_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    主程序()
