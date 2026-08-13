#!/usr/bin/env python3
"""运行 CA-MeMo 七组真实对照实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ca_memo_core import (
    BM25索引,
    大模型裁判消息,
    主动搜索问题,
    主召回问题,
    共形处置集合,
    决策等级,
    固定冲突问题,
    固定确认问题,
    处置概率,
    拟合共形阈值,
    拟合权威检索阈值,
    权威检索候选,
    独立权威验证,
    独立检查问题,
    确定性执行,
    硬验证,
    第三投票问题,
    聚合记忆,
    调用统计,
    逻辑校准器,
    解析并规范化,
    选择路由阈值,
    提取可靠性特征,
    汇总方法,
    解析大模型裁判,
    验证大模型裁判,
    首轮是否完全正确,
)
from inference_backend import 创建后端, 生成记录, 校验官方Swift


项目根目录 = Path(__file__).resolve().parents[2]
默认方法 = (
    "memory_single,fixed_three_stage,simple_vote,calibrated_route,"
    "calibrated_search,calibrated_search_verifier,all_authority_rules"
)


def 解析参数() -> argparse.Namespace:
    """定义本地与 API 黑盒实验参数。"""

    parser = argparse.ArgumentParser(description="CA-MeMo 校准、主动搜索与独立验证完整实验")
    parser.add_argument("--memory-model", required=True, help="第26课训练后的 Memory 检查点或 API 模型名")
    parser.add_argument("--memory-backend", choices=("local", "api"), default="local")
    parser.add_argument("--memory-base-url")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-concurrency", type=int, default=8)
    parser.add_argument("--judge-backend", choices=("none", "api"), default="none")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--request-logprobs", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    parser.add_argument("--methods", default=默认方法)
    parser.add_argument("--maximum-calibration-samples", type=int, default=0)
    parser.add_argument("--maximum-test-samples", type=int, default=0)
    parser.add_argument("--conformal-alpha", type=float, default=0.1)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument(
        "--rules", type=Path, default=项目根目录 / "datasets/memo_rule_memory/rules.jsonl"
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=项目根目录 / "datasets/calibrated_adaptive_memo/calibration.jsonl",
    )
    parser.add_argument(
        "--test", type=Path, default=项目根目录 / "datasets/calibrated_adaptive_memo/test.jsonl"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=项目根目录 / "outputs/27_calibrated_adaptive_memo/full_experiment",
    )
    return parser.parse_args()


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 载入缓存(path: Path, expected: int) -> list[生成记录] | None:
    """只接受行数完整的生成缓存。"""

    if not path.exists():
        return None
    records = [生成记录.从字典(json.loads(line)) for line in path.open(encoding="utf-8") if line.strip()]
    return records if len(records) == expected else None


def 保存缓存(path: Path, records: list[生成记录]) -> None:
    """原子性写入生成缓存，避免中断后把半文件当完整结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    写_jsonl(temporary, [record.转字典() for record in records])
    temporary.replace(path)


class 推理缓存:
    """让不同对照共享同一批真实生成，同时保持资源统计按实际路径切片。"""

    def __init__(self, backend, root: Path, force: bool, identity: dict[str, Any]):
        self.backend = backend
        self.root = root
        self.force = force
        self.root.mkdir(parents=True, exist_ok=True)
        identity_path = self.root / "identity.json"
        if identity_path.exists():
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
            # 兼容课程早期没有记录价格的零价格缓存；非零价格变化仍会拒绝误用旧成本。
            existing.setdefault("input_price_per_million", 0.0)
            existing.setdefault("output_price_per_million", 0.0)
            if existing != identity and not force:
                raise RuntimeError(
                    "生成缓存属于另一后端或模型；请更换 --output-dir，或确认后使用 --force-regenerate"
                )
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def 调用(self, name: str, questions: list[str], max_tokens: int = 512) -> list[生成记录]:
        """按稳定名称读取或生成一组 Memory 响应。"""

        from ca_memo_core import 记忆消息

        return self.调用消息(name, [记忆消息(question) for question in questions], max_tokens)

    def 调用消息(
        self,
        name: str,
        messages_batch: list[list[dict[str, str]]],
        max_tokens: int,
        backend=None,
    ) -> list[生成记录]:
        """对任意消息批次做带输入指纹的缓存，API 裁判也复用此逻辑。"""

        path = self.root / f"{name}.jsonl"
        meta_path = self.root / f"{name}.meta.json"
        fingerprint = hashlib.sha256(
            json.dumps(
                {"messages": messages_batch, "max_tokens": max_tokens},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        metadata = {"sha256": fingerprint, "count": len(messages_batch), "max_tokens": max_tokens}
        meta_matches = (
            meta_path.exists()
            and json.loads(meta_path.read_text(encoding="utf-8")) == metadata
        )
        cached = None if self.force or not meta_matches else 载入缓存(path, len(messages_batch))
        if cached is not None:
            return cached

        records = (backend or self.backend).生成(messages_batch, max_tokens=max_tokens)
        保存缓存(path, records)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return records


def 记忆批次(
    cases: list[dict[str, Any]],
    records: list[生成记录],
    rules_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """按案例解析与规范化一批生成。"""

    return [
        解析并规范化(record, rules_by_id, case["category"])
        for case, record in zip(cases, records)
    ]


def 结果对象(decision: str, rule_ids: list[str], reason: str) -> dict[str, Any]:
    """构造统一预测对象。"""

    return {
        "decision": decision,
        "matched_rules": list(dict.fromkeys(rule_ids)),
        "evidence": [],
        "reason": reason,
        "valid": True,
        "wrapper_valid": True,
    }


def 安全拒答(reason: str) -> dict[str, Any]:
    """统一以 REVIEW 表示需要人工处理。"""

    return 结果对象("REVIEW", [], reason)


def 候选是否含金标准(candidates: list[dict[str, Any]], gold: list[str]) -> bool:
    """Pick@N 的候选覆盖定义：至少有一个候选完整含有金规则集合。"""

    target = set(gold)
    return bool(target) and any(target <= set(memory.get("rule_ids", [])) for memory in candidates)


def 主程序() -> None:
    """运行校准、路由、主动搜索、独立验证和七组对照。"""

    args = 解析参数()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    known = set(默认方法.split(","))
    if not methods or not set(methods) <= known:
        raise ValueError(f"methods 只能来自：{默认方法}")

    rules = 读_jsonl(args.rules)
    calibration = 读_jsonl(args.calibration)
    test = 读_jsonl(args.test)
    if args.maximum_calibration_samples:
        calibration = calibration[:args.maximum_calibration_samples]
    if args.maximum_test_samples:
        test = test[:args.maximum_test_samples]
    cases = calibration + test
    split_at = len(calibration)
    rules_by_id = {rule["rule_id"]: rule for rule in rules}
    authority_index = BM25索引(rules)

    request_logprobs = {"auto": None, "true": True, "false": False}[args.request_logprobs]
    backend = 创建后端(
        args.memory_backend,
        args.memory_model,
        batch_size=args.batch_size,
        base_url=args.memory_base_url,
        api_key_env=args.api_key_env,
        concurrency=args.api_concurrency,
        request_logprobs=request_logprobs,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    cache_identity = {
        "memory_backend": args.memory_backend,
        "memory_model": args.memory_model,
        "memory_base_url": args.memory_base_url,
        "request_logprobs": args.request_logprobs,
        "batch_size": args.batch_size,
        "judge_backend": args.judge_backend,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "input_price_per_million": args.input_price_per_million,
        "output_price_per_million": args.output_price_per_million,
    }
    cache = 推理缓存(
        backend, args.output_dir / "generation_cache", args.force_regenerate, cache_identity
    )
    judge_backend = None
    if args.judge_backend == "api":
        if not args.judge_model:
            raise ValueError("启用 API 裁判时必须提供 --judge-model")
        judge_backend = 创建后端(
            "api",
            args.judge_model,
            base_url=args.judge_base_url,
            api_key_env=args.api_key_env,
            concurrency=args.api_concurrency,
            request_logprobs=False,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
        )

    # 两个独立快速视角是所有校准方法的共同输入。白盒模式额外使用 logprob，黑盒模式自动缺省。
    primary_records = cache.调用("all_primary", [主召回问题(case) for case in cases])
    check_records = cache.调用("all_independent_check", [独立检查问题(case) for case in cases])
    primary = 记忆批次(cases, primary_records, rules_by_id)
    checks = 记忆批次(cases, check_records, rules_by_id)

    feature_rows = [
        提取可靠性特征([first_record, check_record], [first, check], rules_by_id)
        for first_record, check_record, first, check in zip(primary_records, check_records, primary, checks)
    ]
    calibration_labels = [
        int(首轮是否完全正确(case, memory, rules_by_id))
        for case, memory in zip(calibration, primary[:split_at])
    ]
    calibrator = 逻辑校准器.拟合(feature_rows[:split_at], calibration_labels)
    probabilities = [calibrator.预测(features) for features in feature_rows]
    thresholds = 选择路由阈值(probabilities[:split_at], calibration_labels)

    decision_probabilities = [处置概率([first, check]) for first, check in zip(primary, checks)]
    conformal_threshold = 拟合共形阈值(
        decision_probabilities[:split_at],
        [case["gold_decision"] for case in calibration],
        alpha=args.conformal_alpha,
    )
    authority_threshold = 拟合权威检索阈值(calibration, authority_index)

    # 固定协议、简单投票和主动搜索的附加请求统一批量生成。不同方法只统计自己使用的响应。
    fixed_stage2_records = cache.调用(
        "all_fixed_stage2",
        [固定确认问题(case, first) for case, first in zip(cases, primary)],
    )
    fixed_stage2 = 记忆批次(cases, fixed_stage2_records, rules_by_id)
    fixed_stage3_records = cache.调用(
        "all_fixed_stage3",
        [固定冲突问题(case, [first, second]) for case, first, second in zip(cases, primary, fixed_stage2)],
    )
    fixed_stage3 = 记忆批次(cases, fixed_stage3_records, rules_by_id)
    vote3_records = cache.调用("all_vote3", [第三投票问题(case) for case in cases])
    vote3 = 记忆批次(cases, vote3_records, rules_by_id)

    # 主动分支只对中置信样本生成。索引映射确保结果可恢复并能回填原顺序。
    # 只要不能通过完整“高置信直通门”，且没有低到直接拒答，就必须进入慢路径。
    # 这包含“校准概率高但共形集合不唯一/硬验证失败”的案例，不能漏生成搜索分支。
    medium_indices = []
    for index, probability in enumerate(probabilities):
        conformal_set = 共形处置集合(decision_probabilities[index], conformal_threshold)
        hard = 硬验证(primary[index], rules_by_id, cases[index]["category"])
        high_direct = probability >= thresholds["high"] and hard["passed"] and len(conformal_set) == 1
        if probability >= thresholds["low"] and not high_direct:
            medium_indices.append(index)
    search_questions = []
    search_owner = []
    for index in medium_indices:
        for branch_index, question in enumerate(主动搜索问题(cases[index], primary[index])):
            search_owner.append((index, branch_index))
            search_questions.append(question)
    search_records_flat = cache.调用("medium_active_search", search_questions) if search_questions else []
    search_memories_flat = [
        解析并规范化(record, rules_by_id, cases[owner]["category"])
        for record, (owner, _) in zip(search_records_flat, search_owner)
    ]
    search_records: dict[int, list[生成记录]] = {index: [] for index in medium_indices}
    search_memories: dict[int, list[dict[str, Any]]] = {index: [] for index in medium_indices}
    for record, memory, (owner, _) in zip(search_records_flat, search_memories_flat, search_owner):
        search_records[owner].append(record)
        search_memories[owner].append(memory)

    # API 裁判是可选的独立模型，只查看中置信案例、聚合候选和权威规则片段。
    judge_records: dict[int, 生成记录] = {}
    judge_results: dict[int, dict[str, Any]] = {}
    judge_allowed_ids: dict[int, list[str]] = {}
    if judge_backend is not None and medium_indices:
        judge_messages = []
        for index in medium_indices:
            aggregate = 聚合记忆([primary[index], checks[index], *search_memories.get(index, [])], minimum_votes=2)
            authority_ids, _ = 权威检索候选(cases[index], authority_index, authority_threshold)
            candidate_ids = list(dict.fromkeys([*aggregate.get("rule_ids", []), *authority_ids]))
            judge_allowed_ids[index] = candidate_ids
            judge_messages.append(大模型裁判消息(cases[index], candidate_ids, rules_by_id))
        cached_judge = cache.调用消息(
            "medium_api_judge", judge_messages, max_tokens=256, backend=judge_backend
        )
        for index, record in zip(medium_indices, cached_judge):
            judge_records[index] = record
            judge_results[index] = 解析大模型裁判(record.text)

    config = {
        "memory_model": args.memory_model,
        "memory_backend": args.memory_backend,
        "ms_swift":校验官方Swift() if args.memory_backend == "local" else None,
        "calibration_samples": len(calibration),
        "test_samples": len(test),
        "calibration_positive_rate": sum(calibration_labels) / len(calibration_labels),
        "route_thresholds": thresholds,
        "conformal_alpha": args.conformal_alpha,
        "conformal_threshold": conformal_threshold,
        "authority_bm25_threshold": authority_threshold,
        "medium_search_cases_all_splits": len(medium_indices),
        "request_logprobs": args.request_logprobs,
        "batch_size": args.batch_size,
        "api_concurrency": args.api_concurrency,
        "judge_backend": args.judge_backend,
        "judge_model": args.judge_model,
        "input_price_per_million": args.input_price_per_million,
        "output_price_per_million": args.output_price_per_million,
        "calibrator": calibrator.转字典(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summaries = []
    for method in methods:
        traces = []
        for local_index, case in enumerate(test):
            index = split_at + local_index
            first = primary[index]
            check = checks[index]
            first_record = primary_records[index]
            check_record = check_records[index]
            probability = probabilities[index]
            conformal_set = 共形处置集合(decision_probabilities[index], conformal_threshold)
            base_prediction = 确定性执行([first], rules_by_id, case["audit_span"])
            base_correct = (
                base_prediction["decision"] == case["gold_decision"]
                and set(first.get("rule_ids", [])) == set(case["gold_rule_ids"])
            )
            searched = False
            verifier_trace = None
            authority_elapsed = 0.0
            route = "fixed"
            candidate_memories: list[dict[str, Any]] = [first]
            used_memories: list[dict[str, Any]] = [first]

            if method == "memory_single":
                prediction = base_prediction
                accepted = bool(first.get("rule_ids"))
                confidence = math_exp(first_record.mean_logprob) if first_record.mean_logprob is not None else float(first.get("valid", False))

            elif method == "fixed_three_stage":
                memories = [first, fixed_stage2[index], fixed_stage3[index]]
                candidate_memories = memories
                used_memories = memories
                aggregate = 聚合记忆(memories)
                prediction = 确定性执行([aggregate], rules_by_id, case["audit_span"])
                accepted = bool(aggregate.get("rule_ids"))
                confidence = (sum(
                    set(memory.get("rule_ids", [])) == set(aggregate.get("rule_ids", [])) for memory in memories
                ) / len(memories))

            elif method == "simple_vote":
                memories = [first, check, vote3[index]]
                candidate_memories = memories
                used_memories = memories
                aggregate = 聚合记忆(memories)
                prediction = 确定性执行([aggregate], rules_by_id, case["audit_span"])
                accepted = bool(aggregate.get("rule_ids"))
                confidence = max(aggregate.get("vote_counts", {}).values(), default=0) / len(memories)

            elif method == "all_authority_rules":
                authority_started = time.perf_counter()
                rule_ids, retrieval = 权威检索候选(case, authority_index, authority_threshold)
                verifier_trace = {"authority_rule_ids": rule_ids, "retrieval": retrieval}
                if rule_ids:
                    authority_memory = {"rule_ids": rule_ids}
                    prediction = 确定性执行([authority_memory], rules_by_id, case["audit_span"])
                    accepted = True
                    confidence = 1.0
                else:
                    prediction = 安全拒答("权威规则源未找到超过校准阈值的规则")
                    accepted = False
                    confidence = 1.0
                authority_elapsed = time.perf_counter() - authority_started
                used_memories = []
                candidate_memories = [{"rule_ids": rule_ids}]

            else:
                # 三个自适应方法共用同一校准路由；区别仅在中置信慢路径。
                hard = 硬验证(first, rules_by_id, case["category"])
                if probability >= thresholds["high"] and hard["passed"] and len(conformal_set) == 1:
                    route = "high_direct"
                    prediction = base_prediction
                    accepted = True
                    used_memories = [first, check]
                    candidate_memories = [first, check]
                elif probability < thresholds["low"]:
                    route = "low_abstain"
                    prediction = 安全拒答("校准概率低于拒答阈值，转人工复核")
                    accepted = False
                    used_memories = [first, check]
                    candidate_memories = [first, check]
                elif method == "calibrated_route":
                    route = "medium_abstain"
                    prediction = 安全拒答("中置信样本未启用搜索，转人工复核")
                    accepted = False
                    used_memories = [first, check]
                    candidate_memories = [first, check]
                else:
                    route = "medium_search"
                    searched = True
                    branches = search_memories.get(index, [])
                    branch_records = search_records.get(index, [])
                    candidate_memories = [first, check, *branches]
                    used_memories = candidate_memories
                    aggregate = 聚合记忆(candidate_memories, minimum_votes=2)
                    if method == "calibrated_search_verifier":
                        authority_started = time.perf_counter()
                        verifier_trace = 独立权威验证(
                            case, aggregate, rules_by_id, authority_index, authority_threshold
                        )
                        authority_elapsed = time.perf_counter() - authority_started
                        authority_ids = verifier_trace["authority_rule_ids"]
                        if authority_ids:
                            aggregate = {**aggregate, "rule_ids": authority_ids}
                        if not authority_ids:
                            prediction = 安全拒答("独立权威规则源未确认任何规则")
                            accepted = False
                        else:
                            prediction = 确定性执行([aggregate], rules_by_id, case["audit_span"])
                            accepted = bool(verifier_trace["passed"] or verifier_trace["corrected"])
                        if index in judge_results:
                            judge = judge_results[index]
                            verifier_trace["api_judge"] = judge
                            record = judge_records[index]
                            verifier_trace["api_judge_resource"] = record.转字典()
                            judge_check = 验证大模型裁判(
                                judge, judge_allowed_ids[index], rules_by_id, case["audit_span"]
                            )
                            verifier_trace["api_judge_checks"] = judge_check["checks"]
                            accepted = accepted and judge_check["passed"]
                            if accepted:
                                prediction = judge_check["prediction"]
                                prediction["reason"] = "API 裁判通过 ID 和确定性处置硬检查"
                    else:
                        prediction = 确定性执行([aggregate], rules_by_id, case["audit_span"])
                        accepted = bool(aggregate.get("rule_ids")) and len(conformal_set) == 1
                confidence = probability

            # OOD 没有规则时，即便 prediction=REVIEW，也只有 accepted=False 才算真正拒答。
            traces.append({
                **case,
                "method": method,
                "prediction": prediction,
                "accepted": accepted,
                "confidence": float(max(0.0, min(1.0, confidence))),
                "route": route,
                "searched": searched,
                "base_correct": base_correct,
                "confidence_target": (
                    base_correct
                    if method == "memory_single" or method.startswith("calibrated_")
                    else (
                        prediction["decision"] == case["gold_decision"]
                        and set(prediction["matched_rules"]) == set(case["gold_rule_ids"])
                    )
                ),
                "candidate_contains_gold": 候选是否含金标准(candidate_memories, case["gold_rule_ids"]),
                "conformal_set": conformal_set if method.startswith("calibrated_") else None,
                "features": feature_rows[index],
                "memory_trace": candidate_memories,
                "verifier_trace": verifier_trace,
                "resource": 调用统计(used_memories),
            })
            traces[-1]["resource"]["authority_seconds"] = authority_elapsed
            traces[-1]["resource"]["elapsed_seconds"] += authority_elapsed
            if method == "calibrated_search_verifier" and index in judge_records:
                judge_record = judge_records[index]
                traces[-1]["resource"]["judge_calls"] += 1.0
                traces[-1]["resource"]["prompt_tokens"] += judge_record.prompt_tokens
                traces[-1]["resource"]["completion_tokens"] += judge_record.completion_tokens
                traces[-1]["resource"]["elapsed_seconds"] += judge_record.elapsed_seconds
                traces[-1]["resource"]["estimated_cost"] += judge_record.estimated_cost

        summary = 汇总方法(traces)
        写_jsonl(args.output_dir / f"{method}.jsonl", traces)
        (args.output_dir / f"{method}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    (args.output_dir / "comparison.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def math_exp(value: float) -> float:
    """把平均 logprob 转为 [0,1]，同时防止极端值溢出。"""

    import math

    return math.exp(max(-30.0, min(0.0, float(value))))


if __name__ == "__main__":
    主程序()
