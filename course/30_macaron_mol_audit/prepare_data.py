#!/usr/bin/env python3
"""从 BeaverTails 确定性分层抽取 2000 条并生成多 LoRA 训练视图。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable

from retrieval import 规则案例检索器, 审核输入, 渲染上下文
from taxonomy import (
    专家,
    专家系统提示,
    单体系统提示,
    数据源,
    数据源版本,
    样本总数,
    类别,
    路由系统提示,
    规则记录,
    随机种子,
    风险路由,
    划分规模,
    命中类别,
)


课程目录 = Path(__file__).resolve().parent
数据目录 = 课程目录 / "data"
知识目录 = 数据目录 / "knowledge"
视图目录 = 数据目录 / "views"
规范样本文件 = 数据目录 / "beavertails_2000.jsonl"
清单文件 = 数据目录 / "manifest.json"


def 解析参数() -> argparse.Namespace:
    """定义可重复采样开关。"""

    parser = argparse.ArgumentParser(description="准备 Macaron 内容审核课程数据")
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="忽略已有规范样本，重新流式扫描 BeaverTails",
    )
    return parser.parse_args()


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """用稳定紧凑格式写 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取规范样本。"""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 文件摘要(path: Path) -> str:
    """计算文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 内容摘要(row: dict[str, Any]) -> str:
    """用请求和回复识别重复内容。"""

    raw = (str(row["prompt"]) + "\0" + str(row["response"])).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def 联合签名(row: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """把安全标签和完整多标签组合视为一个分层单元。"""

    return bool(row["is_safe"]), tuple(命中类别(row["category"]))


def 最大余数配额(counts: dict[Hashable, int], target: int) -> dict[Hashable, int]:
    """用 Hamilton 最大余数法让各联合标签组合按比例分到精确总量。"""

    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError(f"配额目标 {target} 超出总体 {total}")
    if not total:
        return {key: 0 for key in counts}
    ideals = {key: value * target / total for key, value in counts.items()}
    quotas = {key: min(value, math.floor(ideals[key])) for key, value in counts.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (
            -(ideals[key] - math.floor(ideals[key])),
            -counts[key],
            repr(key),
        ),
    )
    for key in order:
        if remaining <= 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"最大余数分配后仍缺少 {remaining} 条")
    return quotas


def 加载流式数据() -> Any:
    """禁用 Xet，直接使用 Hugging Face 的流式 Parquet 读取。"""

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from datasets import load_dataset

    return load_dataset(数据源, split=数据源版本, streaming=True)


def 分层采样() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """扫描唯一源数据两次，精确维持联合标签组合比例。"""

    print("第一次扫描：统计去重后的联合标签分布", flush=True)
    raw_counts: Counter[tuple[bool, tuple[str, ...]]] = Counter()
    unique_counts: Counter[tuple[bool, tuple[str, ...]]] = Counter()
    seen_content: set[str] = set()
    raw_rows = 0
    for raw_rows, row in enumerate(加载流式数据(), start=1):
        signature = 联合签名(row)
        raw_counts[signature] += 1
        digest = 内容摘要(row)
        if digest in seen_content:
            continue
        seen_content.add(digest)
        unique_counts[signature] += 1
        if raw_rows % 50000 == 0:
            print(f"已扫描 {raw_rows} 条，唯一内容 {len(seen_content)} 条", flush=True)
    unique_rows = len(seen_content)
    # 配额以官方原始 300k 分布为准；第二遍只从唯一内容中抽取，兼顾分布和数据泄漏控制。
    quotas = 最大余数配额(dict(raw_counts), 样本总数)
    insufficient = {
        签名文本(signature): (quota, unique_counts[signature])
        for signature, quota in quotas.items()
        if quota > unique_counts[signature]
    }
    if insufficient:
        raise RuntimeError(f"以下联合层的唯一内容不足以满足原始分布配额：{insufficient}")

    print("第二次扫描：按联合标签配额执行蓄水池采样", flush=True)
    randomizer = random.Random(随机种子)
    reservoirs: dict[tuple[bool, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    stratum_seen: Counter[tuple[bool, tuple[str, ...]]] = Counter()
    seen_content.clear()
    for source_index, row in enumerate(加载流式数据(), start=1):
        digest = 内容摘要(row)
        if digest in seen_content:
            continue
        seen_content.add(digest)
        signature = 联合签名(row)
        quota = quotas.get(signature, 0)
        if quota <= 0:
            continue
        stratum_seen[signature] += 1
        bucket = reservoirs[signature]
        normalized = {
            "record_id": f"bt-{digest[:20]}",
            "source_index": source_index,
            "prompt": str(row["prompt"]).strip(),
            "response": str(row["response"]).strip(),
            "is_safe": bool(row["is_safe"]),
            "category_flags": {category: bool(row["category"].get(category, False)) for category in 类别},
        }
        normalized["categories"] = 命中类别(normalized["category_flags"])
        normalized["routes"] = 风险路由(normalized["categories"])
        if not normalized["is_safe"] and not normalized["routes"]:
            # 数据源若只给出 UNSAFE 而没有细类，则交给覆盖面最广的 L4 兜底。
            normalized["routes"] = ["L4"]
        if len(bucket) < quota:
            bucket.append(normalized)
        else:
            replace = randomizer.randrange(stratum_seen[signature])
            if replace < quota:
                bucket[replace] = normalized

    rows = [row for signature in sorted(reservoirs, key=repr) for row in reservoirs[signature]]
    if len(rows) != 样本总数:
        raise RuntimeError(f"分层样本应为 {样本总数} 条，实际 {len(rows)} 条")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("规范样本仍存在重复 record_id")
    randomizer.shuffle(rows)
    population = {
        "raw_rows": raw_rows,
        "unique_rows": unique_rows,
        "duplicates_removed": raw_rows - unique_rows,
        "distribution_rows": raw_rows,
        "joint_strata": len(raw_counts),
        "unique_joint_strata": len(unique_counts),
        "signature_counts": {签名文本(key): value for key, value in sorted(raw_counts.items(), key=lambda item: repr(item[0]))},
        "unique_signature_counts": {签名文本(key): value for key, value in sorted(unique_counts.items(), key=lambda item: repr(item[0]))},
    }
    return rows, population


def 签名文本(signature: tuple[bool, tuple[str, ...]]) -> str:
    """把联合签名转换成可读的清单键。"""

    safe, categories = signature
    return ("SAFE" if safe else "UNSAFE") + "::" + ("|".join(categories) if categories else "NONE")


def 分层划分(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按联合标签依次分配训练、验证和测试，确保总量精确且无交叉。"""

    grouped: dict[tuple[bool, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["is_safe"], tuple(row["categories"]))].append(row)
    randomizer = random.Random(随机种子 + 1)
    for bucket in grouped.values():
        randomizer.shuffle(bucket)
    counts = {key: len(bucket) for key, bucket in grouped.items()}
    train_quotas = 最大余数配额(counts, 划分规模["train"])
    remaining_counts = {key: counts[key] - train_quotas[key] for key in counts}
    validation_quotas = 最大余数配额(remaining_counts, 划分规模["validation"])
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signature, bucket in grouped.items():
        train_end = train_quotas[signature]
        validation_end = train_end + validation_quotas[signature]
        for split, items in (
            ("train", bucket[:train_end]),
            ("validation", bucket[train_end:validation_end]),
            ("test", bucket[validation_end:]),
        ):
            split_rows[split].extend({**item, "split": split} for item in items)
    ordered = []
    for split, expected in 划分规模.items():
        randomizer.shuffle(split_rows[split])
        if len(split_rows[split]) != expected:
            raise RuntimeError(f"{split} 应为 {expected} 条，实际 {len(split_rows[split])} 条")
        ordered.extend(split_rows[split])
    return ordered


def 查询文本(row: dict[str, Any]) -> str:
    """生成检索查询，不引用金标签。"""

    return f"{row['prompt']}\n{row['response']}"


def 训练检索模式(record_id: str) -> str:
    """确定性混合四种检索条件，避免模型只适应 RAG 或只适应裸输入。"""

    bucket = int(hashlib.sha256(record_id.encode()).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "none"
    if bucket <= 2:
        return "rules"
    if bucket <= 4:
        return "cases"
    return "full"


def 标签文本(categories: list[str]) -> str:
    """类别 ID 内部含逗号，因此多标签必须使用竖线分隔。"""

    return "|".join(categories) if categories else "NONE"


def 目标文本(row: dict[str, Any], target: str) -> str:
    """为路由器、单体基线或领域专家生成严格答案。"""

    if target == "router":
        decision = "SAFE" if row["is_safe"] else "UNSAFE"
        routes = ",".join(row["routes"][:2]) if row["routes"] else "NONE"
        return f"<decision>{decision}</decision><routes>{routes}</routes>"
    if target == "baseline":
        decision = "SAFE" if row["is_safe"] else "UNSAFE"
        return f"<decision>{decision}</decision><labels>{标签文本(row['categories'])}</labels>"
    domain_categories = [category for category in 专家[target]["类别"] if category in row["categories"]]
    decision = "UNSAFE" if domain_categories else "NO_RISK"
    return f"<decision>{decision}</decision><labels>{标签文本(domain_categories)}</labels>"


def 生成视图(
    rows: list[dict[str, Any]],
    retriever: 规则案例检索器,
    target: str,
    split: str,
) -> list[dict[str, Any]]:
    """用训练案例库实时检索，生成 ms-swift messages 数据。"""

    system = 路由系统提示 if target == "router" else 单体系统提示 if target == "baseline" else 专家系统提示(target)
    result = []
    for row in rows:
        mode = 训练检索模式(row["record_id"]) if split == "train" else "full"
        retrieved = retriever.检索(
            查询文本(row), mode, exclude_record_id=row["record_id"], rule_top_k=3, case_top_k=3
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": 审核输入(row["prompt"], row["response"], 渲染上下文(retrieved)),
            },
            {"role": "assistant", "content": 目标文本(row, target)},
        ]
        result.append(
            {
                "messages": messages,
                "record_id": row["record_id"],
                "is_safe": row["is_safe"],
                "categories": row["categories"],
                "routes": row["routes"],
                "retrieval_mode": mode,
            }
        )
    return result


def 生成评测上下文(
    rows: list[dict[str, Any]], retriever: 规则案例检索器
) -> list[dict[str, Any]]:
    """预先冻结四种测试检索结果，确保六个 LoRA 使用完全相同的外部证据。"""

    result = []
    for row in rows:
        full = retriever.检索(
            查询文本(row), "full", exclude_record_id=row["record_id"], rule_top_k=3, case_top_k=3
        )
        modes = {
            "none": {"rules": [], "cases": []},
            "rules": {"rules": full["rules"], "cases": []},
            "cases": {"rules": [], "cases": full["cases"]},
            "full": full,
        }
        for mode, retrieved in modes.items():
            result.append(
                {
                    "record_id": row["record_id"],
                    "mode": mode,
                    "context": 渲染上下文(retrieved),
                    "rules": [
                        {
                            "rule_id": rule["rule_id"],
                            "version": rule["version"],
                            "category": rule["category"],
                            "score": rule["retrieval_score"],
                        }
                        for rule in retrieved["rules"]
                    ],
                    "cases": [
                        {
                            "record_id": case["record_id"],
                            "is_safe": case["is_safe"],
                            "categories": case["categories"],
                            "score": case["retrieval_score"],
                        }
                        for case in retrieved["cases"]
                    ],
                }
            )
    return result


def 表面扰动(text: str, salt: str) -> str:
    """对部分长英文词插入点号并改变大小写，语义不变但表面形式偏离训练分布。"""

    words = re.split(r"([A-Za-z]{6,})", text)
    changed = 0
    for index in range(1, len(words), 2):
        word = words[index]
        selector = int(hashlib.sha256(f"{salt}:{index}:{word.lower()}".encode()).hexdigest()[:8], 16)
        if selector % 5 != 0 or changed >= 6:
            continue
        if selector % 2:
            words[index] = ".".join(word.lower())
        else:
            words[index] = " ".join(character.upper() if offset % 2 == 0 else character.lower() for offset, character in enumerate(word))
        changed += 1
    if changed == 0:
        candidate = next((index for index in range(1, len(words), 2)), None)
        if candidate is not None:
            words[candidate] = ".".join(words[candidate].lower())
    result = "".join(words)
    if result == text:
        # 极短句可能没有六字母长词；退化为扰动第一个三字母以上单词。
        result = re.sub(
            r"[A-Za-z]{3,}",
            lambda match: ".".join(match.group(0).lower()),
            text,
            count=1,
        )
    return result


def 构造泛化评测(rows: list[dict[str, Any]], target: int = 100) -> list[dict[str, Any]]:
    """从测试集按联合标签抽取一半并构造不改变语义的表面扰动副本。"""

    grouped: dict[tuple[bool, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["is_safe"], tuple(row["categories"]))].append(row)
    quotas = 最大余数配额({key: len(items) for key, items in grouped.items()}, target)
    selected = []
    for signature, items in grouped.items():
        ordered = sorted(items, key=lambda row: hashlib.sha256(row["record_id"].encode()).hexdigest())
        selected.extend(ordered[: quotas[signature]])
    result = []
    for row in selected:
        result.append(
            {
                **row,
                "record_id": row["record_id"] + "-obfuscated",
                "source_record_id": row["record_id"],
                "prompt": 表面扰动(row["prompt"], row["record_id"] + ":prompt"),
                "response": 表面扰动(row["response"], row["record_id"] + ":response"),
                "evaluation_split": "obfuscated",
            }
        )
    return sorted(result, key=lambda row: row["record_id"])


def 专家训练子集(rows: list[dict[str, Any]], route: str) -> list[dict[str, Any]]:
    """保留全部领域正例，并确定性抽取最多两倍负例抑制 NO_RISK 偏置。"""

    positives = [row for row in rows if route in row["routes"]]
    negatives = [row for row in rows if route not in row["routes"]]
    randomizer = random.Random(随机种子 + int(route[1:]) * 101)
    randomizer.shuffle(negatives)
    selected = [*positives, *negatives[: min(len(negatives), len(positives) * 2)]]
    randomizer.shuffle(selected)
    return selected


def 分布统计(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """统计安全率、类别边际率、专家边际率和多标签比例。"""

    total = len(rows)
    return {
        "rows": total,
        "safe_count": sum(row["is_safe"] for row in rows),
        "safe_rate": sum(row["is_safe"] for row in rows) / max(1, total),
        "multi_label_count": sum(len(row["categories"]) > 1 for row in rows),
        "multi_label_rate": sum(len(row["categories"]) > 1 for row in rows) / max(1, total),
        "category": {
            category: {
                "count": sum(category in row["categories"] for row in rows),
                "rate": sum(category in row["categories"] for row in rows) / max(1, total),
            }
            for category in 类别
        },
        "route": {
            route: {
                "count": sum(route in row["routes"] for row in rows),
                "rate": sum(route in row["routes"] for row in rows) / max(1, total),
            }
            for route in 专家
        },
    }


def 从人口签名统计(population: dict[str, Any]) -> dict[str, Any]:
    """从联合签名恢复源总体的边际分布。"""

    total = population["distribution_rows"]
    safe_count = 0
    category_counts = Counter()
    route_counts = Counter()
    multi_label_count = 0
    for signature_text, count in population["signature_counts"].items():
        status, raw_categories = signature_text.split("::", 1)
        categories = [] if raw_categories == "NONE" else raw_categories.split("|")
        safe_count += count if status == "SAFE" else 0
        multi_label_count += count if len(categories) > 1 else 0
        category_counts.update({category: count for category in categories})
        route_counts.update({route: count for route in set(风险路由(categories))})
    return {
        "rows": total,
        "safe_count": safe_count,
        "safe_rate": safe_count / total,
        "multi_label_count": multi_label_count,
        "multi_label_rate": multi_label_count / total,
        "category": {
            category: {"count": category_counts[category], "rate": category_counts[category] / total}
            for category in 类别
        },
        "route": {
            route: {"count": route_counts[route], "rate": route_counts[route] / total}
            for route in 专家
        },
    }


def 主程序() -> None:
    """生成规范样本、规则库、案例库、训练视图和分布清单。"""

    args = 解析参数()
    数据目录.mkdir(parents=True, exist_ok=True)
    if 规范样本文件.exists() and not args.force_resample:
        print(f"复用已有规范样本：{规范样本文件}", flush=True)
        rows = 读_jsonl(规范样本文件)
        if not 清单文件.exists():
            raise FileNotFoundError("规范样本存在但 manifest.json 缺失，请使用 --force-resample")
        old_manifest = json.loads(清单文件.read_text(encoding="utf-8"))
        population = old_manifest["source_scan"]
    else:
        sampled, population = 分层采样()
        rows = 分层划分(sampled)
        写_jsonl(规范样本文件, rows)

    if len(rows) != 样本总数:
        raise RuntimeError(f"规范样本不是 {样本总数} 条")
    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != dict(划分规模):
        raise RuntimeError(f"划分规模异常：{dict(split_counts)}")
    split_ids = {
        split: {row["record_id"] for row in rows if row["split"] == split}
        for split in 划分规模
    }
    for left in split_ids:
        for right in split_ids:
            if left < right and split_ids[left] & split_ids[right]:
                raise RuntimeError(f"{left} 与 {right} 存在数据泄漏")

    rules_path = 知识目录 / "rules.jsonl"
    # 规则库允许独立升级；重复准备数据时不得静默覆盖人工维护过的版本。
    rules = 读_jsonl(rules_path) if rules_path.exists() else 规则记录()
    train_rows = [row for row in rows if row["split"] == "train"]
    base_cases = [
        {
            "record_id": row["record_id"],
            "prompt": row["prompt"],
            "response": row["response"],
            "is_safe": row["is_safe"],
            "categories": row["categories"],
            "routes": row["routes"],
            "source_split": "train",
            "source": "BeaverTails 分层训练样本",
            "reviewed_by": "原始数据集标注",
            "reviewed_at": "2026-08-26",
        }
        for row in train_rows
    ]
    cases_path = 知识目录 / "cases.jsonl"
    reviewed_cases = (
        [row for row in 读_jsonl(cases_path) if row.get("source_split") == "reviewed"]
        if cases_path.exists()
        else []
    )
    cases = [*base_cases, *reviewed_cases]
    写_jsonl(rules_path, rules)
    写_jsonl(cases_path, cases)
    retriever = 规则案例检索器(rules, cases)

    generated_files = [规范样本文件, rules_path, cases_path]
    targets = ["router", "baseline", *专家.keys()]
    for split in ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        for target in targets:
            target_rows = (
                专家训练子集(split_rows, target)
                if split == "train" and target in 专家
                else split_rows
            )
            filename_target = target.lower()
            path = 视图目录 / f"{filename_target}_{split}.jsonl"
            写_jsonl(path, 生成视图(target_rows, retriever, target, split))
            generated_files.append(path)

    evaluation_inputs_path = 数据目录 / "evaluation_inputs.jsonl"
    evaluation_contexts = 数据目录 / "evaluation_contexts.jsonl"
    test_rows = [row for row in rows if row["split"] == "test"]
    clean_inputs = [{**row, "source_record_id": row["record_id"], "evaluation_split": "clean"} for row in test_rows]
    evaluation_inputs = [*clean_inputs, *构造泛化评测(test_rows)]
    写_jsonl(evaluation_inputs_path, evaluation_inputs)
    写_jsonl(evaluation_contexts, 生成评测上下文(evaluation_inputs, retriever))
    generated_files.extend((evaluation_inputs_path, evaluation_contexts))

    population_distribution = 从人口签名统计(population)
    sample_distribution = 分布统计(rows)
    maximum_category_deviation = max(
        abs(sample_distribution["category"][category]["rate"] - population_distribution["category"][category]["rate"])
        for category in 类别
    )
    manifest = {
        "course": "Macaron-V1 风格多 LoRA 内容审核",
        "source": 数据源,
        "source_split": 数据源版本,
        "source_url": "https://huggingface.co/datasets/PKU-Alignment/BeaverTails",
        "license": "CC BY-NC 4.0",
        "seed": 随机种子,
        "sample_size": 样本总数,
        "split_sizes": dict(划分规模),
        "stratification": "按原始 300k 的 is_safe + 完整多标签联合签名使用 Hamilton 配额；每层从内容去重后的记录中蓄水池采样",
        "source_scan": population,
        "source_distribution": population_distribution,
        "sample_distribution": sample_distribution,
        "maximum_category_rate_deviation_percentage_points": maximum_category_deviation * 100,
        "experts": 专家,
        "training_retrieval_mix": {"none": "10%", "rules": "20%", "cases": "20%", "full": "50%"},
        "expert_training_balance": "每个专家保留全部领域正例，并确定性抽取不超过正例两倍的领域负例；规范 2000 条样本本身不重采样",
        "knowledge_leakage_guard": "基础案例只来自 train；可追加人工复核案例；训练检索排除自身；validation/test 不自动进入案例库",
        "files": {
            str(path.relative_to(课程目录)): {
                "sha256": 文件摘要(path),
                "bytes": path.stat().st_size,
            }
            for path in generated_files
        },
    }
    清单文件.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sample_size": len(rows),
                "splits": dict(split_counts),
                "safe_rate": sample_distribution["safe_rate"],
                "multi_label_rate": sample_distribution["multi_label_rate"],
                "max_category_deviation_pp": maximum_category_deviation * 100,
                "data_dir": str(数据目录),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    主程序()
