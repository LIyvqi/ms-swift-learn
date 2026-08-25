#!/usr/bin/env python3
"""在固定四种检索上下文上批量生成一个 LoRA 的审核输出。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from retrieval import 审核输入
from taxonomy import 专家, 专家系统提示, 单体系统提示, 类别, 路由系统提示


课程目录 = Path(__file__).resolve().parent
项目根目录 = 课程目录.parents[1]


def 解析参数() -> argparse.Namespace:
    """定义单 LoRA 推理参数。"""

    parser = argparse.ArgumentParser(description="生成一个 Macaron LoRA 的固定测试集输出")
    parser.add_argument("--target", required=True, choices=("baseline", "router", "l1", "l2", "l3", "l4"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=项目根目录 / "models/Qwen3.5-0.8B-Base")
    parser.add_argument("--data-dir", type=Path, default=课程目录 / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--maximum-samples", type=int, default=0)
    return parser.parse_args()


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL。"""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def 文件摘要(path: Path) -> str:
    """计算文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 适配器摘要(adapter: Path) -> str:
    """对适配器权重而非目录名建立内容指纹。"""

    candidates = sorted(adapter.glob("adapter_model*.safetensors"))
    if not candidates:
        candidates = sorted(adapter.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"适配器目录缺少 safetensors：{adapter}")
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(文件摘要(path)))
    return digest.hexdigest()


def 解析决定(text: str, allowed: set[str]) -> str | None:
    """从可能带额外文本的生成中抽取严格 decision。"""

    match = re.search(r"<decision>\s*([^<]+?)\s*</decision>", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip().upper()
    return value if value in allowed else None


def 解析标签(text: str, allowed: set[str]) -> list[str] | None:
    """使用竖线拆多标签，类别 ID 自身包含的逗号不会造成歧义。"""

    match = re.search(r"<labels>\s*(.*?)\s*</labels>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.upper() == "NONE":
        return []
    labels = [item.strip() for item in raw.split("|") if item.strip()]
    if not labels or any(label not in allowed for label in labels):
        return None
    return [category for category in 类别 if category in set(labels)]


def 解析路由(text: str) -> list[str] | None:
    """解析最多两个专家 ID。"""

    match = re.search(r"<routes>\s*(.*?)\s*</routes>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip().upper()
    if raw == "NONE":
        return []
    routes = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not routes or len(routes) > 2 or len(set(routes)) != len(routes) or any(route not in 专家 for route in routes):
        return None
    return routes


def 解析输出(target: str, text: str) -> dict[str, Any]:
    """按目标适配器的协议抽取输出并标记格式合法性。"""

    if target == "router":
        decision = 解析决定(text, {"SAFE", "UNSAFE"})
        routes = 解析路由(text)
        valid = decision is not None and routes is not None
        if valid and decision == "SAFE" and routes:
            valid = False
        if valid and decision == "UNSAFE" and not routes:
            valid = False
        return {"decision": decision, "routes": routes, "labels": None, "valid_format": valid}
    route = target.upper()
    allowed_labels = set(类别 if target == "baseline" else 专家[route]["类别"])
    allowed_decisions = {"SAFE", "UNSAFE"} if target == "baseline" else {"NO_RISK", "UNSAFE"}
    decision = 解析决定(text, allowed_decisions)
    labels = 解析标签(text, allowed_labels)
    valid = decision is not None and labels is not None
    if valid and decision in {"SAFE", "NO_RISK"} and labels:
        valid = False
    if valid and decision == "UNSAFE" and not labels and target != "baseline":
        valid = False
    return {"decision": decision, "routes": None, "labels": labels, "valid_format": valid}


def 系统提示(target: str) -> str:
    """返回与训练完全一致的系统提示。"""

    if target == "router":
        return 路由系统提示
    if target == "baseline":
        return 单体系统提示
    return 专家系统提示(target.upper())


def 主程序() -> None:
    """加载一次模型和一个 LoRA，完成全部检索模式推理。"""

    args = 解析参数()
    canonical = 读取_jsonl(args.data_dir / "evaluation_inputs.jsonl")
    if args.maximum_samples:
        canonical = canonical[: args.maximum_samples]
    row_index = {row["record_id"]: row for row in canonical}
    contexts = [
        row for row in 读取_jsonl(args.data_dir / "evaluation_contexts.jsonl")
        if row["record_id"] in row_index
    ]
    expected = len(canonical) * 4
    if len(contexts) != expected:
        raise RuntimeError(f"评测上下文应为 {expected} 条，实际 {len(contexts)} 条")

    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine

    engine = TransformersEngine(
        str(args.model),
        adapters=[str(args.adapter)],
        torch_dtype=torch.bfloat16,
        attn_impl="eager",
        device_map="cuda:0",
        max_batch_size=args.batch_size,
    )
    messages_batch = []
    for context in contexts:
        row = row_index[context["record_id"]]
        messages_batch.append(
            [
                {"role": "system", "content": 系统提示(args.target)},
                {
                    "role": "user",
                    "content": 审核输入(row["prompt"], row["response"], context["context"]),
                },
            ]
        )
    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0)
    outputs: list[str] = []
    began = time.perf_counter()
    for start in range(0, len(messages_batch), args.batch_size):
        requests = [
            InferRequest(messages=messages, chat_template_kwargs={"enable_thinking": False})
            for messages in messages_batch[start:start + args.batch_size]
        ]
        responses = engine.infer(requests, request_config=request_config)
        outputs.extend(response.choices[0].message.content or "" for response in responses)
    elapsed = time.perf_counter() - began

    adapter_sha256 = 适配器摘要(args.adapter)
    traces = []
    for context, output in zip(contexts, outputs):
        row = row_index[context["record_id"]]
        traces.append(
            {
                "record_id": row["record_id"],
                "source_record_id": row["source_record_id"],
                "evaluation_split": row["evaluation_split"],
                "target": args.target,
                "mode": context["mode"],
                "raw_output": output,
                **解析输出(args.target, output),
                "gold_is_safe": row["is_safe"],
                "gold_categories": row["categories"],
                "gold_routes": row["routes"],
                "adapter": str(args.adapter),
                "adapter_sha256": adapter_sha256,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "target": args.target,
        "adapter": str(args.adapter),
        "adapter_sha256": adapter_sha256,
        "samples": len(canonical),
        "clean_samples": sum(row["evaluation_split"] == "clean" for row in canonical),
        "obfuscated_samples": sum(row["evaluation_split"] == "obfuscated" for row in canonical),
        "generations": len(traces),
        "elapsed_seconds": elapsed,
        "generations_per_second": len(traces) / elapsed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "valid_format_rate": sum(trace["valid_format"] for trace in traces) / len(traces),
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
