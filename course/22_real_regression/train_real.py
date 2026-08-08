#!/usr/bin/env python3
"""基于 ms-swift 模型与 LoRA 接口复现回归感知 REAL 的最小教学版本。"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from swift import get_model_processor
from swift.tuners import Swift


@dataclass
class 轨迹:
    """一条学生采样轨迹及其回归标签。"""

    prompt_ids: list[int]
    prefix_ids: list[int]
    score: int
    format_ok: bool


def 读取_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def 最新检查点(directory: Path) -> Path:
    checkpoints = list(directory.glob("**/checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"找不到 SFT 检查点：{directory}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def 去掉结尾填充(ids: list[int], pad_id: int, eos_id: int | None) -> list[int]:
    result = []
    for token in ids:
        if token == pad_id or (eos_id is not None and token == eos_id):
            break
        result.append(token)
    return result


def 查找子序列(sequence: list[int], pattern: list[int]) -> int:
    for start in range(len(sequence) - len(pattern) + 1):
        if sequence[start:start + len(pattern)] == pattern:
            return start
    return -1


@torch.no_grad()
def 生成轨迹(model, tokenizer, rows: list[dict], num_rollouts: int, max_new_tokens: int,
         temperature: float) -> list[轨迹]:
    """按 prompt 分组采样，并保留到 `<score>` 为止的精确 token 前缀。"""

    model.eval()
    prompt_texts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt_texts, padding=True, truncation=True, max_length=640, return_tensors="pt")
    tokenizer.padding_side = old_padding_side
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generated = model.generate(
        **inputs,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.95,
        num_return_sequences=num_rollouts,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    input_width = inputs["input_ids"].shape[1]
    marker_ids = tokenizer.encode("<score>", add_special_tokens=False)
    rollouts: list[轨迹] = []
    for output_index, output in enumerate(generated[:, input_width:].tolist()):
        row_index = output_index // num_rollouts
        generated_ids = 去掉结尾填充(output, tokenizer.pad_token_id, tokenizer.eos_token_id)
        marker_start = 查找子序列(generated_ids, marker_ids)
        format_ok = marker_start >= 0
        if format_ok:
            prefix_ids = generated_ids[:marker_start + len(marker_ids)]
        else:
            prefix_ids = generated_ids + marker_ids

        prompt_ids = inputs["input_ids"][row_index][inputs["attention_mask"][row_index].bool()].tolist()
        rollouts.append(轨迹(
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            score=int(rows[row_index]["score"]),
            format_ok=format_ok,
        ))
    return rollouts


def 左填充轨迹(rollouts: list[轨迹], digit_ids: list[int], pad_id: int, device: torch.device):
    suffixes = [item.prefix_ids + [digit_ids[item.score - 1]] for item in rollouts]
    sequences = [item.prompt_ids + suffix for item, suffix in zip(rollouts, suffixes)]
    max_length = max(map(len, sequences))
    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row_index, sequence in enumerate(sequences):
        input_ids[row_index, -len(sequence):] = torch.tensor(sequence, device=device)
        attention_mask[row_index, -len(sequence):] = 1
    return input_ids, attention_mask, suffixes


def 前向损失(model, rollouts: list[轨迹], digit_ids: list[int], pad_id: int, num_rollouts: int,
         beta_supp: float, beta_supp_extra: float, format_penalty: float):
    """一次可求导前向同时计算 RLOO CoT 策略梯度和数字预测精修。"""

    input_ids, attention_mask, suffixes = 左填充轨迹(rollouts, digit_ids, pad_id, model.device)
    max_suffix = max(map(len, suffixes))
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=max_suffix + 1,
    )
    logits = outputs.logits.float()
    returned_length = logits.shape[1]
    digit_id_tensor = torch.tensor(digit_ids, device=model.device)

    expected_scores = []
    gold_logps = []
    prefix_mean_logps = []
    for row_index, (item, suffix) in enumerate(zip(rollouts, suffixes)):
        prefix_length = len(item.prefix_ids)
        suffix_length = len(suffix)
        start = returned_length - suffix_length - 1
        if start < 0:
            raise RuntimeError("logits_to_keep 返回长度不足，无法覆盖生成前缀")

        digit_logits = logits[row_index, returned_length - 2]
        log_denominator = torch.logsumexp(digit_logits, dim=-1)
        digit_logps = digit_logits[digit_id_tensor] - log_denominator
        digit_probs = digit_logps.exp()
        digit_values = torch.arange(1, 6, device=model.device, dtype=torch.float32)
        expected_scores.append((digit_probs * digit_values).sum())
        gold_logps.append(digit_logps[item.score - 1])

        prefix_logits = logits[row_index, start:start + prefix_length]
        prefix_targets = torch.tensor(item.prefix_ids, device=model.device)
        selected = prefix_logits.gather(1, prefix_targets[:, None]).squeeze(1)
        prefix_logps = selected - torch.logsumexp(prefix_logits, dim=-1)
        prefix_mean_logps.append(prefix_logps.mean())

    expected = torch.stack(expected_scores)
    gold_logp = torch.stack(gold_logps)
    cot_logp = torch.stack(prefix_mean_logps)
    labels = torch.tensor([item.score for item in rollouts], device=model.device, dtype=torch.float32)

    regression_reward = -((expected.detach() - labels) ** 2)
    accuracy_reward = gold_logp.detach().exp()
    format_reward = torch.tensor(
        [0.0 if item.format_ok else -format_penalty for item in rollouts],
        device=model.device,
    )
    rewards = regression_reward + beta_supp * accuracy_reward + format_reward
    if len(rollouts) % num_rollouts != 0:
        raise ValueError("轨迹总数必须能被 num_rollouts 整除")
    grouped = rewards.view(-1, num_rollouts)
    if num_rollouts > 1:
        advantages = grouped - (grouped.sum(dim=1, keepdim=True) - grouped) / (num_rollouts - 1)
    else:
        advantages = grouped - grouped.mean(dim=1, keepdim=True)
    advantages = advantages / (advantages.std(dim=1, keepdim=True, unbiased=False) + 1e-6)
    advantages = advantages.clamp(-1.0, 1.0).reshape(-1).detach()

    cot_loss = -(advantages * cot_logp).mean()
    l2_loss = ((expected - labels) ** 2).mean()
    nll_loss = -gold_logp.mean()
    refinement_loss = beta_supp_extra * (l2_loss + beta_supp * nll_loss)
    total_loss = cot_loss + refinement_loss
    metrics = {
        "loss": total_loss.detach().item(),
        "cot_loss": cot_loss.detach().item(),
        "l2_loss": l2_loss.detach().item(),
        "nll_loss": nll_loss.detach().item(),
        "reward": rewards.mean().item(),
        "expected_score": expected.detach().mean().item(),
        "gold_probability": torch.stack([
            torch.exp(value) for value in gold_logps
        ]).detach().mean().item(),
        "format_rate": sum(item.format_ok for item in rollouts) / len(rollouts),
    }
    return total_loss, metrics


@torch.no_grad()
def rail_evaluate(model, tokenizer, rows: list[dict], digit_ids: list[int], batch_prompts: int,
              max_new_tokens: int) -> dict[str, float]:
    """用数字 token 的期望值进行 RAIL 推理评测。"""

    predictions: list[float] = []
    labels: list[float] = []
    format_hits = 0
    for start in range(0, len(rows), batch_prompts):
        batch = rows[start:start + batch_prompts]
        rollouts = 生成轨迹(model, tokenizer, batch, 1, max_new_tokens, 0.0)
        input_ids, attention_mask, suffixes = 左填充轨迹(
            rollouts, digit_ids, tokenizer.pad_token_id, model.device
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=max(map(len, suffixes)) + 1,
        )
        digit_logits = outputs.logits[:, -2, :].float()
        full_log_denominator = torch.logsumexp(digit_logits, dim=-1, keepdim=True)
        probs = torch.exp(digit_logits[:, digit_ids] - full_log_denominator)
        expected = (probs * torch.arange(1, 6, device=model.device)).sum(dim=-1)
        predictions.extend(expected.cpu().tolist())
        labels.extend(float(item.score) for item in rollouts)
        format_hits += sum(item.format_ok for item in rollouts)

    pred = torch.tensor(predictions)
    gold = torch.tensor(labels)
    pred_centered = pred - pred.mean()
    gold_centered = gold - gold.mean()
    pearson = (pred_centered * gold_centered).sum() / (
        pred_centered.square().sum().sqrt() * gold_centered.square().sum().sqrt() + 1e-8
    )
    rounded = pred.round().clamp(1, 5)
    return {
        "mse": ((pred - gold) ** 2).mean().item(),
        "mae": (pred - gold).abs().mean().item(),
        "pearson": pearson.item(),
        "rounded_accuracy": (rounded == gold).float().mean().item(),
        "format_rate": format_hits / len(rows),
        "mean_expected_score": pred.mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="回归感知 REAL 最小复现")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-prompts", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--beta-supp", type=float, default=1.0)
    parser.add_argument("--beta-supp-extra", type=float, default=0.01)
    parser.add_argument("--format-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = 读取_jsonl(args.train_data)
    eval_rows = 读取_jsonl(args.eval_data)

    model, processor = get_model_processor(
        str(args.model),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_impl="eager",
    )
    model = Swift.from_pretrained(model, str(args.adapter), is_trainable=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    digit_ids = []
    for digit in range(1, 6):
        ids = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"数字 {digit} 不是单 token：{ids}")
        digit_ids.append(ids[0])

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    baseline = rail_evaluate(
        model, tokenizer, eval_rows, digit_ids, args.batch_prompts, args.max_new_tokens
    )
    print("训练前 RAIL：" + json.dumps(baseline, ensure_ascii=False))

    log_path = args.output_dir / "training_log.jsonl"
    for step in range(1, args.max_steps + 1):
        rows = random.sample(train_rows, k=min(args.batch_prompts, len(train_rows)))
        rollouts = 生成轨迹(
            model, tokenizer, rows, args.num_rollouts, args.max_new_tokens, args.temperature
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = 前向损失(
            model,
            rollouts,
            digit_ids,
            tokenizer.pad_token_id,
            args.num_rollouts,
            args.beta_supp,
            args.beta_supp_extra,
            args.format_penalty,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        metrics.update({"step": step, "grad_norm": float(grad_norm)})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print("REAL 步骤：" + json.dumps(metrics, ensure_ascii=False))

    checkpoint = args.output_dir / f"checkpoint-{args.max_steps}"
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)
    final_metrics = rail_evaluate(
        model, tokenizer, eval_rows, digit_ids, args.batch_prompts, args.max_new_tokens
    )
    report = {
        "训练前": baseline,
        "训练后": final_metrics,
        "参数": vars(args) | {"model": str(args.model), "adapter": str(args.adapter),
                             "train_data": str(args.train_data), "eval_data": str(args.eval_data),
                             "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("训练后 RAIL：" + json.dumps(final_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
