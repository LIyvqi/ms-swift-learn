# 04：单教师 OPD 在线蒸馏

本目录让当前学生在线生成回答，再由对应 LoRA 教师为这些回答提供 token 级分布信号。脚本仍使用 `rlhf_type=grpo`，但配置 `teacher_model` 与 `teacher_adapters` 后，ms-swift 4.4.3 会启用 OPD-RL 路径。本实验没有任务奖励，优化信号只来自教师。

## 前置条件与执行方式

需要先准备：

1. `course/01_lora_sft/` 产生的 CoT 或 Direct 教师适配器。
2. `course/02_full_sft/` 产生的完整学生检查点。

```bash
STYLE=cot bash course/04_opd/train.sh
STYLE=direct bash course/04_opd/train.sh

# 本项目最佳 CoT 方案
STEPS=200 RUN_TAG=tune_200_lr5e6_kl02 STYLE=cot \
LEARNING_RATE=5e-6 TEACHER_KL_COEF=0.2 \
MAX_GRAD_NORM=0.5 MAX_COMPLETION_LENGTH=192 \
  bash course/04_opd/train.sh
```

## OPD 数据格式

OPD 使用提示词数据，`messages` 不含 `assistant`。纯教师蒸馏本身不需要标准答案，但建议保留 `solution` 与 `final_answer`，以便之后做统一任务评测。

```json
{"id":"opd-0001","question":"12 名学生平均分成 3 组，每组多少人？","solution":"12÷3=4，因此每组 \\boxed{4} 人。","final_answer":"4","teacher_tag":"cot","messages":[{"role":"system","content":"请逐步计算并把答案放入 \\boxed{}。"},{"role":"user","content":"12 名学生平均分成 3 组，每组多少人？"}]}
```

通用要求：

- `messages` 的最后一个角色通常是 `user`，不能放标准 `assistant` 回答。
- CoT 提示应配 CoT 教师，Direct 提示应配 Direct 教师。
- `solution` 不参与本脚本损失，但保留它能复用 GRPO 奖励和评测工具。
- 如果自己的任务只有无标注提示，也可以做纯 OPD，但必须有能够覆盖该任务的教师。

## 教师与学生参数

| 参数 | 含义 |
|---|---|
| `STUDENT` | 学生检查点；默认寻找最新全参混合 SFT 检查点 |
| `teacher_model` | 教师基础模型，本课程与学生共享 Qwen3.5 Base |
| `TEACHER_ADAPTER` | 教师 LoRA 路径；默认按 `STYLE` 查找最新教师 |
| `TEACHER_KL_COEF` | 教师 token 分布信号权重；默认 0.5，调参实验使用 0.2 |

教师优势信号可理解为：

```text
教师权重 ×（教师对当前 token 的对数概率 - 学生对当前 token 的对数概率）
```

学生生成哪些 token，教师就评价哪些 token，因此它与固定答案上的离线蒸馏不同。

## 其他关键参数

| 参数 | 当前值 | 含义与影响 |
|---|---:|---|
| `num_generations` | 1 | 每题生成一次；纯教师信号不依赖组内奖励比较 |
| `RL_BATCH` | 2 | 在线提示 batch |
| `LEARNING_RATE` | `2e-5` | 策略 LoRA 学习率 |
| `MAX_GRAD_NORM` | 1.0 | 梯度裁剪阈值 |
| `MAX_COMPLETION_LENGTH` | 256 | rollout 最大 token 数 |
| `vllm_gpu_memory_utilization` | 0.35 | colocate vLLM 显存比例 |
| `vllm_enforce_eager` | `true` | 避免当前 ROCm/Qwen3.5 图捕获兼容问题 |
| `lora_rank` / `lora_alpha` | 16 / 32 | 学生策略 LoRA 容量与缩放 |
| `log_completions` | `true` | 保存在线生成，便于检查长度与风格 |

## 输出与指标

输出位于 `outputs/04_opd_<风格>_<后缀>/`。重点观察：

- `teacher_kl`：教师与学生在当前生成 token 上的差异。
- `completions/mean_length` 和截断率：是否发生长度爆炸。
- `grad_norm`：教师信号是否导致过大更新。
- 固定验证集正确率：判断蒸馏是否真正提升任务能力。

## 实验注意事项

- `teacher_kl` 下降只说明学生更接近教师，不保证答案更正确。
- Base 与指令教师的结束符分布可能不同，所以必须先做全参 SFT 学生起点。
- 教师越弱，蒸馏上限越低；本项目 Direct 教师能力较弱，Direct-OPD 提升有限。
- 纯 KL 目标没有长度奖励。若生成频繁触顶，应降低教师权重/学习率，并加入结束符或长度目标再实验。
- 不要把教师 LoRA 错配到不同基础模型或不同 tokenizer。
- 自定义无标注数据应覆盖真实使用分布，否则学生只会在狭窄提示上模仿教师。
