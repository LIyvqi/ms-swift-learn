# 03：GRPO 强化学习基线

本目录使用任务奖励训练学生，不提供教师模型。每道题生成多个候选回答，奖励函数分别检查最终数值答案和 `\boxed{}` 格式，GRPO 使用同组候选之间的相对优势更新 LoRA。

## 前置条件与执行方式

必须先完成全参混合 SFT，否则 Base 模型可能不会正确响应指令或结束生成。

```bash
STYLE=cot bash course/03_grpo/train.sh
STYLE=direct bash course/03_grpo/train.sh

# 课程调参方案
STEPS=200 RUN_TAG=tune_200_lr5e6_g4 STYLE=cot \
LEARNING_RATE=5e-6 NUM_GENERATIONS=4 RL_BATCH=4 \
MAX_GRAD_NORM=0.5 MAX_COMPLETION_LENGTH=192 \
  bash course/03_grpo/train.sh
```

## GRPO 数据格式

强化学习数据只提供提示，`messages` 中不能预填 `assistant`。当前奖励函数还需要顶层 `solution`，并从其中最后一个 `\boxed{}` 提取标准答案。

```json
{"id":"rl-0001","question":"每箱有 8 瓶水，5 箱共有多少瓶？","solution":"8×5=40，因此共有 \\boxed{40} 瓶。","final_answer":"40","teacher_tag":"cot","messages":[{"role":"system","content":"请逐步计算并把最终答案放入 \\boxed{}。"},{"role":"user","content":"每箱有 8 瓶水，5 箱共有多少瓶？"}]}
```

字段要求：

| 字段 | 是否必需 | 用途 |
|---|---|---|
| `messages` | 是 | rollout 的输入提示；末尾通常是 `user` |
| `solution` | 当前插件必需 | `course_gsm8k_accuracy` 的参考答案来源 |
| `id` | 建议 | 定位高奖励、低奖励和异常样本 |
| `final_answer` | 建议 | 便于独立评测；当前训练奖励不直接读取它 |
| `teacher_tag` | GRPO 不需要 | 为了与 MOPD 数据视图兼容而保留 |

### 换成自己的任务

如果任务不是单一数值答案，不能原样使用 GSM8K 奖励。应在 `course/plugins/` 中实现新奖励，并把 `--reward_funcs` 改成注册名称。奖励函数参数名可以对应数据顶层字段，例如当前 `solution` 会按样本传入。

## 关键参数

| 参数 | 当前值 | 含义与影响 |
|---|---:|---|
| `rlhf_type` | `grpo` | 使用组相对策略优化 |
| `STUDENT` | 最新全参 SFT | 学生起点，可传检查点路径覆盖 |
| `reward_funcs` | 正确性、格式 | 两项奖励相加形成训练信号 |
| `NUM_GENERATIONS` | 2 | 每个提示生成的候选数；至少两个才有组内比较意义 |
| `RL_BATCH` | 2 | 每设备提示 batch；与候选数共同影响 rollout 数量和显存 |
| `LEARNING_RATE` | `2e-5` | LoRA 策略学习率；在线训练通常需要保守调节 |
| `MAX_GRAD_NORM` | 1.0 | 梯度裁剪上限，抑制高方差更新 |
| `max_length` | 512 | 数据提示的最大长度 |
| `MAX_COMPLETION_LENGTH` | 256 | 在线生成的最大 token 数 |
| `use_vllm` | `true` | 使用 vLLM 加速 rollout |
| `vllm_mode` | `colocate` | 训练和 rollout 共用同一进程/GPU |
| `vllm_gpu_memory_utilization` | 0.35 | vLLM 计划使用的显存比例 |
| `sleep_level` | 1 | 训练阶段释放部分 vLLM 显存 |
| `tuner_type` | `lora` | 只更新策略 LoRA |
| `logging_steps` | 1 | 每一步记录奖励、长度、KL 和梯度信息 |

## 输出与观察指标

输出位于 `outputs/03_grpo_<风格>_<后缀>/`。重点观察：

- `rewards/GSM8KAccuracy/mean`：答案正确奖励。
- `rewards/GSM8KFormat/mean`：格式奖励。
- `reward_std` 与零方差比例：同组候选是否存在可学习差异。
- `completions/mean_length` 与截断率：模型是否学会合理结束。
- `grad_norm`：策略更新是否稳定。

## 实验注意事项

- 候选回答奖励完全相同的组无法提供有效相对优势；应提高采样多样性、候选数或改进奖励。
- 当前课程评测使用温度 0，但训练 rollout 会采样，两者表现可能不同。
- 格式奖励很容易学会，模型可能只学会输出 `\boxed{}` 而没有学会解题。
- 稀疏正确性奖励对 0.8B 模型较难；先提高 SFT 学生能力通常比盲目增加 GRPO 步数更有效。
- `NUM_GENERATIONS × RL_BATCH × MAX_COMPLETION_LENGTH` 会快速放大显存和计算量，调参时一次只改一个量。
- 奖励解析器目前适合普通整数和小数，不完整支持嵌套 LaTeX、区间、多答案或单位换算。
