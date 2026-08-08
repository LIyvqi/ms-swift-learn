# 01：LoRA 监督微调教师

本目录在同一个 `Qwen3.5-0.8B-Base` 上训练两种轻量教师：CoT 教师输出解题步骤，Direct 教师只输出最终答案。训练只保存 LoRA 增量参数，不复制完整基础模型。

## 脚本入口

```bash
# CoT 教师
bash course/01_lora_sft/train_cot.sh

# Direct 教师
bash course/01_lora_sft/train_direct.sh

# 通用入口
STYLE=cot EPOCHS=3 LEARNING_RATE=1e-4 bash course/01_lora_sft/train.sh
```

`train_cot.sh` 和 `train_direct.sh` 只负责设置 `STYLE`，实际参数都在 `train.sh`。

## 通用 SFT 数据格式

SFT 数据必须包含完整的“输入 + 标准回答”。`messages` 的最后一项必须是 `assistant`，这一段内容会参与交叉熵损失计算。

```json
{"id":"math-0001","question":"一个盒子有 4 排球，每排 6 个，共有多少个？","solution":"4×6=24，因此共有 24 个。","final_answer":"24","teacher_tag":"cot","messages":[{"role":"system","content":"你是一名数学助手，请逐步计算并把最终答案放入 \\boxed{}。"},{"role":"user","content":"一个盒子有 4 排球，每排 6 个，共有多少个？"},{"role":"assistant","content":"<think>共有 4 排，每排 6 个，所以 4×6=24。</think>\n\\boxed{24}"}]}
```

最小可用格式只需要 `messages`，但推荐保留 `id`、`question`、`solution` 和 `final_answer`，以便后续转成强化学习数据和做评测。

Direct 数据格式相同，只需把系统指令和 `assistant` 改成直接回答，例如：

```json
{"id":"math-0001","messages":[{"role":"system","content":"只给出最终答案，并使用 \\boxed{}。"},{"role":"user","content":"一个盒子有 4 排球，每排 6 个，共有多少个？"},{"role":"assistant","content":"\\boxed{24}"}]}
```

### 自定义数据要求

- 每行必须是合法 JSON，`messages` 必须是数组。
- `role` 使用 `system`、`user`、`assistant`；`system` 可以省略。
- 单轮训练通常是 `user → assistant`，多轮训练可重复这两个角色。
- 最后一个 `assistant` 不能为空，否则没有监督目标。
- CoT 与 Direct 最好分成两个文件，不要只依赖文本规则临时判断风格。
- 当前脚本路径由 `course/common.sh` 的 `dataset_path` 生成。换数据时可修改 `DATA_ROOT`/命名规则，或在脚本中把 `--dataset` 与 `--val_dataset` 改为自己的路径。

## 关键参数

| 参数 | 当前值 | 含义与影响 |
|---|---:|---|
| `STYLE` | `cot` | 选择 `cot_train.jsonl` 或 `direct_train.jsonl` |
| `tuner_type` | `lora` | 只训练低秩增量矩阵，显存和磁盘占用较小 |
| `lora_rank` | 16 | LoRA 秩；越大容量越高，同时增加参数量与显存 |
| `lora_alpha` | 32 | LoRA 缩放系数；本实验相当于 `alpha/rank=2` |
| `lora_dropout` | 0.05 | LoRA 分支的随机失活率，用于轻度正则化 |
| `torch_dtype` | `bfloat16` | 使用 BF16 训练，适合本机 ROCm GPU |
| `attn_impl` | `eager` | 使用已实测稳定的注意力路径 |
| `max_length` | 512 | 输入与回答合计最多 512 token，超长样本会被截断 |
| `SFT_BATCH` | 8 | 单卡训练和验证 batch；冒烟测试自动改为 1 |
| `gradient_accumulation_steps` | 1 | 不做额外梯度累积；有效 batch 等于单卡 batch |
| `LEARNING_RATE` | `1e-4` | LoRA 学习率；可通过环境变量覆盖 |
| `warmup_ratio` | 0.05 | 前 5% 步数逐渐升高学习率 |
| `save_total_limit` | 1 | 每个实验最多保留一个检查点，节省磁盘 |
| `gradient_checkpointing` | `false` | 本机显存充足，以显存换速度 |

`STEPS`、`EPOCHS` 与 `SMOKE=1` 互斥：

```bash
SMOKE=1 STYLE=cot bash course/01_lora_sft/train.sh
STEPS=100 STYLE=cot bash course/01_lora_sft/train.sh
EPOCHS=3 RUN_TAG=my_cot STYLE=cot bash course/01_lora_sft/train.sh
```

## 输出与前置关系

- CoT 输出：`outputs/01_lora_cot_<后缀>/`
- Direct 输出：`outputs/01_lora_direct_<后缀>/`
- 核心文件：`adapter_model.safetensors`、`adapter_config.json`、`args.json`
- 后续 OPD、MOPD 和 GKD 会把这些 LoRA 当作教师适配器。

## 实验注意事项

- 先做 `SMOKE=1`，确认数据模板和反向传播正确，再跑完整轮次。
- CoT 数据被截断时，最容易丢失末尾最终答案；换成长文本数据要相应提高 `max_length`。
- LoRA 不能单独推理，必须同时提供它训练时使用的基础模型。
- 验证 loss 最低不一定代表生成正确率最高。本项目 CoT 最佳生成检查点来自 3 轮计划中的第 2 轮。
- Direct 数据更短，loss 往往更稳定，但不代表任务能力更强。
- 自定义数据先检查重复、空回答、极端长度和训练/验证泄漏。
