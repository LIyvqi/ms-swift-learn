# Qwen3.5-0.8B 训练与蒸馏课程

本目录使用同一个 `Qwen3.5-0.8B-Base` 和固定 1000 条 GSM8K 数据，按从监督学习到在线/离线蒸馏的顺序组织。所有实验默认跑 900 条训练、100 条验证；加 `SMOKE=1` 时只读 16 条并训练 1 step。

所有实验已经进一步完成 100 步实测，定量结果、稳定性问题和调参建议见 [RESULTS_100_STEPS.md](RESULTS_100_STEPS.md)。多轮、学习率、散度参数、batch 与统一生成评测的最终对照见 [TUNING_RESULTS.md](TUNING_RESULTS.md)。

## 先做环境检查

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
bash course/00_setup/verify.sh
```

## 数据设计

原始数据来自 ModelScope 的 `modelscope/gsm8k`，用 seed 42 确定性抽取 1000 条。生成脚本是 `tools/prepare_gsm8k.py`。

| 视图 | assistant 内容 | 用途 |
|---|---|---|
| `cot_*` | `<think>步骤</think>` + `\boxed{答案}` | 显式推理 SFT/教师/GKD |
| `direct_*` | 只有 `\boxed{答案}` | 不展示推理的 SFT/教师/GKD |
| `mixed_*` | 两种风格各半 | Base 模型的全参 instruct-format SFT |
| `prompts_cot_*` | 无 assistant，仅 CoT 指令 | GRPO/OPD rollout |
| `prompts_direct_*` | 无 assistant，仅直接回答指令 | 无思维链 GRPO/OPD |
| `prompts_multi_*` | 两种 prompt 交替，并带 `teacher_tag` | MOPD 双教师路由 |

这里的“思维链”是 GSM8K 数据中公开的监督解题步骤，不是任何模型的隐藏内部推理。

## 推荐学习顺序

### 1. LoRA SFT：训练两位轻量教师

```bash
bash course/01_lora_sft/train_cot.sh
bash course/01_lora_sft/train_direct.sh
```

两次训练共享基础权重，只保存约 1.25% 的可训练 LoRA 参数。一个教师学显式步骤，一个教师学只给答案。对比两者的 loss、token accuracy 和生成风格。

### 2. 全参 SFT：先把 Base 变成可对话学生

```bash
bash course/02_full_sft/train.sh
```

这一步使用 50% CoT + 50% direct 数据更新文本模型全部参数，并产出后续实验共同的学生起点。ms-swift 官方 on-policy distillation 示例特别提醒：Base 与 instruct teacher 的结束符分布不同，直接做 reverse KL 容易导致长度爆炸，所以应先 SFT 学会 instruct 格式和正确终止。

### 3. GRPO：只有任务奖励，没有教师

```bash
STYLE=cot bash course/03_grpo/train.sh
STYLE=direct bash course/03_grpo/train.sh
```

每个题目采样 2 个回答，`course/plugins/gsm8k_rewards.py` 给出两种奖励：数值答案正确性和 `\boxed{}` 格式。GRPO 使用组内相对 advantage 更新策略。这是后面 OPD 的对照组。

### 4. OPD-RL：SFT 学生 + 单教师在线蒸馏

```bash
STYLE=cot bash course/04_opd/train.sh
STYLE=direct bash course/04_opd/train.sh
```

脚本严格使用第 2 步的全参 SFT checkpoint 作为学生。教师是基础模型加第 1 步的对应 LoRA。ms-swift 在 `rlhf_type=grpo` 下检测到 `teacher_model` 后自动启用 OPD-RL；本实验不加任务 reward，观察纯教师信号：

```text
A_t = teacher_kl_coef × (log p_teacher(y_t) - log p_student(y_t))
```

学生回答由当前策略在线采样，因此与普通离线 SFT 不同。

### 5. MOPD：两个风格教师按样本路由

一条命令会启动两个本地 vLLM 教师、等待 API 就绪、执行训练并在结束时关闭服务：

```bash
bash course/05_mopd/run.sh
```

教师 8001 端口负责 `teacher_tag=cot`，8002 端口负责 `teacher_tag=direct`。两个教师都是同一 0.8B Base 的不同 LoRA 风格专家；本实验的重点是学习多教师 API、logprob 和 tag 路由，不宣称它是“大模型教小模型”的能力蒸馏。若以后磁盘允许，可把服务脚本中的 adapters 替换成不同规模教师，训练脚本无需改变。

Qwen3.5 在这套 ROCm/vLLM 组合上必须把 rollout 和教师服务限定为纯文本，并启用 eager 执行。相关参数已经固化在脚本中，请勿随意删除；原因与实测结果见 [SMOKE_RESULTS.md](SMOKE_RESULTS.md)。

### 6. 离线蒸馏：GKD

```bash
STYLE=cot bash course/06_offline_gkd/train.sh
STYLE=direct bash course/06_offline_gkd/train.sh
```

选择 ms-swift 4.4.3 官方内置的 Generalized Knowledge Distillation。固定数据响应已经存在，不做学生在线 rollout；`lmbda=0` 表示纯 off-policy/offline，`beta=0.5` 使用对称 JSD，`sft_alpha=0.1` 同时保留少量标签交叉熵以稳定训练。

对照建议：保持数据、步数和 LoRA rank 不变，比较 `beta=0`（forward KL）、`0.5`（JSD）、`1`（reverse KL）对输出覆盖度和长度的影响。

## 先跑完整冒烟测试链路

```bash
SMOKE=1 bash course/01_lora_sft/train_cot.sh
SMOKE=1 bash course/01_lora_sft/train_direct.sh
SMOKE=1 bash course/02_full_sft/train.sh
SMOKE=1 STYLE=cot bash course/03_grpo/train.sh
SMOKE=1 STYLE=direct bash course/03_grpo/train.sh
SMOKE=1 STYLE=cot bash course/04_opd/train.sh
SMOKE=1 STYLE=direct bash course/04_opd/train.sh
SMOKE=1 STYLE=cot bash course/06_offline_gkd/train.sh
SMOKE=1 STYLE=direct bash course/06_offline_gkd/train.sh
SMOKE=1 bash course/05_mopd/run.sh
```

冒烟测试输出带 `_smoke` 后缀，不会被正式实验误用。正式实验不设置 `SMOKE`，脚本会自动寻找同一模式下最近的前置检查点。

本项目已经实际完成上述十条冒烟测试命令；它们验证的是环境、两种输出风格、教师路由、反向传播与保存链路，不是训练质量结论。具体指标见 [SMOKE_RESULTS.md](SMOKE_RESULTS.md)。

## 固定步数实验

设置 `STEPS` 会读取完整的 900 条训练集，并把结果隔离到 `_<步数>step` 目录。例如：

```bash
STEPS=100 bash course/01_lora_sft/train_cot.sh
```

`STEPS` 与 `SMOKE=1` 不能同时使用。按完整课程训练时仍需遵守前面的依赖顺序；后续脚本会自动寻找相同 `_100step` 模式的前置 checkpoint。

## 看结果

```bash
tensorboard --logdir outputs --port 6006
find outputs -type d -name 'checkpoint-*' -print
find outputs -name logging.jsonl -print
```

重点指标：

- SFT：`loss`、`token_acc`、验证 loss。
- GRPO：每个 reward 的均值、reward std、completion length。
- OPD/MOPD：`teacher_kl`、completion length、是否正确结束。
- GKD：distillation loss、SFT loss、CoT/direct 两种风格的格式保持率。

## 参数实验顺序

一次只改一个变量：先改 `STYLE`，再改学习率，然后 LoRA rank，最后改序列长度或 batch。每次保留 `args.json`、`logging.jsonl` 和 TensorBoard 曲线；不要只凭一两个生成样例判断训练效果。
