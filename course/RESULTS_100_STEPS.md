# 100 步训练实验报告

日期：2026-08-08（UTC）

本报告对应完整的 100 步复测。所有实验都读取 900 条训练数据，而不是只读取 16 条冒烟测试数据；每项实验均完成反向传播并保存了 `checkpoint-100`。这些结果足以观察训练趋势和暴露稳定性问题，但仍不能等同于完整的模型能力评测。

## 实验配置

- 基础模型：Qwen3.5-0.8B-Base。
- 训练框架：官方 ms-swift Git 标签 v4.4.3。
- 监督微调批大小：8；强化学习和蒸馏批大小：2。
- 数据：固定的 900 条 GSM8K 训练数据，另有 100 条监督微调验证数据。
- 输出目录统一带 `_100step` 后缀，与旧的单步冒烟测试完全隔离。
- 每个实验只保留最终检查点，以控制磁盘占用。

## 监督微调结果

| 实验 | 首个记录 loss | 末个记录 loss | 验证 loss | 验证 token 准确率 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| CoT LoRA | 1.1272 | 0.3563 | 0.4563 | 0.8710 | 22.79 GiB |
| Direct LoRA | 1.2481 | 0.4617 | 0.5505 | 0.8056 | 9.24 GiB |
| CoT/Direct 混合全参数 SFT | 1.0230 | 0.3961 | 0.4831 | 0.8604 | 24.14 GiB |

三组监督微调的训练 loss 都明显下降，验证指标也保持有限，没有出现数值溢出。两个 LoRA 检查点作为后续教师，全参数 SFT 检查点作为后续学生起点。

## GRPO 结果

GRPO 是基于组内相对奖励的策略优化。这里每道题生成两个候选回答，奖励由“最终数值是否正确”和“是否使用 `\boxed{}` 格式”两部分组成。

勘误：本页是旧答案型实验记录。表中所谓 CoT 的平均回答只有 7.45 token，后来逐条检查确认思考块全部为空，不能用于证明 CoT 学习。新的显式 CoT 数据、thinking 参数、过程奖励和冒烟结果见 [第 03 课](03_grpo/README.md)。

| 风格 | 平均总奖励 | 平均正确性奖励 | 平均格式奖励 | 非零梯度步 | 平均回答长度 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| CoT | 1.095 | 0.095 | 1.000 | 72/100 | 7.45 token | 68.61 GiB |
| Direct | 1.080 | 0.080 | 1.000 | 96/100 | 7.33 token | 68.61 GiB |

模型很快稳定掌握了 `\boxed{}` 格式，因此格式奖励饱和；数学正确性奖励仍然稀疏。两个候选答案偶尔得到完全相同的奖励，此时组内相对优势为零，对应训练步不会获得任务奖励梯度。下一轮 GRPO 学习可把 `num_generations` 从 2 增加到 4，并加入部分分数奖励，但显存和生成耗时都会增加。

## 单教师 OPD 结果

OPD 是在线策略蒸馏：学生先自行生成回答，再由教师对生成 token 计算概率，用教师与学生的分布差异更新学生。本轮教师已经先完成 100 步 LoRA，因此不再出现单步教师测试时教师 KL 接近零的问题。

| 风格 | 平均教师 KL | 末步教师 KL | 非零梯度步 | 平均回答长度 | 平均截断率 | 最大梯度范数 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoT | 0.4240 | 0.1675 | 100/100 | 125.44 token | 15.5% | 104 | 72.60 GiB |
| Direct | 0.6670 | 0.1687 | 100/100 | 35.58 token | 2.0% | 288 | 72.15 GiB |

两种风格都获得了持续的非零蒸馏信号。CoT 更容易生成长回答并触及 256-token 上限；Direct 的平均长度较短，但出现过较高的梯度尖峰。日志中的梯度范数是裁剪前数值，仍应把它作为降低学习率和收紧梯度裁剪阈值的依据。

## 双教师 MOPD 结果

MOPD 使用两个本地教师服务：CoT 样本路由到 8001 端口，Direct 样本路由到 8002 端口。两位教师都由同一个基础模型加对应的 100-step LoRA 适配器组成。

| 平均教师 KL | 末步教师 KL | 非零梯度步 | 平均回答长度 | 平均截断率 | 最大梯度范数 | 峰值显存 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5310 | 0.3445 | 100/100 | 183.55 token | 58.0% | 1152 | 71.19 GiB |

双教师启动、标签路由、token 概率请求、反向传播、检查点保存和退出清理全部成功。与此同时，58% 的平均截断率和较大的梯度尖峰说明默认纯 KL 配置还不稳定。该检查点适合用于学习和诊断，不建议直接视为当前最佳模型。

下一轮 MOPD 建议依次做以下单变量对照：

1. 把学习率从 `2e-5` 降到 `5e-6`。
2. 显式设置更严格的梯度裁剪阈值，例如 `max_grad_norm=0.5`。
3. 增强结束符监督，或者加入长度惩罚与任务正确性奖励。
4. 在解决终止问题后，再把最大生成长度从 256 提高，避免单纯放大无效长回答。

## 离线 GKD 结果

GKD 使用数据集中固定的标准回答，不需要学生在线生成，因此不受回答长度漂移影响。这里 `lmbda=0` 表示完全离线，`beta=0.5` 表示使用对称的詹森－香农散度，`sft_alpha=0.1` 保留少量标签交叉熵。

| 风格 | 首步 loss | 末步 loss | 平均 loss | 最大梯度范数 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| CoT | 0.1407 | 0.0944 | 0.0993 | 0.529 | 13.79 GiB |
| Direct | 0.2328 | 0.1429 | 0.1610 | 3.145 | 4.67 GiB |

两种离线 GKD 都稳定完成，loss 总体下降，显存远低于在线蒸馏。若目的是先理解知识蒸馏并获得稳定基线，建议先学习离线 GKD，再对比 OPD 和 MOPD 的在线采样特性。

## 检查点位置

```text
outputs/01_lora_cot_100step/.../checkpoint-100
outputs/01_lora_direct_100step/.../checkpoint-100
outputs/02_full_sft_mixed_100step/.../checkpoint-100
outputs/03_grpo_cot_100step/.../checkpoint-100
outputs/03_grpo_direct_100step/.../checkpoint-100
outputs/04_opd_cot_100step/.../checkpoint-100
outputs/04_opd_direct_100step/.../checkpoint-100
outputs/05_mopd_100step/.../checkpoint-100
outputs/06_offline_gkd_cot_100step/.../checkpoint-100
outputs/06_offline_gkd_direct_100step/.../checkpoint-100
```

每个运行目录中的 `args.json` 保存完整参数，`logging.jsonl` 保存逐步指标，`runs/` 可供 TensorBoard 绘制曲线。原始终端日志位于 `outputs/training_100step_logs/`。

## 如何复现

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh

STEPS=100 bash course/01_lora_sft/train_cot.sh
STEPS=100 bash course/01_lora_sft/train_direct.sh
STEPS=100 bash course/02_full_sft/train.sh
STEPS=100 STYLE=cot bash course/03_grpo/train.sh
STEPS=100 STYLE=direct bash course/03_grpo/train.sh
STEPS=100 STYLE=cot bash course/04_opd/train.sh
STEPS=100 STYLE=direct bash course/04_opd/train.sh
STEPS=100 bash course/05_mopd/run.sh
STEPS=100 STYLE=cot bash course/06_offline_gkd/train.sh
STEPS=100 STYLE=direct bash course/06_offline_gkd/train.sh
```

这些命令存在前置依赖，应按顺序运行，不要并行占用同一块 GPU。
