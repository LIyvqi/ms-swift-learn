# 06：离线 GKD 知识蒸馏

本目录使用 Generalized Knowledge Distillation。学生不在线探索新回答，而是在数据集已有的固定 `assistant` 回答上，对齐教师与学生的 token 分布，并可混入少量标准 SFT 损失。

## 前置条件与执行方式

需要全参 SFT 学生和对应风格的 LoRA 教师：

```bash
STYLE=cot bash course/06_offline_gkd/train.sh
STYLE=direct bash course/06_offline_gkd/train.sh

# 精度优先方案
EPOCHS=1 RUN_TAG=cot_beta0 STYLE=cot \
LEARNING_RATE=1e-5 GKD_BETA=0 MAX_GRAD_NORM=0.5 \
  bash course/06_offline_gkd/train.sh
```

## GKD 数据格式

离线 GKD 必须有固定 `assistant` 回答，因为教师和学生都在这些已知 token 位置上计算分布差异。

```json
{"id":"gkd-0001","question":"9 个小组，每组 4 人，共有多少人？","solution":"9×4=36，因此共有 36 人。","final_answer":"36","teacher_tag":"cot","messages":[{"role":"system","content":"请逐步计算并把答案放入 \\boxed{}。"},{"role":"user","content":"9 个小组，每组 4 人，共有多少人？"},{"role":"assistant","content":"<think>共有 9 组，每组 4 人，所以 9×4=36。</think>\n\\boxed{36}"}]}
```

Direct 数据同样必须有 `assistant`，只是回答更短：

```json
{"id":"gkd-0002","messages":[{"role":"system","content":"只给出最终答案，并使用 \\boxed{}。"},{"role":"user","content":"7 乘 8 等于多少？"},{"role":"assistant","content":"\\boxed{56}"}]}
```

### 自定义数据要求

- 最后一个 `assistant` 是离线蒸馏轨迹，不能为空。
- 固定回答不一定必须由人工编写，也可以先由教师生成并过滤，但要记录来源和质量标准。
- 若教师认为数据回答错误，KL 与 SFT 标签会产生冲突；应先清洗错误标签。
- `solution`、`final_answer`、`teacher_tag` 对 GKD 核心损失不是必需字段，但建议保留用于评测与数据追踪。
- 长回答会显著增加教师前向计算和大词表 logits 显存，应检查 token 长度分布。

## 蒸馏参数

| 参数 | 当前值 | 含义与影响 |
|---|---:|---|
| `rlhf_type` | `gkd` | 使用 ms-swift 内置 GKD 训练器 |
| `STUDENT` | 最新全参 SFT | 学生模型起点 |
| `TEACHER_ADAPTER` | 对应风格 LoRA | 教师由 Base 加 LoRA 组成 |
| `GKD_LMBDA` | 0 | 在线/学生采样轨迹比例；0 表示完全离线、只用数据固定回答 |
| `GKD_BETA` | 0.5 | 散度形式；0 为 forward KL，0.5 为对称 JSD，1 接近 reverse KL |
| `temperature` | 1.0 | 计算教师/学生分布时的温度 |
| `SFT_ALPHA` | 0.1 | 标准标签交叉熵权重，用于稳定训练和保留答案格式 |

在本项目中，`beta=0` 正确率更高但梯度更大，`beta=0.5` 更稳定且格式更好。不同任务不能直接照搬这一结论。

## 训练参数

| 参数 | 当前值 | 含义与影响 |
|---|---:|---|
| `RL_BATCH` | 2 | 每设备训练 batch；GKD 复用这个环境变量名 |
| `LEARNING_RATE` | `2e-5` | 学生 LoRA 学习率 |
| `MAX_GRAD_NORM` | 1.0 | 梯度裁剪阈值 |
| `max_length` | 512 | 提示与固定回答的最大 token 数 |
| `lora_rank` / `lora_alpha` | 16 / 32 | 学生 LoRA 容量和缩放 |
| `SAVE_TOTAL_LIMIT` | 1 | 保留检查点数；轮次对照时设为 2 |
| `gradient_checkpointing` | `false` | 关闭重计算以提高速度 |
| `EPOCHS` | 可选 | 按轮训练并每轮保存；与 `STEPS`、`SMOKE` 互斥 |

大 batch 示例：

```bash
EPOCHS=2 RUN_TAG=cot_b16 STYLE=cot RL_BATCH=16 \
LEARNING_RATE=2e-5 GKD_BETA=0 SAVE_TOTAL_LIMIT=2 \
  bash course/06_offline_gkd/train.sh
```

## 输出与选择检查点

输出位于 `outputs/06_offline_gkd_<风格>_<后缀>/`。

当前脚本没有传入独立 `--val_dataset`，所以不能只依赖训练器保存策略选择最佳轮次。应使用 `course/07_tuning/run_final_eval.sh` 或自己的固定验证集，对每轮检查点实际生成并评分。

重点观察：

- 蒸馏总 loss 及前后阶段趋势。
- `grad_norm` 和梯度裁剪频率。
- 生成正确率、格式率和输出长度。
- batch 提升后的样本吞吐与峰值显存。

## 实验注意事项

- 不同 `beta` 的 loss 数值不可直接横向比较，因为散度定义不同。
- batch 变大后每轮优化步数会减少，学习率调度也随之改变；这不仅是速度变化，也是训练参数变化。
- 本项目 batch 16 对 Direct 数据提速明显，对长短差异大的 CoT 数据收益较小。
- 1 至 2 轮通常足够初筛。除 CoT forward KL 外，本项目第二轮没有收益。
- 固定轨迹限制了学生看到的输出空间；若希望学生从自身分布学习，需要提高 `lmbda` 并重新评估稳定性。
- 教师与学生必须使用兼容 tokenizer，否则 token 级分布无法正确对齐。
