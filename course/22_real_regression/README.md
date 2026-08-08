# 同名方法二：Regression-Aware REAL 最小复现

本节复现用户指定仓库 [YasminZhang/REAL](https://github.com/YasminZhang/REAL) 的核心思想：面向 1～5 分 LLM-as-a-Judge，不把“预测 4、真值 5”和“预测 1、真值 5”都当成同一种错误，而是直接利用有序分数距离。

官方实现基于 verl、Ray、FSDP 与 vLLM，目标模型为 8B/32B，训练数据约 10 万条；ms-swift 原生的 `loss_type=real` 是第 21 节另一个同名算法，不能直接复现本论文。因此本节用 ms-swift 的模型加载、Qwen3.5 模板和 LoRA checkpoint 接口，写了一个单卡教学训练循环。它复现目标函数核心，不宣称复现论文规模与最终 benchmark 数值。

## 数据格式

运行：

```bash
bash course/22_real_regression/prepare_data.sh
```

数据由 24 道仓库已有 GSM8K 题目派生。每题构造 1～5 分各一条候选回答，共 90 条训练、30 条独立题目验证，另有 10 条冒烟数据。标准 SFT 格式：

```json
{"messages":[{"role":"system","content":"按 1～5 分评分……"},{"role":"user","content":"题目……参考数值答案：16；候选回答：17"},{"role":"assistant","content":"<think>参考答案为16，候选值为17，绝对误差为1……</think><score>4</score>"}],"score":4}
```

REAL 在线数据去掉 assistant，但保留顶层整数 `score`：

```json
{"messages":[{"role":"system","content":"按 1～5 分评分……"},{"role":"user","content":"题目……参考数值答案：16；候选回答：17"}],"score":4}
```

自己的数据必须让 `score` 真正表示有序等级。若标签只是互不相关的五个类别，不应该计算平方误差。训练/验证按原始问题切分，不能让同一道题的不同候选跨集合泄漏。

## 先做评分格式 SFT

```bash
bash course/22_real_regression/train_sft.sh
```

这一步非常重要：RAIL 直接使用数字 token 在完整词表 softmax 中的概率。如果模型在 `<score>` 后几乎不产生数字，五个数字 token 的总概率会很低，期望分数会接近 0，而不是 1～5。SFT 让模型先学会 `<think>…</think><score>N</score>` 协议。

这里的 `<think>` 是数据中显式提供、可核查的数值误差依据，不是模型隐藏推理。

## REAL 的四个核心组件

### 1. RAIL 期望分数

在 `<score>` 后取完整词表 softmax 中数字 1～5 的概率：

```text
E[s] = Σ(k=1..5) p(token=k) × k
```

没有对五个数字重新归一化，这一点与官方源码一致。`evaluation.json` 报告 MSE、MAE、Pearson、四舍五入准确率、格式率和平均期望分数。

### 2. 回归感知奖励

每个 prompt 采样 `REAL_ROLLOUTS=4` 条显式分析轨迹：

```text
R_reg = -(E[s] - y)²
R_prob = p(token=y)
R = R_reg + beta_supp × R_prob + format_penalty
```

因此预测越接近真值，奖励越高；正确数字概率提供额外精修信号。

### 3. RLOO 降方差

同组第 i 条轨迹的基线是其余三条奖励均值：

```text
A_i = R_i - mean(R_j, j≠i)
```

脚本随后做组内标准化和 [-1,1] 裁剪。它不需要额外 value model。

### 4. 广义梯度的两部分

总损失包含：

- CoT exploration：`-A × mean(log p(采样分析前缀))`，让回归奖励更好的分析轨迹更可能出现。
- Prediction refinement：对 `<score>` 后的数字分布直接反向传播平方误差与正确数字 NLL。

```text
L = L_CoT + beta_supp_extra × (MSE(E[s], y) + beta_supp × NLL(y))
```

reward 和 RLOO advantage 会停止梯度，预测精修分支保留梯度。这正是“策略依赖奖励不能只套普通 policy gradient”的关键。

## 运行

```bash
bash course/22_real_regression/prepare_data.sh
bash course/22_real_regression/train_sft.sh
bash course/22_real_regression/train_real.sh
```

冒烟：

```bash
SMOKE=1 bash course/22_real_regression/train_sft.sh
SMOKE=1 bash course/22_real_regression/train_real.sh
```

正式默认 50 step。做 100 step：

```bash
REAL_STEPS=100 bash course/22_real_regression/train_real.sh
```

## 性能参数

- `REAL_BATCH_PROMPTS=16`、`REAL_ROLLOUTS=4`：每步同时处理 64 条轨迹，用批量提高大显存机器利用率。
- `REAL_MAX_NEW_TOKENS=48`：显式评分依据很短，限制长度能显著加快 rollout。
- `logits_to_keep`：只保留生成前缀和数字位置的 logits，避免构造 `batch × 全部输入长度 × 24.8 万词表` 的巨大张量。
- `REAL_BETA_SUPP=1.0`：正确数字概率奖励和 NLL 的权重。
- `REAL_BETA_SUPP_EXTRA=0.01`：预测精修分支权重；设 0 会只剩 CoT policy gradient，不再是完整核心复现。

显存仍有余量时可把 `REAL_BATCH_PROMPTS` 提到 20 或 24；必须同时观察每步耗时，批量过大但吞吐不升就没有意义。最小 2 prompt × 2 rollout 冒烟实测峰值为 6.54 GiB，因此正式默认已提高八倍轨迹数。

## 与官方实现的边界

本实现保留单卡 LoRA、同一次可求导前向、短合成评分数据与基础 RLOO；官方实现还有分布式 rollout buffer、动态微批、FSDP、多 benchmark 和更完整的训练调度。课程的目标是让每个公式都能在 0.8B 模型上运行和修改，而不是把缩小实验包装成论文结果复现。
