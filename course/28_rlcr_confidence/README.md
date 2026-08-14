# 第 28 课：RLCR 分类置信度强化学习

本课不是用一句提示词让模型“自称有信心”，而是对 `Qwen3.5-0.8B-Base` 先做 LoRA 格式
SFT，再分别进行三组 100-step 在线强化学习。类别与数值置信度都是模型真正生成的 token，
奖励由金标签和生成的数值共同计算，梯度会更新 LoRA 参数。

对应研究是 [Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty](https://arxiv.org/abs/2507.16806)
提出的 RLCR 思路，以及 [Rewarding Doubt](https://openreview.net/pdf/7dc238561a81bdd1cc2949814d255de6caaf0c3d.pdf)
采用的对数 proper scoring rule。本课是在中文四分类上的教学缩小复现，不宣称逐项等同原论文的大规模实验。

## 要解决的问题

普通分类强化学习只优化是否答对：

```text
r_correct = 1[预测类别 = 金标签]
```

它没有理由区分“容易且答对”和“犹豫且答错”，甚至可能始终输出高置信度。RLCR 要求策略生成：

```text
<answer>财经</answer><confidence>0.73</confidence>
```

令 `c` 表示预测是否正确，`q` 表示自报的正确概率。本课比较：

```text
正确性基线：r = c + 0.2 × r_format
Brier-RLCR：r = c - (q-c)^2 + 0.2 × r_format
对数 RLCR：r = c + 0.2 × [c log(q) + (1-c) log(1-q)] + 0.2 × r_format
```

Brier 和对数评分都是严格 proper scoring rule：在能够表达真实概率且按期望优化的理想条件下，
虚报概率不会带来更高期望奖励。实现中对 `q` 限制到 `[0,1]`，对数评分再裁剪到
`[0.01,0.99]`，避免 `log(0)`。

## 训练流程

```text
Qwen3.5-0.8B-Base
        │
        ├─ 280 条格式 SFT，LoRA，2 轮
        │          ↓
        │    同一个 checkpoint-36
        │          │
        ├──────────┼──────────┐
        ↓          ↓          ↓
   正确性 RLOO  Brier-RLCR  对数 RLCR
     100 步      100 步       100 步
        └──────────┼──────────┘
                   ↓
      160 ID 校准 + 160 ID 测试 + 100 OOD 测试
```

三组强化学习必须从完全相同的格式 SFT 适配器和参考适配器开始，才能把差异归因于奖励。
`advantage_estimator=rloo` 表示每个 prompt 的多个采样互相构成 leave-one-out 基线；它运行在
ms-swift 的 GRPO Trainer 中，但优化器采用 RLOO advantage。

## 数据格式

完整字段、各划分条数、OOD 来源和自定义数据示例见
[共用数据说明](../../datasets/confidence_news/README.md)。RL 训练最小格式如下：

```json
{"messages":[{"role":"system","content":"严格输出答案和置信度。"},{"role":"user","content":"新闻：央行公布新的存款利率。"}],"label":"财经","record_id":"my-0001","is_ood":false}
```

这里没有 assistant，因为输出必须由当前策略在线采样。`label` 仅传给奖励函数，不能泄漏到
模型输入。格式 SFT 的 assistant 置信度是与类别和难度无关的均匀占位值，只用于教数值语法；
真实的置信行为来自后续 proper-scoring-rule 强化学习。

## 代码组成

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | 严格拆分 ID/OOD、SFT、RL、校准、测试和 Verifier 数据 |
| `rlcr_rewards.py` | 注册正确性、Brier、对数和格式四个 ms-swift 奖励 |
| `train_format_sft.sh` | LoRA 格式热身 |
| `train_rlcr.sh` | `METHOD=correctness/brier/log` 三种在线训练 |
| `confidence_metrics.py` | ECE、Brier、NLL、AURC、AUROC、Platt 与风险覆盖 |
| `evaluate.py` | 对四个检查点做真实确定性生成，不从训练日志猜结果 |
| `test_rlcr.py` | 数据泄漏、解析、奖励方向和校准边界测试 |
| `run_full.sh` | 从数据到四模型对照的一键入口 |

## 关键参数

### 格式 SFT

| 参数 | 默认值 | 含义与取舍 |
|---|---:|---|
| `tuner_type` | `lora` | 只训练适配器，避免为短格式热身复制完整模型 |
| `lora_rank/lora_alpha` | `16/32` | 约 1082 万可训练参数，占总参数 1.25% |
| `FORMAT_SFT_BATCH` | `16` | 单卡 micro-batch；本机峰值约 46.4 GiB |
| `FORMAT_SFT_LR` | `1e-4` | LoRA 格式学习率 |
| `FORMAT_SFT_EPOCHS` | `2` | 280 条训练数据共 36 个更新步 |
| `max_length` | `768` | 从左侧保留长度范围内的新闻与完整输出 |

### 在线 RLCR

| 参数 | 默认值 | 含义与取舍 |
|---|---:|---|
| `METHOD` | `brier` | `correctness`、`brier` 或 `log` |
| `RLCR_STEPS` | `100` | 每组真实反向传播 100 步 |
| `RLCR_BATCH` | `16` | 一个更新批次含 4 个 prompt 组 |
| `NUM_GENERATIONS` | `4` | 每个 prompt 采 4 个回答，提供组内相对信号 |
| `TEMPERATURE` | `0.7` | 保留类别和置信数值探索；2.0 会大量破坏格式 |
| `RLCR_LR` | `5e-6` | 在线 LoRA 学习率 |
| `RLCR_BETA` | `0.001` | 与冻结参考适配器的 KL 约束强度 |
| `VLLM_MEMORY` | `0.40` | colocate vLLM 的显存比例 |
| `max_completion_length` | `64` | 足够容纳 17-token 固定协议，并限制跑偏长输出 |

100 步的日志显示名义 `epoch=0.4167`。原因是每步 batch 16 包含 4 个不同 prompt、每个各采样
4 次；所以实际处理 400 个 prompt 组和 1600 个 completion，不应误写为“完整遍历 960 条一轮”。

## 运行方法

完整复现：

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
bash course/28_rlcr_confidence/run_full.sh
```

分步运行：

```bash
python course/28_rlcr_confidence/prepare_data.py
bash course/28_rlcr_confidence/train_format_sft.sh
METHOD=correctness bash course/28_rlcr_confidence/train_rlcr.sh
METHOD=brier bash course/28_rlcr_confidence/train_rlcr.sh
METHOD=log bash course/28_rlcr_confidence/train_rlcr.sh
```

一步测试不会冒充正式结果：

```bash
SMOKE=1 METHOD=brier bash course/28_rlcr_confidence/train_rlcr.sh
```

脚本通过 `course/confidence_common.sh` 强制检查 `third_party/ms-swift-official-4.4.3` 必须位于
官方 `v4.4.3` 标签提交 `e1287928be4451b9ed5e2fb00a24ad3c8f61287b`。该标签源码内部版本字符串
显示 `4.5.0.dev0`，所以不能只看 `swift.__version__` 判断版本。

## 如何阅读结果

不要只看分类 Accuracy。置信度至少同时报告：

- ECE：分箱后置信度与实际正确率的平均差，越低越好。
- Brier：`mean((q-c)^2)`，越低越好。
- NLL：对错误高置信尤其敏感，越低越好。
- correctness AUROC：置信度能否把正确样本排在错误样本前，越高越好。
- AURC：风险—覆盖曲线下面积，越低越好。
- coverage/selective risk：按校准集阈值自动处理多少，以及覆盖内错误率。
- OOD false accept rate：域外新闻被高置信接受的比例，越低越好。

本次真实结果见 [RESULTS.md](RESULTS.md)。关键发现是 Brier 奖励明显改善总体校准误差，但贪心
生成时收敛成固定 `0.90`，因而不能区分逐样本难度；这正是下一课训练独立 Verifier 的原因。

## 实验注意事项

1. `temperature=0` 适合最终可重复评测，但不适合 RL rollout；没有采样差异就没有可靠组内 advantage。
2. 只用 `<answer>` 正确性奖励不能训练出概率语义；必须解析并奖励 `<confidence>`。
3. SFT 占位置信度不能按金标签一律写 1.0，否则会先验地教模型永远自信。
4. 正确率很高的小校准集可能一个错误都没有。代码会使用拉普拉斯平滑常数回退，不能在退化数据上
   强行拟合斜率并宣称校准成功。
5. proper scoring rule 保证的是期望层面的诚实报告，不保证小模型一定学会逐样本排序。
6. OOD 不属于四分类答案空间。若训练阶段没见 OOD，不能期待自报置信度自然具备开放集拒绝能力。
7. `outputs/` 被 Git 忽略且保留在 `/mnt/workspace`，检查点不会上传 GitHub；提交的是数据、代码、
   校验和与可审计实测摘要。
