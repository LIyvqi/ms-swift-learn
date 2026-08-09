# 第 23 课：JitRL 推理期持续强化学习

这一课复现论文 **Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates** 的核心算法。它与前面所有训练课最大的不同是：模型参数始终冻结，不建立优化器，不调用反向传播；Agent 只在推理时检索过去经验，并实时修正当前候选动作的 logits。

- 论文：<https://arxiv.org/abs/2601.18510>
- 官方实现：<https://github.com/liushiliushi/JitRL>
- 本地实验模型：`models/Qwen3.5-0.8B-Base`
- 模型加载：`swift.get_model_processor`
- 实测报告：[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)
- 已部署模型 API 接入：[API_DEPLOYMENT.md](API_DEPLOYMENT.md)

## 本课复现了什么

本课严格实现了论文最关键的闭式策略更新：

```text
经验记忆：M = {(s_i, a_i, G_i)}
状态价值：V(s) = 邻居回报的平均值
动作价值：Q(s,a) = 同动作邻居回报的平均值
优势：A(s,a) = Q(s,a) - V(s)
归一化优势：A_norm = A / (max(|A|) + epsilon)
推理期修正：z'(s,a) = z(s,a) + beta * A_norm(s,a)
最终策略：pi(a|s) = softmax(z'(s,a) / temperature)
```

每局结束后，从最后一步向前计算：

```text
G_t = r_t + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
```

`z(s,a)` 不是人工概率。本课把三个候选动作编号为 `1/2/3`，验证三个编号在当前 Qwen tokenizer 中均为单 token，然后直接读取模型最后位置上对应 token 的原始 logits。这样实现的是论文公式里的直接 logits 更新，不需要 API 模型用自然语言报告置信度。

## 与官方完整基准的边界

官方仓库提供 Jericho 和 WebArena 实验，本课目前是“核心算法复现”，不是论文完整 benchmark 复跑：

- 使用本地 Qwen3.5-0.8B-Base，而不是论文中的 API Agent 模型；
- 使用小型确定性多阶段文本环境，方便在教学机器上快速观察连续学习曲线；
- 状态检索使用论文支持的文本 Jaccard，相似状态并列时优先较新的经验；
- 所有合法动作始终在候选集合里，因此不需要额外生成“仅来自记忆”的动作；
- 环境奖励由确定性评估器给出，不使用第二个 LLM 充当反思评估器。

这些简化没有改变经验回报、非参数价值估计、优势归一化和 logits 闭式修正四个核心环节。后续可以在同一个 `经验记忆` 接口上增加 Jericho 或 WebArena 适配器。

## 教学环境

`protocol_env.py` 定义一个四阶段物流恢复任务。每个阶段必须从以下三个协议中选择一个：

```text
琥珀协议、靛蓝协议、白银协议
```

正确映射由固定任务种子生成，但不会写进模型提示。选错立即结束本局并获得 `-1`；通过普通阶段获得 `0.5`；通过最终阶段获得 `2.0`。因此冻结模型第一次遇到任务时只能探索，后续只能依靠保存下来的奖励经验改善行为。

设备批次会变化，但不决定正确动作。经验检索键主动去掉批次噪声、保留阶段和进度，演示真实 Agent 中很重要的“状态抽象”。模型仍然看到完整状态。

为了提高实验速度，环境只有 `4 个阶段 × 4 个批次 × 6 种动作排列 = 96` 个不同决策提示。脚本用一个大 batch 预计算所有真实基础 logits；后面的 5 个随机种子和多个 beta 共享这份冻结策略缓存。这不是伪造 logits，而是利用冻结模型对相同输入输出不变的性质消除重复前向。

## 经验数据通用格式

持续记忆保存为 UTF-8 JSONL：一行是一条 `(状态, 动作, 回报)` 经验。字段必须如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `state` | 字符串 | 可检索的状态表示；应保留决策信息并尽量去掉随机 ID 等噪声 |
| `action` | 字符串 | 当时实际执行的动作，必须能映射回环境合法动作 |
| `return_value` | 浮点数 | 从该步开始的折扣回报 `G_t`，不是只看当前一步的即时奖励 |
| `episode` | 整数 | 经验来自第几局，用于审计和时间排序 |
| `step` | 整数 | 经验在该局中的步号，从 0 开始 |

一条可直接加载的例子：

```json
{"state":"任务 物流恢复 阶段 入口校验 进度 0","action":"琥珀协议","return_value":1.5,"episode":7,"step":0}
```

自定义数据时最常见的错误是把即时奖励 `r_t` 直接填进 `return_value`。若你记录的是整条轨迹，请调用 `经验记忆.添加轨迹(states, actions, rewards, gamma, episode)`，让代码统一计算每一步回报。

## 输出结果格式

完整结果默认写入 `outputs/23_jitrl/run_时间/result.json`，并更新 `outputs/23_jitrl/latest_result.json`。`outputs/` 被 Git 忽略，但位于 `/mnt/workspace` 下，机器重启后仍能保留。

结果顶层字段：

| 字段 | 含义 |
|---|---|
| `metadata` | 模型、PyTorch、耗时、显存、参数指纹和零训练检查 |
| `arguments` | 本次实验的全部命令行参数 |
| `environment` | 合法动作、隐藏正确映射和预计算状态数 |
| `base_logit_statistics` | 冻结模型三个候选动作的 logits 跨度 |
| `settings` | 静态基线及每个 beta 的汇总、逐种子与逐回合轨迹 |

每个 JitRL 设置还会产生 `memory_beta*_seed*.jsonl`。以后在单随机种子实验中可恢复它：

```bash
python course/23_jitrl/run_experiment.py \
  --seeds 11 \
  --resume-memory outputs/23_jitrl/run_时间/memory_beta8_seed11.jsonl
```

恢复的经验会作为新实验的初始记忆，但不会修改模型权重。

## 运行方法

默认命令先跑数学单元测试，再运行 60 局、5 个随机种子、`beta=2/4/8` 的真实模型实验：

```bash
cd /mnt/workspace/ms-swift-learn
bash course/23_jitrl/run.sh
```

复现实测报告中的 100 局设置：

```bash
source ./activate.sh
python course/23_jitrl/test_closed_form.py
python course/23_jitrl/run_experiment.py \
  --episodes 100 \
  --seeds 11,22,33,44,55 \
  --betas 2,4,8 \
  --unseen-probability 0.05 \
  --batch-size 96
```

若模型已经部署为 OpenAI 兼容 API，无需把权重加载进 Agent 进程。先阅读 [API 部署与接入教程](API_DEPLOYMENT.md)，本课提供 `constrained_logprobs`、`top_logprobs` 和 `verbalized` 三种兼容模式。当前 ms-swift/vLLM 服务已完成真实 HTTP 实测：

```bash
# 终端一
PORT=8000 bash course/23_jitrl/serve_api.sh

# 终端二
bash course/23_jitrl/run_api.sh \
  --base-url http://127.0.0.1:8000/v1 \
  --api-model Qwen3.5-0.8B-Base
```

高探索对照：

```bash
python course/23_jitrl/run_experiment.py \
  --episodes 100 \
  --seeds 11,22,33,44,55 \
  --betas 2,4,8,12 \
  --unseen-probability 0.35 \
  --output-dir outputs/23_jitrl_high_exploration
```

## 参数解释

| 参数 | 默认值 | 作用与注意事项 |
|---|---:|---|
| `--episodes` | 60 | 连续交互局数；它不是梯度训练 step |
| `--seeds` | 5 个 | Agent 采样和环境动作排列种子，用多种子避免偶然结论 |
| `--betas` | `2,4,8` | 经验优势相对基础 logits 的强度；过小压不过模型先验，过大会忽略先验 |
| `--temperature` | 0.8 | 修正后策略的采样温度 |
| `--gamma` | 0.5 | 跨步骤信用分配的折扣因子 |
| `--top-k` | 10 | 每次用于估计价值的相似经验数 |
| `--similarity-threshold` | 0.95 | 文本 Jaccard 检索阈值；当前结构化检索键接近精确阶段匹配 |
| `--unseen-probability` | 0.05 | 以乐观值探索未见动作的概率；越大越愿意持续试错 |
| `--optimism-alpha` | 5.0 | 未见动作乐观奖励，实际增量还会除以邻居数 |
| `--batch-size` | 96 | 预计算基础 logits 的批大小；当前 GPU 实测可一次跑完 |
| `--task-seed` | 2026 | 决定隐藏的阶段—正确动作映射 |

论文的超参数不是跨环境通用常数。应先打印基础候选 logits 的跨度，再调 beta；还要一起观察前 10 局和后 10 局成功率，前者反映探索速度，后者反映是否仍在过度探索。

## 如何换成自己的 Agent 环境

只需保留以下边界：

1. 环境在每一步提供文本状态和有限候选动作；
2. 策略能够为每个候选动作给出同一尺度上的 logit；
3. 一局结束后，评估器给出逐步奖励或能转换成逐步奖励的信息；
4. 把轨迹交给 `经验记忆.添加轨迹`；
5. 下一步调用 `修正动作_logits`，再对修正值做 softmax 和采样。

若动作是多 token 文本，不应直接比较最后一个 token。可以把动作约束成单 token 编号，或用动作完整序列的条件对数概率作为 `z(s,a)`。不同长度动作建议用平均 token logprob，避免天然偏向短动作。

## 实验注意事项

- 不要在 JitRL 循环里创建优化器，也不要调用 `loss.backward()`；本课会检查 `requires_grad`、参数版本号和抽样指纹。
- JitRL 学到的是外部经验记忆，不是写进权重的永久技能。删除 JSONL 记忆后，策略会回到静态模型。
- 检索质量决定价值估计质量。把用户 ID、时间戳等噪声原样放进状态，可能导致真正相似的经验无法命中。
- 高 `unseen_probability` 有利于探索，但会让已学会的任务继续试错。本课实测 `0.35` 后期明显不如 `0.05`。
- 记忆会持续增长。长期服务应增加容量上限、去重、质量过滤和隐私清理策略。
- 本课是离散候选动作复现。开放文本 Agent 需要先生成候选集合，再进行同样的 logits 修正。

## 文件说明

| 文件 | 用途 |
|---|---|
| `jitrl_core.py` | 经验 JSONL、Jaccard 检索、回报、价值、优势与闭式修正 |
| `protocol_env.py` | 四阶段文本 Agent 环境和模型提示 |
| `run_experiment.py` | ms-swift 模型加载、真实 logits、对照实验和零参数更新验证 |
| `experiment_common.py` | 本地权重与 API 共用的 Agent 交互循环 |
| `api_policy.py` | OpenAI 兼容 API 的三种候选动作打分方式 |
| `run_api_experiment.py` | 已部署模型 API 的多种子对照实验 |
| `serve_api.sh` / `run_api.sh` | 启动 ms-swift 服务与运行 API 实验 |
| `API_DEPLOYMENT.md` | API 能力分级、部署命令、密钥和生产注意事项 |
| `test_closed_form.py` | 闭式公式、检索、折扣回报与持久化单元测试 |
| `test_api_policy.py` | API logprobs、缺项报错与文本置信度单元测试 |
| `run.sh` | 激活持久化环境并依次运行测试和实验 |
| `EXPERIMENT_RESULTS.md` | 100 局、5 随机种子的实测报告 |
