# 第 29 课：独立 Qwen 置信度 Verifier

第 28 课让分类策略自己生成置信度；本课另起一个 `Qwen3.5-0.8B-Base`，把它训练成独立
Reward/Verifier。输入是新闻和分类策略给出的候选类别，输出分数用于估计该候选正确的概率。
它与分类策略没有共享 LoRA，也不读取策略自报的 confidence，因此属于另一条真实参数训练路线。

本课受 [ConfidNet](https://proceedings.neurips.cc/paper_files/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf)
启发。ConfidNet 原论文冻结分类网络并从隐藏表示学习 True Class Probability；这里选择更容易接入
ms-swift、也更独立的 Qwen Reward Model 变体。独立 verifier 对候选答案排序的研究动机还可参考
[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)。本实现不是
ConfidNet 原始网络结构的逐层复刻，课程会明确区分“思想迁移”和“原论文复现”。

## 为什么需要第二个模型

第 28 课的 Brier-RLCR 在温度 0 下把所有 ID 样本都报为 `0.90`：整体 Brier 变好，但对错
AUROC 只有 0.5，OOD 错误接受率 100%。独立 Verifier 能利用另外一组参数和专门构造的困难负例、
真实 OOD 负例，学习“新闻是否支持这个候选类别”，不受策略自我合理化影响。

```text
新闻 + 策略预测类别
          │
          ├─ 拼接 <verdict>CORRECT</verdict>   → RM score_correct
          └─ 拼接 <verdict>INCORRECT</verdict> → RM score_incorrect
                                      │
                                      ↓
                       delta = score_correct - score_incorrect
                       q_raw = sigmoid(delta)
                                      │
                              Platt + 拒答阈值
                                      ↓
                         接受预测 / 交给人工或搜索
```

这里不是用生成提示词问模型“你觉得对吗”。两个 verdict 都是完整待评分序列，训练后的 reward head
分别前向计算标量，差值才是置信依据。

## 真正训练了哪些参数

`train_verifier.sh` 使用：

```text
swift rlhf --rlhf_type rm --tuner_type full
```

ms-swift 为 Qwen 增加一个 `Linear(1024,1)` score head，并用成对 Bradley–Terry 损失令 chosen
分数高于 rejected。Qwen3.5 带有视觉组件，框架默认冻结没有参与本任务的 `model.visual` 和 merger；
日志实测总参数约 852.99M、可训练参数约 752.39M，占 88.21%。文本主干和 score head 都会反向
传播，并非只训练一句提示词或只做 API 调用。

令一对序列奖励为 `r_chosen` 和 `r_rejected`，核心损失为：

```text
L_pair = -log sigmoid(r_chosen - r_rejected - margin)
L = L_pair + 0.01 × L_center
```

`margin=1.0` 来自数据行；center 项抑制两个奖励共同向正/负无界漂移。

## 数据格式

完整划分和扩展规则见 [共用数据说明](../../datasets/confidence_news/README.md)。训练行的最小形式：

```json
{"messages":[{"role":"system","content":"你是独立候选验证器。"},{"role":"user","content":"新闻：央行公布新的存款利率。\n\n待验证的候选类别：财经"},{"role":"assistant","content":"<verdict>CORRECT</verdict>"}],"rejected_response":"<verdict>INCORRECT</verdict>","margin":1.0,"candidate_correct":true,"is_ood":false}
```

`messages` 的 assistant 是 chosen，`rejected_response` 是同一 prompt 的 rejected。对于错误候选，
两者顺序反过来。`candidate_correct`、`gold_label` 等字段只用于评测和审计，不会写入 user 内容。

训练集共 2020 对：960 个 ID 新闻各有金标签正候选和相邻困难错类候选，共 1920 对；另有
100 个真实 OOD 新闻，每个四分类候选都应判错。验证和测试各 420 对，分别由 160×2 个 ID 候选
和 100 个 OOD 候选组成，来源新闻严格分离。

## 代码组成

| 文件 | 作用 |
|---|---|
| `train_verifier.sh` | 以 ms-swift RM 做文本主干全参数训练 |
| `evaluate_verifier.py` | 对 RLCR 真实预测和静态候选做成对奖励前向、校准与拒答评测 |
| `test_verifier.py` | 验证 ID 正负对、OOD 标签、数据泄漏和指标逻辑 |
| `run_full.sh` | 自动复用/生成数据，训练 Verifier 并连接第 28 课 Brier 轨迹 |

## 关键参数

| 参数 | 默认值 | 含义与本机取舍 |
|---|---:|---|
| `rlhf_type` | `rm` | ms-swift 成对 Reward Model 训练 |
| `tuner_type` | `full` | 更新文本主干和 score head |
| `VERIFIER_EPOCHS` | `2` | 让全部 2020 对数据训练两轮 |
| `VERIFIER_BATCH` | `32` | 单卡训练 micro-batch；在本机稳定峰值约 89 GiB |
| `VERIFIER_EVAL_BATCH` | `16` | 防止 ROCm 在验证切回训练时保留峰值缓存 |
| `VERIFIER_LR` | `1e-5` | 全参数学习率 |
| `CENTER_COEF` | `0.01` | 奖励居中正则权重 |
| `max_length` | `768` | 左截断，优先保留候选与 verdict，同时覆盖最多 600 字新闻 |
| `save_steps` | `20` | 每 20 步持久化一个可恢复参数检查点 |
| `torch_dtype` | `bfloat16` | 当前 AMD 卡原生支持 |
| `attn_impl` | `eager` | 避开此 Qwen3.5/ROCm 组合的注意力兼容问题 |

脚本设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。这不是为了减少模型规模，而是缓解
长序列全参数训练在 ROCm 上的显存碎片。

## 运行方法

先确保第 28 课已经生成 Brier 策略轨迹，再运行：

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
bash course/29_independent_confidence_verifier/run_full.sh
```

仅训练模型：

```bash
bash course/29_independent_confidence_verifier/train_verifier.sh
```

一步全参数冒烟：

```bash
SMOKE=1 bash course/29_independent_confidence_verifier/train_verifier.sh
```

中断后可以从已保存参数恢复：

```bash
VERIFIER_RESUME=/绝对路径/checkpoint-80 \
  bash course/29_independent_confidence_verifier/train_verifier.sh
```

当前脚本使用 `save_only_model=true` 节省 100G 持久盘，所以恢复会加载模型和 trainer step，重新创建
优化器；它适合意外中断后的教学续跑，不等价于保存完整 optimizer/scheduler 的逐 bit 无缝恢复。
严肃研究若要完全恢复，应把它改为 `false` 并为 Adam 状态预留数 GB。

手动评测：

```bash
source course/confidence_common.sh
activate_confidence_env
VERIFIER="$(latest_confidence_checkpoint outputs/29_independent_confidence_verifier/verifier_clean)"
python course/29_independent_confidence_verifier/evaluate_verifier.py \
  --verifier "${VERIFIER}" --batch-size 32
```

上面的 `verifier_clean` 是本次仓库作者保留的无恢复歧义正式检查点目录。自己运行 `run_full.sh`
时默认输出到 `outputs/29_independent_confidence_verifier/verifier`，应把命令中的目录相应改回
`verifier`。

## 评测口径

评测脚本读取第 28 课 `brier_rlcr.jsonl` 的真实策略预测，但 Verifier 输入只含新闻和
`predicted_label`，不含金标签、是否正确或策略自报置信度。它同时报告：

- 160 条 ID 测试上的 ECE、Brier、NLL、AURC、对错 AUROC；
- 校准集选出的拒答阈值、自动覆盖率和覆盖内风险；
- 100 条 OOD 测试的错误接受率；
- 420 对静态候选的 verdict 排序准确率和 OOD 错误检测率；
- 每条预测两次 Verifier 前向的 token 数和延迟。

静态候选测试验证 RM 是否学会训练任务；连接策略测试才回答“它能否给真实分类预测提供置信度”。
两者不能只选较高的一个报告。

## 显存实验与注意事项

正式训练前真实试过 batch 128、96、64、48：

- 128 和 96 在第一步反向传播接近 185～186 GiB 时 OOM；
- 64 训练到第 12 步后因碎片 OOM，日志峰值约 169.1 GiB；
- 48 首轮训练稳定、验证对准确率 98.83%，但大验证 batch 在切回第二轮时触发峰值 OOM；
- 最终默认采用训练 32、验证 16，牺牲少量吞吐换取可复现稳定性。

这说明“显存还有空余”不能只看某一瞬间：全参 Adam、两个偏好序列、最长 batch、验证缓存、
内核临时区和保存检查点会在不同阶段形成峰值。课程保留这些失败记录，因为它比给出未经验证的
大 batch 更有教学价值。

最终实测指标见 [RESULTS.md](RESULTS.md)。
