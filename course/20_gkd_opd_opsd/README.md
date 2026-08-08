# GKD、OPD-RL 与 OPSD 蒸馏路线

这三种方法都让学生对齐教师分布，但教师数据来源和损失位置不同。本节补上独立 OPSD 实验，并把仓库已有的 GKD、OPD-RL 课程串成完整路线。

## 三者区别

| 方法 | 学生回答来源 | 教师是谁 | 教师信号如何进入训练 |
|---|---|---|---|
| GKD | 离线标签或在线学生 rollout | 独立教师或固定教师 | token 分布散度直接作为 loss |
| OPD-RL | 在线学生 rollout | 独立教师/服务 | 教师与学生 log-ratio 加到逐 token advantage |
| OPSD | 在线学生 rollout | 同一模型在特权 prompt 下的分布 | 可走 GKD loss，也可走 OPD-RL advantage |

仓库第 06 节已经实现离线 GKD，第 04 节已经实现纯教师 OPD-RL，第 05 节进一步实现多教师 OPD。它们都做过真实训练，不在本节复制同一脚本。

## OPSD 数据格式

文件 `datasets/alignment_news/opsd_*.jsonl`：

```json
{"messages":[{"role":"user","content":"请判断新闻类别……"}],"teacher_prompt":"原问题……\n\n特权参考信息：人工标准类别是‘体育’。","label":"体育"}
```

学生只编码 `messages`；教师侧用同一模型、同一当前权重编码 `teacher_prompt`。`teacher_prompt` 必须是 JSON 顶层字段。自定义数据可放参考解答、检索证据或环境完整状态，但不能在部署时偷偷传给学生。

## 运行 OPSD

```bash
bash course/10_alignment_data/prepare_data.sh
bash course/11_sft_dft/train_sft.sh
bash course/20_gkd_opd_opsd/train_opsd.sh
```

本脚本选择 GKD+OPSD 动态模式：不传 `teacher_model`，因此教师是当前模型在特权上下文下的分布；`lmbda=1` 表示用学生在线 rollout，`beta=0.5` 使用 JSD。若设 `beta=0` 是 forward KL，设 `beta=1` 是 reverse KL。

## GKD 参数复习

- `lmbda=0`：纯离线数据响应，见第 06 节。
- `lmbda=1`：完全使用学生在线生成，本节 OPSD 采用此值。
- `sft_alpha`：额外监督交叉熵，本节没有固定回答，所以设 0。
- `gkd_logits_topk`：只蒸馏教师 top-k logits，可省显存但损失尾部分布信息。

## OPD-RL 参数复习

在 `rlhf_type=grpo` 下设置 `teacher_model` 或 `teacher_model_server`，ms-swift 就自动启用 OPD-RL。逐 token 信号是：

```text
A_t = A_base + teacher_kl_coef × (log p_teacher(y_t) - log p_student(y_t))
```

不设置 reward 时是纯蒸馏；同时设置自定义 reward 时是任务 RL 与蒸馏混合。重点监控 `teacher_kl`、completion length 和任务准确率。REAL/FIPO 这类需要序列标量 advantage 的 loss 与 OPD-RL 教师信号不兼容，源码会直接报错。

## 性能注意

OPSD 的一次更新要做学生 rollout、学生前向和特权教师前向。脚本用 `VLLM_MEMORY=0.50`、正式 `OPSD_BATCH=64` 和 24 token 短回答提高本机利用率，并明确使用 left 截断，确保长样本保留末尾教师回答。冒烟集只有 16 条，而 GKD 会丢弃不足一个 batch 的尾批，因此脚本自动用 16。教师上下文如果比学生长很多，显存开销会由教师侧主导，应按最长 `teacher_prompt` 估算 batch，而不是只看学生 prompt。

本机 4 步复测中，batch=64 总耗时 57.9 秒、峰值 109.6 GiB、吞吐 4.42 样本/秒。继续加到 128 时，前两步峰值只升到 119.7 GiB，但不同序列形状触发大量 Triton 自动调优，平均约 90 秒/步，吞吐明显更差，因此课程保留 64。显存占用率只是约束，不是优化目标；应以完整运行的样本吞吐为准。

当前 ROCm vLLM 0.26 单进程执行器会直接读取 `RANK`等 torchrun 环境变量，GKD CLI 没有自动补齐。脚本只在变量不存在时设置单卡默认值，多卡 torchrun 传入的值不会被覆盖。
