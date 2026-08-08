# 人类偏好对齐课程实测报告

本报告记录第 10～22 节在当前 AMD MI308X、Qwen3.5-0.8B-Base 和 ms-swift 源码环境中的真实运行结果。训练产物位于被 Git 忽略的 `outputs/`；表中数值来自各运行的 `logging.jsonl` 或 `evaluation.json`，不是手工估算。

## 冒烟链路

| 方法 | 实际批量/轨迹 | 单步结果 | 框架报告峰值显存 |
|---|---:|---:|---:|
| SFT | 16 条 | loss 0.7647 | 21.07 GiB |
| DFT | 16 条 | loss 0.07995 | 21.07 GiB |
| DPO | 16 对 | loss 0.7702 | 46.39 GiB |
| RM 全参数 | 16 对 | loss 1.3020，偏好准确率 62.5% | 6.03 GiB |
| PPO | 1 次更新 | policy loss 0.00671，value loss 2.353 | 39.94 GiB |
| KTO | 32 条点偏好 | loss 0.5000 | 47.64 GiB |
| CPO | 16 对 | loss 1.3413，偏好准确率 87.5% | 72.81 GiB |
| SimPO | 16 对 | loss 0.8695，偏好准确率 87.5% | 65.14 GiB |
| ORPO | 16 对 | loss 0.8169，偏好准确率 87.5% | 72.81 GiB |
| GRPO | 32 条 rollout | reward 0.1563 | 137.40 GiB |
| DAPO | 32 条 rollout | loss 0.000632 | 137.39 GiB |
| GSPO | 32 条 rollout | reward 0.1563 | 137.39 GiB |
| OPSD | 16 条 rollout | loss 0.02670 | 101.45 GiB |
| Rewards-as-Labels REAL | 32 条 rollout | loss 2.1808 | 137.40 GiB |

冒烟数据只证明数据编码、模型加载、损失、反向传播与 checkpoint 保存链路正常，不能用单步 loss 排算法名次。GRPO 首次单步耗时约 45 秒，其中包含 Triton/TileLang 自动调优；缓存建立后 DAPO/GSPO 同规模单步约 15～16 秒。

## Regression-Aware REAL 最小实测

2 个 prompt × 2 条 rollout、24 token 的最小冒烟已完成生成、RLOO、CoT 策略梯度、回归/NLL 精修和 RAIL 评测，峰值显存 6.54 GiB。

| 指标 | 训练前 | 1 步后 |
|---|---:|---:|
| MSE | 1.2017 | 1.0450 |
| MAE | 0.9569 | 0.9056 |
| Pearson | 0.9134 | 0.9072 |
| 格式率 | 100% | 100% |
| 平均期望分数 | 2.4422 | 2.5204 |

一步后 MSE/MAE 改善说明梯度方向与保存链路正常；Pearson 的微小波动和 20% 四舍五入准确率在一步实验中没有质量意义。正式脚本已把默认提到 16 prompt × 4 rollout，即每步 64 条轨迹。

## 多轮正式训练

| 方法 | 轮数/步数 | 最终验证指标 | 总耗时 | 采样峰值 |
|---|---:|---:|---:|---:|
| SFT，batch=48 | 3 轮 / 18 步 | loss 0.0124，token accuracy 99.78% | 3 分 42 秒 | 165.17 GiB |
| DFT，batch=48 | 3 轮 / 18 步 | loss 0.00156，token accuracy 99.33% | 1 分 20 秒 | 174.66 GiB |
| DPO，batch=48，384 token，left 截断 | 3 轮 / 18 步 | loss 0.3592，preference accuracy 98.96%，reward margin 1.094 | 2 分 4 秒 | 143.20 GiB |
| RM，batch=128，384 token | 3 轮 / 6 步 | loss 0.1427，preference accuracy 96.88%，reward margin 4.402 | 3 分 34 秒 | 39.17 GiB |
| PPO，batch=64，16 token | 3 轮 / 12 次更新 | RM score 2.469，RLHF reward 1.809，value loss 0.0779，KL 13.19 | 6 分 49 秒 | 108.50 GiB |
| 增强 RM，batch=128，left 截断 | 3 轮 / 12 步 | loss 0.6879，preference accuracy 95.31%，reward margin 2.041 | 3 分 23 秒 | 框架 22.15 GiB |
| 增强 RM-PPO，batch=64，24 token，left 截断 RM | 1 轮 / 4 次更新 | RM score 3.062，RLHF reward 2.502，KL 5.605；独立准确率 98.44%，格式率 100% | 2 分 24 秒 | 59.45 GiB |
| KTO，batch=64，384 token，left 截断 | 3 轮 / 24 步 | loss 0.2256，reward margin 2.674 | 2 分 51 秒 | 98.04 GiB |
| CPO，batch=32，384 token，left 截断 | 3 轮 / 24 步 | loss 0.2786，preference accuracy 100%，reward margin 1.457 | 1 分 41 秒 | 167.20 GiB |
| SimPO，batch=32，384 token，left 截断 | 3 轮 / 24 步 | loss 0.05569，preference accuracy 100%，reward margin 4.938 | 1 分 40 秒 | 149.09 GiB |
| ORPO，batch=32，384 token，left 截断 | 3 轮 / 24 步 | loss 0.03371，preference accuracy 100%，reward margin 0.1335 | 1 分 42 秒 | 151.42 GiB |
| GRPO，batch=32，8 候选，24 token | 4 步 | 最后 reward 0.6625，KL 0.000806，稳态约 6.3 秒/步 | 39.4 秒 | 129.24 GiB |
| DAPO，batch=32，8 候选，24 token | 4 步 | 最后 reward 0.5875，train loss -0.04996，稳态约 3.8 秒/步 | 29.5 秒 | 129.23 GiB |
| GSPO，batch=32，8 候选，24 token | 4 步 | 最后 reward 0.6563，序列裁剪比例 0.3683，稳态约 3.8 秒/步 | 29.3 秒 | 129.23 GiB |
| OPSD，batch=64，24 token，left 截断 | 4 步 | train loss 0.008591，吞吐 4.42 样本/秒 | 57.9 秒 | 109.6 GiB |
| Rewards-as-Labels REAL，batch=32，8 候选 | 4 步 | 最后 loss 2.075，reward 0.6625，KL 0.00110 | 30.4 秒 | 129.2 GiB |

GRPO 系列的 reward 来自不同随机 batch 和高温 rollout，不能把最后一步数值当作算法排行榜；这组四步复测主要用于确认多步更新稳定性、裁剪行为与显存效率。

OPSD 的 batch=128 压测在 2 步后停止：峰值 119.7 GiB，但动态形状反复触发 Triton 自动调优，平均约 90 秒/步，明显慢于 batch=64。最终默认根据吞吐选择 64，而不是只追求更高显存占用。

## 显存压力测与最终选择

- SFT `batch=96` 和 64 在普通批次分别约 116 GiB 和 76 GiB，但都在第二轮的更长新闻批次出现 191.6 GiB 短暂峰值并 OOM。最终默认回调为 48，不用单个短批次代表整个数据分布。
- 在线 GRPO 系列使用 `VLLM_MEMORY=0.50`，训练器实测约 137 GiB，对 191.7 GiB 显存保留约 50 GiB 的长样本余量。
- CPO/ORPO 在冒烟集只有 16 对时已约 73 GiB，正式 `batch=32` 预计接近 140～150 GiB，因此没有盲目调到 64。
- DPO 在 768 token 下，batch=48 在第一轮后触发 191.64 GiB 峰值，batch=32 也在第 15/24 步因临时 FP32 输出峰值 OOM。对齐阶段因此统一改为 384 token；这保留新闻标题和正文前部，而不是单纯继续缩小 batch。
- DPO 在 left 截断下 batch=64 的单步峰值只有 128.49 GiB，但正式第 4 步达到 182.34 GiB 后还需申请 26.88 GiB并 OOM；batch=48 三轮以 143.20 GiB 峰值完整结束，所以课程默认定为 48。
- RM 不产生全词表语言模型 logits，batch=128 的全参数训练外部峰值仍只有 39.17 GiB；但 256 条数据下再提高到 256 会让每轮只剩一次更新，所以保留 128 以兼顾吞吐与教学曲线。
- PPO 的 32-token 首步耗时约 42 秒；改为 16 token 后稳态约 34 秒/步，提速约 19%，完整运行峰值 108.50 GiB，但 64 条独立评测的严格格式率从 SFT 的 100% 降到 0%，原因是右花括号被截断。最终默认折中为 24 token，不能用破坏输出约束换取表面吞吐。
- 24-token 公平复测确认初版 PPO 会主动提前结束：准确率 96.88%、格式率 0%，不是评测长度伪影。RM 加入残缺格式困难负例，并把 KL 系数提高到 0.1 后，一轮 PPO 恢复为 98.44% 准确率和 100% 格式率；课程默认采用这组更保守配置。
- 峰值以框架 CUDA/ROCm 内存统计为主；按秒采样的 `rocm-smi` 可能漏掉短暂峰值，因此不用它覆盖框架报告。

## 已验证的框架边界

1. RM 用 LoRA 直接从 causal-LM adapter 继续时，当前组合不会可靠保存新的 `score.weight`。课程改为先合并 SFT LoRA，再全参数训练 RM，并已检查 safetensors 内存在 score head。
2. PPO 主循环按 `total_episodes` 而非 Trainer `max_steps` 计算；脚本已换算 `STEPS`。
3. KTO 循环错位 batch 构造 KL 对照；四分类重复短答案必须去重，本课程使用 prompt 中约定的唯一记录编号。
4. 当前 ms-swift 4.5.0.dev0 原生 REAL 与动态 OPSD 预探测冲突；第 21 节用进程内兼容插件绕过，未修改 `third_party` 源码。
5. ROCm vLLM 0.26 的 GKD 入口需要单卡 `RANK/WORLD_SIZE/MASTER_*` 环境变量；脚本只在外部未设置时补齐。
