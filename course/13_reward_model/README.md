# RM：训练奖励模型

Reward Model 把“chosen 比 rejected 好”的成对标注变成一个标量评分器。它是第 14 节 PPO 的奖励来源，也可以独立用于回答排序。

## 数据格式

RM 使用 `rm_*.jsonl`。它沿用 DPO 的成对格式，但每个 prompt 同时构造“错误类别”和“类别正确但格式残缺”两种 rejected：

```json
{"messages":[{"role":"user","content":"请判断新闻类别……"},{"role":"assistant","content":"\\boxed{体育}"}],"rejected_response":"\\boxed{财经}","margin":1.0}
```

格式困难负例示例：chosen 为 `\boxed{体育}`，rejected 为缺少右花括号的 `\boxed{体育`。初版 RM 只有错误类别负例，PPO 虽保持较高分类准确率，却把严格格式率从 100% 降到 0%；这个负例正是根据独立评测补入的。

ms-swift 在语言模型顶部增加单值 value head，分别计算 `r_chosen` 和 `r_rejected`。若提供 `margin`，损失要求 `r_chosen-r_rejected` 超过该间隔。自定义数据的 margin 应表达标注置信度或质量差距，不能随意设成很大的数。

## 运行

```bash
bash course/11_sft_dft/train_sft.sh
bash course/11_sft_dft/merge_sft.sh
bash course/13_reward_model/train.sh
```

输出 checkpoint 除 LoRA 外还应包含 value head 权重；第 14 节会自动寻找最新 RM 检查点。

## 关键参数

- `center_rewards_coefficient=0.01`：惩罚 chosen 与 rejected 奖励整体漂移，防止只抬高两个分数而不改善间隔。
- `RM_BATCH=128`：一条样本编码两个回答；RM 不计算全词表语言模型 logits，实测显存远低于 DPO，所以单独提高批量。
- `RM_LR=1e-5`：从合并后的 SFT 模型做全参数 RM 训练。
- `tuner_type=full`：确保 Qwen3.5 新初始化的 `score` 奖励头和主干一起保存。当前本机 ms-swift 开发版在“已有 CausalLM LoRA 转 seq_cls LoRA”时不会保存新 score head，因此课程不用那条有数据丢失风险的路径。
- `margin`：来自数据顶层，本课程固定为 1.0。

## 指标解释

- `rewards/accuracies`：奖励模型把 chosen 排在 rejected 前面的比例，是最直观指标。
- `rewards/margins`：两个评分的平均差值。
- `rewards/chosen`、`rewards/rejected`：绝对值不必固定，但不应一起无界漂移。
- `center_rewards_loss`：中心约束的贡献。

训练准确率高不代表能泛化到当前策略生成的新回答，这是 PPO 中常见的 reward hacking 来源。自己的开放式数据应留出按 prompt 划分的验证集，并加入长度、格式和安全性不同的困难负例。

## 本机三轮结果

初版 256 对数据只含错误类别负例：batch=128、384 token 共训练 3 轮/6 步，验证偏好准确率 96.88%，但后续 PPO 的严格格式率降为 0。增强版扩展到 512 对，错误类别与残缺格式各半；采用 left 截断保留回答，3 轮/12 步耗时 3 分 23 秒，框架峰值 22.15 GiB，最终混合验证准确率 95.31%、奖励间隔 2.041。曾错误使用 right 截断时只有 62.5%，因为回答尾部被裁掉；该对照说明截断策略本身就是数据质量的一部分。最终 `model.safetensors` 已直接检查到 `score.weight`。
