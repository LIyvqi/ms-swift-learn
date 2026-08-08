# PPO：奖励模型驱动的在线对齐

PPO 是经典 RLHF 的在线阶段。当前策略生成回答，RM 给终局奖励，价值模型估计回报，参考策略提供 KL 约束；因此它比 DPO 占用更多显存，也更容易因奖励模型偏差而不稳定。

## 前置与运行

```bash
bash course/11_sft_dft/train_sft.sh
bash course/11_sft_dft/merge_sft.sh
bash course/13_reward_model/train.sh
bash course/14_ppo/train.sh
```

依赖关系如下：训练策略加载 SFT adapter；PPO 的 LoRA 参考策略由 PEFT 的 adapter-disable 路径提供，因此不能传当前版本不支持的 `ref_adapters`；奖励模型与价值模型都由完整 RM checkpoint 初始化。当前 ms-swift 要求它们使用同一模型系列与 tokenizer，本课程全部基于同一个 Qwen3.5-0.8B-Base。

## 数据格式

`prompts_*.jsonl` 只包含输入，绝不能带 assistant 标准答案：

```json
{"messages":[{"role":"system","content":"任务约束"},{"role":"user","content":"请判断新闻类别……"}],"label":"体育"}
```

顶层 `label` 不会直接参与 PPO 损失；它保留下来用于统一生成评测。回答奖励来自第 13 节 RM。

## 关键参数

- `PPO_BATCH=64`：正式训练的 rollout 批量；冒烟模式自动改为 8。
- `ROLLOUT_BATCH=64`：生成/前向评分时的本地批量，优先用它吃满空余显存。
- `PPO_MAX_COMPLETION_LENGTH=24`：四分类答案只需少量 token。32 token 的首轮实测耗时 42 秒；16 token 虽更快，却恰好截断 `\boxed{类别}` 的右花括号，独立评测格式率降到 0%。因此默认取 24，在吞吐与完整格式之间留出余量。开放式任务应按答案长度分布重新设置。
- `PPO_SAVE_STEPS=4`：每四次 PPO 更新保存一次，避免每步都写 LoRA checkpoint 拖慢教学实验。
- `truncation_strategy=left`：PPO 数据只有 prompt，必须保留末尾的用户约束和 assistant 起始标记。用 right 截断长新闻会让模型续写正文；成对偏好算法也要保留末尾回答，因此本课程的 RLHF 脚本统一用 left。
- `num_ppo_epochs=4`：同一批 rollout 重用四轮；过大容易对旧样本过拟合。
- `num_mini_batches=4`：一次 PPO 更新拆成四个小批。
- `PPO_KL=0.1`：惩罚偏离 SFT 参考策略。旧默认 0.05 的三轮实验 KL 达到 13.19 并出现格式退化；配合增强 RM 的一轮对照最终 KL 为 5.605，因此把 0.1 设为默认。
- `cliprange=0.2`：限制策略概率比变化。
- `vf_coef=0.1`、`cliprange_value=0.2`：价值损失权重和裁剪。
- `gamma=1.0`、`lam=0.95`：回报折扣与 GAE 平滑系数。
- `whiten_rewards=true`：批内标准化奖励，适合本教学小数据，但极小 batch 下会增加噪声。

## 效率与稳定性

PPO 同时驻留策略、参考、RM 和价值模型。0.8B 模型在约 192 GiB 显存上默认 batch 已明显高于官方保守示例。若显存利用仍低，先提高 `ROLLOUT_BATCH`，再提高 `PPO_BATCH`；若生成阶段 OOM，反向调整。PPO 在 ms-swift 中不应传 `max_grad_norm`，这是当前实现的兼容限制。

TRL PPO 不用 Transformers Trainer 的 `max_steps` 直接控制主循环，而是由 `num_train_epochs × 数据量` 换算 `total_episodes`。脚本已将 `STEPS` 换算为对应轮数；冒烟模式只做一次 8 token 短 rollout。

除奖励均值外，应同时观察 KL、策略 clip fraction、value loss、回答长度和真实分类准确率。RM 奖励上升但真实准确率下降，就是奖励投机而不是有效对齐。

## 本机三轮结果

初版 RM 只含错误类别负例。用它做 batch=64、16 token 的 3 轮 PPO，共 12 次更新和 768 条 rollout，耗时 6 分 49 秒，外部显存峰值 108.50 GiB；最后 KL 达 13.19。即使把公平评测上限提高到 24 token，独立验证准确率也只有 96.88%，严格格式率为 0%，模型会主动在右花括号前结束。这证明只看 RM 分数会选到奖励投机 checkpoint。

修正版 RM 同时含错误类别和残缺格式负例，并统一使用 left 截断。用 `PPO_KL=0.1`、24 token 训练 1 轮/4 次更新，耗时 2 分 24 秒，框架报告峰值 59.45 GiB，最后 KL 为 5.605；64 条贪心评测恢复到 98.44% 准确率与 100% 格式率，与 SFT 持平。训练中的随机高温 rollout 仍偶尔产生异常字符串，所以这不是“RM 已解决所有格式漏洞”，而是说明困难负例、KL 和独立评测三者都不可少。
