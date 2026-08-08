# 同名方法一：ms-swift 原生 Rewards-as-Labels REAL

本节专门防止同名混淆。ms-swift 的 `--loss_type real` 对应论文《Rewards as Labels: Revisiting RLVR from a Classification Perspective》，不是用户指定的 Regression-Aware REAL。后者在第 22 节。

## 核心思想

原生 REAL 不再用标量 advantage 直接缩放每个 token，而是把组内正奖励回答视为正类、非正奖励回答视为负类。每条回答的分数是当前策略相对旧策略的平均 log-ratio：

```text
s = mean_t(log π_current(y_t) - log π_old(y_t))
```

正样本希望 s 增大，负样本希望 s 减小，并用 `real_tau` 控制分类边界温度。它试图修复 GRPO 中“低置信正确 token 梯度反而较弱”和“少数负样本梯度主导”的问题。

## 数据格式

与 GRPO 相同，只放 prompt 和自定义 reward 需要的顶层 `label`：

```json
{"messages":[{"role":"user","content":"请判断新闻类别……"}],"label":"体育"}
```

每个 group 必须同时有正、负样本，否则该 group 的分类 loss 为 0。因此 `num_generations=8`、适当 temperature 和有区分度的 reward 非常重要。

## 运行

```bash
bash course/11_sft_dft/train_sft.sh
bash course/21_real_rewards_as_labels/train.sh
```

`REAL_BATCH=32` 必须能被 8 个 generation 整除。`scale_rewards` 会被 ms-swift 强制设为 `none`，因为该损失直接把 reward 符号当标签，不能再做 GRPO 归一化。`real_tau=0.5` 越小分类边界越尖锐，但梯度也更敏感。

当前本地 ms-swift 4.5.0.dev0 有一个动态 OPSD 探测边界：即使 batch 中没有 `teacher_prompt`，初始化时也会预标记为有教师，而原生 REAL 不支持教师分支。`course/plugins/real_loss_compat.py` 只在本训练进程中关闭未使用的动态 OPSD，不修改 `third_party` 源码，也不影响第 20 节。升级到已修正的 ms-swift 后可以移除该兼容插件。

本机用 batch=32、每组 8 个候选连续训练 4 步，总耗时 30.4 秒、峰值 129.2 GiB，最后 loss 2.075、reward 0.6625、KL 0.00110。高温 rollout 的 batch 难度不同，loss 和 reward 不要求逐步单调；这组实验用于验证原生 REAL 的正负标签损失能连续更新并保存 checkpoint。

## 与第 22 节的根本差别

本节是 RLVR 回答正负分类，不理解 1～5 分的顺序距离，也不计算数字 token 的期望值；第 22 节优化回归误差，包含 CoT exploration、prediction refinement、RLOO 和 RAIL。只有英文缩写相同，目标函数完全不同。
