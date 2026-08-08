# DPO：直接偏好优化

DPO 把“偏好回答应比拒绝回答更可能出现”直接写成分类损失，不需要先训练显式奖励模型，也不需要在线采样。脚本从第 11 节 SFT LoRA 出发，同时把同一个 SFT checkpoint 固定为参考策略。

## 数据格式

文件为 `datasets/alignment_news/pairwise_*.jsonl`：

```json
{"messages":[{"role":"user","content":"请判断新闻类别……"},{"role":"assistant","content":"\\boxed{体育}"}],"rejected_response":"\\boxed{财经}","margin":1.0}
```

`messages` 的 assistant 是 chosen，`rejected_response` 是 rejected。两者必须共享完全相同的历史上下文。DPO 本身不使用本课程的 `margin` 字段；保留它是为了让同一数据也能训练 RM。

## 运行

```bash
bash course/11_sft_dft/train_sft.sh
bash course/12_dpo/train.sh
```

使用 DFT 起点：

```bash
SFT_ADAPTER=outputs/11_sft_dft/dft/某个检查点 bash course/12_dpo/train.sh
```

## 关键参数

- `beta=0.1`：隐式奖励相对参考模型的缩放；越大越保守。可用 `DPO_BETA` 覆盖。
- `ref_adapters`：固定的 SFT 参考策略。不能让它随 DPO 一起更新。
- `rpo_alpha=0.1`：在 DPO 损失中混入少量 chosen 的 SFT NLL，降低小数据退化风险；设为 0 可做原始 DPO 对照。
- `DPO_BATCH=48`：每条记录实际编码 chosen 和 rejected 两个序列。left 截断下，batch=64 单步峰值只有 128.49 GiB，但正式训练第 4 步达到 182.34 GiB 后还需申请 26.88 GiB，最终 OOM；batch=48 三轮峰值 143.20 GiB 并完整通过，因此不能用首批决定默认值。
- `ALIGNMENT_MAX_LENGTH=384`：统一对齐阶段的序列上限。768 token 压力测中，DPO batch=48 和 32 都会因长新闻、FP32 logits 与 ROCm 临时工作区在后续轮次触顶；分类任务的前 384 token 已包含标题和主要正文。迁移长文任务时需降 batch 再提高该值。
- `DPO_LR=2e-5`：偏好阶段通常比 SFT 使用更小学习率。

## 本机三轮结果

最终配置使用 batch=48、384 token、left 截断跑满 3 轮，共 18 步，耗时 2 分 4 秒；最终验证损失为 0.3592，偏好准确率为 98.96%，奖励间隔为 1.094，外部峰值 143.20 GiB。旧 delete 截断基线的 loss 更低但会直接丢弃长样本，不能与当前完整数据结果机械比较。

## 应看哪些指标

`rewards/chosen` 应逐渐高于 `rewards/rejected`，`rewards/margins` 应转正；同时要在独立生成评测中检查分类准确率，不能只追求训练 preference accuracy。若 margin 很快饱和而生成变差，优先降低学习率或训练轮数，其次提高 beta。

## 常见错误

- 未先 SFT：Base 模型尚未学会对话格式，DPO 的相对信号通常不够。
- chosen 和 rejected 长度差异过大：模型可能只学到长度偏好。本课程两者长度相同以隔离该因素。
- `adapters` 与 `ref_adapters` 指向不同起点：隐式奖励基准会发生偏移，实验不再是标准 DPO。
