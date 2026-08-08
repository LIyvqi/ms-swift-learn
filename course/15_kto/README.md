# KTO：只用好坏标签的偏好优化

KTO 不要求同一 prompt 同时拥有 chosen 与 rejected。它把每个回答标成 desirable 或 undesirable，并以参考策略为基准优化效用，适合线上点赞/点踩这类天然点反馈。

## 数据格式

文件为 `kto_*.jsonl`：

```json
{"messages":[{"role":"user","content":"请判断新闻类别……回答末尾抄写记录编号 A001"},{"role":"assistant","content":"\\boxed{体育}\n记录编号：A001"}],"label":true,"gold_label":"体育"}
```

这里 `label` 必须是 JSON 布尔值，不能写成字符串 `"true"`。`gold_label` 只是课程评测字段。自定义数据中正负回答可以来自不同 prompt，但训练/验证划分仍应按 prompt 去重，避免同一问题泄漏。

当回答空间很小时还有一个 ms-swift 特有注意点：KTO 会把 batch 内回答循环错位构造 KL 对照。如果两条新闻都只回答 `\\boxed{体育}`，错位后仍完全相同，模板会丢弃该样本。本课程让模型按 prompt 抄写唯一记录编号，从而保证每个 KL 对照不同。开放问答通常天然具有多样回答，但仍应在训练日志中搜索 `template.encode` 警告。

## 运行

```bash
bash course/11_sft_dft/train_sft.sh
bash course/15_kto/train.sh
```

## 关键参数

- `beta=0.1`：控制相对参考策略的偏移幅度。
- `desirable_weight`：正反馈损失权重。
- `undesirable_weight`：负反馈损失权重。
- `KTO_BATCH=64`：KTO 内部还会构造 KL 对照批，冒烟 batch=32 已实测约 47.6 GiB。

论文建议让 `desirable_weight × 正样本数` 与 `undesirable_weight × 负样本数` 的比例约在 1 到 4/3 之间。本课程正负各 256 条，因此两个权重都设 1。若自己的点赞样本是点踩的十倍，不能继续机械使用 1:1。

## 对比 DPO

DPO 的监督信号是同一 prompt 下的相对顺序，KTO 是单条回答的绝对好坏。成对标注质量高时优先从 DPO 开始；只有稀疏点赞/点踩时 KTO 更自然。两者都需要固定参考策略，也都应先做 SFT。

## 本机三轮结果

batch=64、384 token、left 截断共训练 3 轮/24 步，耗时 2 分 51 秒，外部显存峰值 98.04 GiB。验证损失从第 1 轮的 0.3686 降到第 3 轮的 0.2256，chosen/rejected 奖励间隔从 1.112 增至 2.674。right 截断版本曾达到 180.94 GiB 且指标更差，因为它保留了长 prompt 却破坏回答；课程只保留 left 结果。
