# SFT 与 DFT：对齐训练的共同起点

本节用相同数据、LoRA 结构和训练参数对比普通 SFT 与 DFT。后续偏好算法默认读取 SFT 检查点；把环境变量 `SFT_ADAPTER` 指向 DFT 检查点，就能做完全相同的 DFT 起点实验。

## 数据格式

文件为 `datasets/alignment_news/sft_*.jsonl`。通用格式是带标准回答的 `messages`：

```json
{"messages":[{"role":"system","content":"任务约束"},{"role":"user","content":"新闻正文"},{"role":"assistant","content":"\\boxed{体育}"}],"label":"体育"}
```

最后一条必须是 assistant。`label` 是本课程评测字段，SFT/DFT 损失只读取 assistant token，因此迁移自己的数据时可以省略它。

## 两个损失的区别

普通 SFT 对所有有效目标 token 做交叉熵。DFT 在本机 ms-swift 中由 `--enable_dft_loss true` 开启，对每个 token 的交叉熵乘以当前模型赋给正确 token 的概率，并对这个权重停止梯度。它降低模型当前极不相信的 token 对更新的支配，适合含噪或难度不均数据；但在干净小数据上也可能降低困难样本的学习速度，所以必须和 SFT 做验证集对照。

## 运行

```bash
bash course/11_sft_dft/train_sft.sh
bash course/11_sft_dft/train_dft.sh
bash course/11_sft_dft/merge_sft.sh
```

冒烟和固定步数：

```bash
SMOKE=1 bash course/11_sft_dft/train_sft.sh
STEPS=100 bash course/11_sft_dft/train_dft.sh
```

默认 `SFT_BATCH=48`、`DFT_BATCH=48`，是针对本机约 192 GiB 显存经过完整数据压力测后的配置。batch=96 和 64 在普通批次中分别只显示约 116 GiB 和 76 GiB，但都在第二轮遇到更长新闻时出现约 191.6 GiB 短暂峰值并 OOM。这来自大词表 logits 与线性注意力临时工作区，不能只看平均值。OOM 时按 40、32、24 降低。不要先增加梯度累积，因为累积只扩大有效 batch，不会提高单步 GPU 利用率。

SFT batch=48 已完整跑完三轮，采样峰值 165.17 GiB，总耗时 3 分 42 秒，最终验证 loss 0.0124、token accuracy 99.78%。
DFT 在同一 batch 下三轮耗时 1 分 20 秒（SFT 首次运行已完成大部分内核编译），峰值 174.66 GiB，最终验证 loss 0.00156、token accuracy 99.33%。DFT loss 的绝对数值受动态权重缩放，不能与 SFT loss 直接比大小。

`merge_sft.sh` 把 SFT LoRA 合并成普通 Hugging Face 模型，供 RM 全参数训练使用。正式模型写入 `models/alignment-news-sft-merged`，只增加约一个 0.8B 模型的磁盘占用；冒烟模型写在 `outputs`。

## 关键参数

- `enable_dft_loss`：是否使用 DFT 动态权重，是两个脚本唯一的算法差异。
- `lora_rank=16`、`lora_alpha=32`：LoRA 容量和缩放；本任务输出空间很小，不需要高 rank。
- `target_modules=all-linear`：对所有线性层注入 LoRA。
- `learning_rate=2e-4`：LoRA SFT 的起始学习率，可用 `SFT_LR` 或 `DFT_LR` 覆盖。
- `max_length=768`：输入和回答合计长度；截断可能删掉新闻末尾，但类别信号通常在前部。
- `EPOCHS=3`：默认训练三轮；设置 `STEPS` 后固定步数优先。

## 实验注意事项

DFT 与 DPO 不是一类方法：DFT 仍只学习标准答案，没有 chosen/rejected 比较，也没有参考模型。比较时应固定随机数据、LoRA rank、学习率和步数，并同时看验证 loss、生成格式率与分类准确率。
