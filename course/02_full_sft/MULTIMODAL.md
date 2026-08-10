# 第 02 课补充：多模态全参数 SFT

本课训练后续多模态 GRPO 和 OPD 共用的学生起点。默认 `STYLE=mixed`：每个源样本只出现一次，Direct 与显式 CoT 交替，训练集仍然是 160 条而不是复制成 320 条。

## 运行命令

```bash
source ./activate.sh

SMOKE=1 STYLE=mixed bash course/02_full_sft/train_multimodal.sh

# 默认推荐：同时学习两种输出协议
EPOCHS=3 STYLE=mixed RUN_TAG=mm_mixed_e3 LEARNING_RATE=1e-5 \
  bash course/02_full_sft/train_multimodal.sh

# 消融：所有样本只训练一种风格
EPOCHS=3 STYLE=direct RUN_TAG=mm_direct_e3 \
  bash course/02_full_sft/train_multimodal.sh
EPOCHS=3 STYLE=cot RUN_TAG=mm_cot_e3 \
  bash course/02_full_sft/train_multimodal.sh
```

## mixed 数据到底是什么

mixed 不是 ms-swift 的特殊类型，它只是普通 SFT JSONL 中交替出现两种 assistant 目标：

```json
{"id":"mm-0001","style":"direct","modality":"image_text","images":["datasets/multimodal_200/images/a.jpg"],"messages":[{"role":"system","content":"直接回答。"},{"role":"user","content":"<image>\n题目……"},{"role":"assistant","content":"<answer>A</answer>"}]}
```

下一行可以是：

```json
{"id":"mm-0002","style":"cot","modality":"text_only","messages":[{"role":"system","content":"显式分析后回答。"},{"role":"user","content":"题目……"},{"role":"assistant","content":"<think>计算和判断过程……</think>\n<answer>42</answer>"}]}
```

同一行只使用一种目标风格。`style` 是教学元数据，真正决定损失的是最后一条 assistant 内容。

## “全参数”与冻结视觉编码器

脚本使用 `tuner_type=full`，完整更新语言模型参数；但默认设置：

```text
freeze_vit=true
freeze_aligner=true
```

因此准确说法是“语言模型全参数多模态 SFT”，不是视觉编码器全参数微调。图片仍会经过视觉编码器并生成视觉 token，只是视觉权重不更新。这个设计减少小样本过拟合和显存占用。

如果确实要训练视觉部分：

```bash
FREEZE_VIT=false FREEZE_ALIGNER=false \
MM_SFT_BATCH=1 LEARNING_RATE=2e-6 EPOCHS=1 STYLE=mixed \
  bash course/02_full_sft/train_multimodal.sh
```

先做单步验证并独立监控物理显存；200 条教学数据不足以证明视觉全参微调具有泛化收益。

## 关键参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `STYLE` | `mixed` | 选择 `mixed`、`direct` 或 `cot` 数据视图 |
| `tuner_type` | `full` | 更新未冻结的全部模型参数 |
| `MAX_LENGTH` | 2048 | 同时容纳视觉 token 与显式过程 |
| `MAX_PIXELS` | 1048576 | 控制视觉预处理成本 |
| `MM_SFT_BATCH` | 继承 `SFT_BATCH` | 默认正式训练为 8，可按显存调整 |
| `LEARNING_RATE` | `1e-5` | 全参语言模型学习率 |
| `save_total_limit` | 1 | 只留一个检查点，控制磁盘占用 |

## 评估重点

- 按 `modality` 分别计算纯文本、纯图像、图文混合正确率。
- 按 `style` 分别统计 Direct 和 CoT 的严格格式率。
- 检查 CoT 回答是否在 `<answer>` 前正常闭合，不能只看 loss。
- 对图像题记录“声称看不到图片”的比例；此类回答说明视觉链路或模板可能有问题。

输出位于 `outputs/02_full_sft_multimodal_<风格>_<后缀>/`。第 03、04 课默认查找最近的此类检查点，也可以通过 `STUDENT=/绝对路径` 固定实验起点。
