# 第 01 课补充：多模态 LoRA SFT

本实验使用 Qwen3.5-0.8B-Base 自带的视觉编码器，在 200 个固定源样本上分别训练无思维链 Direct 教师和显式 CoT 教师。LoRA 默认只加在语言模型部分，视觉编码器与对齐层保持冻结。

## 运行命令

```bash
source ./activate.sh

# 每种模态各取一条，先验证图片预处理和反向传播
SMOKE=1 STYLE=direct bash course/01_lora_sft/train_multimodal.sh
SMOKE=1 STYLE=cot bash course/01_lora_sft/train_multimodal.sh

# 160 条训练、40 条验证，分别训练两个教师
EPOCHS=3 STYLE=direct RUN_TAG=mm_direct_e3 \
  bash course/01_lora_sft/train_multimodal.sh
EPOCHS=3 STYLE=cot RUN_TAG=mm_cot_e3 \
  bash course/01_lora_sft/train_multimodal.sh
```

`STYLE` 必须显式区分两种目标：

- `direct`：回答只有 `<answer>最终答案</answer>`。
- `cot`：回答为 `<think>公开、可审计的推理草稿</think><answer>最终答案</answer>`。

这里的显式 CoT 是数据中提供的监督目标，不代表可以观察到模型内部不可见的全部思考状态。

## 三种输入的数据格式

多模态 SFT 和文本 SFT 的核心差别是顶层 `images` 与消息中的 `<image>` 占位符。每个占位符必须对应一张图片。

```json
{"id":"mm-demo","modality":"image_text","style":"cot","images":["datasets/multimodal_200/images/example.jpg"],"final_answer":"B","messages":[{"role":"system","content":"请分析题目并严格输出 <think>推理过程</think><answer>最终答案</answer>。"},{"role":"user","content":"<image>\n题目：观察图中曲线，选择正确选项。\n选项：\nA. ……\nB. ……"},{"role":"assistant","content":"<think>图中曲线在目标区间单调上升，因此选择 B。</think>\n<answer>B</answer>"}]}
```

- 纯文本：不写 `images`，用户消息不含 `<image>`。
- 纯图像：用户消息只有 `<image>`，完整题面已经合成在图片里。
- 图文混合：用户消息同时包含 `<image>`、题目和选项。
- SFT 最后一条必须是非空 `assistant`，这一段才参与监督损失。

完整字段说明及三类样例见 [数据集说明](../../datasets/multimodal_200/README.md)。

## 关键参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `tuner_type` | `lora` | 只保存低秩增量参数 |
| `LORA_RANK` / `LORA_ALPHA` | 16 / 32 | LoRA 容量与缩放 |
| `FREEZE_VIT` | `true` | 不给视觉编码器添加 LoRA |
| `FREEZE_ALIGNER` | `true` | 冻结视觉到语言的对齐层 |
| `MAX_LENGTH` | 2048 | 图片 token、题目和回答的总长度上限 |
| `MAX_PIXELS` | 1048576 | 限制单图像素，防止极大图片拖慢训练 |
| `MM_SFT_BATCH` | 继承 `SFT_BATCH` | 单设备 batch；冒烟时自动为 1 |
| `LEARNING_RATE` | `1e-4` | LoRA 学习率 |

默认冻结视觉部分的原因是：基础模型已经有视觉能力，而本课程只有 140 个视觉源样本；直接更新视觉编码器更容易过拟合。若研究视觉域适配，可设置 `FREEZE_VIT=false`，但应降低学习率、增加数据并单独验证灾难性遗忘。

## 自定义数据迁移

换成自己的图片分类、票据 OCR 或图表问答数据时：

1. 先按原始样本划分训练/验证，再生成 Direct 和 CoT 两个视图。
2. 图片应放在仓库持久化目录，JSONL 使用相对仓库根目录的路径。
3. 保证 `<image>` 数量和 `images` 长度一致。
4. CoT 参考过程必须来自可靠标注；不能用最终答案反向拼出看似合理的伪过程。
5. Direct 与 CoT 分开评估格式遵循率，不能只比较验证 loss。

输出位于 `outputs/01_lora_multimodal_<风格>_<后缀>/`，第 04 课会按风格查找这些 LoRA 教师。
