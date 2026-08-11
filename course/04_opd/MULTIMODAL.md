# 第 04 课补充：多模态 OPD 在线蒸馏

本课把第 01 课的多模态 LoRA 当教师，把第 02 课的多模态全参检查点当学生。学生先针对当前文本/图片 Prompt rollout，教师再评价这些已生成 token 的分布；数据本身不预填 assistant。

## 完整依赖顺序

```text
第 01 课 Direct/CoT 多模态 LoRA 教师
                   +
第 02 课 mixed 多模态全参 SFT 学生
                   ↓
第 04 课 Direct/CoT 多模态 OPD
```

执行：

```bash
SMOKE=1 STYLE=direct bash course/04_opd/train_multimodal.sh
SMOKE=1 STYLE=cot bash course/04_opd/train_multimodal.sh

STEPS=100 STYLE=direct RUN_TAG=mm_direct_100 \
  bash course/04_opd/train_multimodal.sh
STEPS=100 STYLE=cot RUN_TAG=mm_cot_100 \
  bash course/04_opd/train_multimodal.sh
```

可显式固定依赖，避免自动选择其他实验的最新检查点：

```bash
STUDENT=/绝对路径/学生检查点 \
TEACHER_ADAPTER=/绝对路径/教师LoRA检查点 \
SMOKE=1 STYLE=cot bash course/04_opd/train_multimodal.sh
```

## 数据格式

OPD 和 GRPO 一样使用 Prompt-only 数据：

```json
{"id":"mm-demo","modality":"image_only","style":"direct","images":["datasets/multimodal_200/images/example.jpg"],"messages":[{"role":"system","content":"请直接作答，并严格输出 <answer>最终答案</answer>。用户只提供图片。"},{"role":"user","content":"<image>"}]}
```

图片占位符和 `images` 路径同时传给学生 rollout 与教师前向。若 Prompt 预填了标准 assistant，实验就不再是对学生当前生成分布的在线蒸馏。

## Direct 与 CoT 教师不能混用

- `STYLE=direct`：自动寻找 `01_lora_multimodal_direct` 教师，关闭 thinking，生成上限 256 token。128 token 冒烟实测会全部截断。
- `STYLE=cot`：自动寻找 `01_lora_multimodal_cot` 教师，开启 thinking，生成上限 1024 token。

把 Direct 教师配给 CoT Prompt，教师往往会压低过程 token 概率；反过来则可能让 Direct 学生冗长续写。脚本按风格查找教师，但手动设置 `TEACHER_ADAPTER` 时需要自己保证一致。

## 关键参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `TEACHER_KL_COEF` | 0.3 | 教师 token 分布信号权重 |
| `num_generations` | 1 | 每个 Prompt 生成一次；纯 OPD 不依赖组内相对奖励 |
| `RL_BATCH` | 4 | 在线生成 batch |
| `GENERATION_BATCH` | 同 `RL_BATCH` | 每轮集中送入 vLLM 的生成 batch |
| `LEARNING_RATE` | `5e-6` | 学生 LoRA 学习率 |
| `VLLM_MEMORY` | 0.50 | 给教师前向和训练部分保留更多空间 |
| `MM_PROCESSOR_CACHE_GB` | 2 | 缓存重复使用的图像预处理结果 |
| `freeze_vit` | true | 学生策略不更新视觉编码器 |

OPD 的教师优势可以概括为：

```text
教师权重 ×（教师对学生当前 token 的对数概率 - 学生对该 token 的对数概率）
```

教师只影响学生已经采样出的 token，并不等于把教师标准答案做普通 SFT。

## 评估与风险

- 教师 KL 下降只说明学生更像教师，不保证答案更正确。
- 视觉教师本身若不会读图，蒸馏会稳定复制错误；必须先单独评估第 01 课教师。
- CoT 生成触顶时应先检查结束格式，再考虑提高上限；无限加长会显著拖慢在线训练。
- 图片缓存提高速度但占用额外显存，迁移到小显存设备时可降低 `MM_PROCESSOR_CACHE_GB`。
- 应和第 03 课任务奖励 GRPO 做对照：一个学习教师分布，一个学习显式任务 reward。

自己的无标注图片也可做 OPD，但前提是教师在同一输入分布上可靠。没有参考答案时至少抽样人工审查，不能仅依据 KL 曲线宣称能力提升。
